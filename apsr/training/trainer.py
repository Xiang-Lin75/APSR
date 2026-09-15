"""Single-device P training with paired targets and validation-only selection."""

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apsr import APSRP
from apsr.data.loading import build_dataset
from apsr.losses.objective import TrainingObjective
from apsr.losses.operators import (
    compute_si_snr,
    clip_grad_norm_with_diagnostics,
    seed_dataloader_worker,
)
from apsr.utils.runtime import write_json, sha256


def capture_rng():
    np_state = np.random.get_state()
    return dict(
        python=random.getstate(),
        numpy=[
            np_state[0],
            np_state[1].tolist(),
            np_state[2],
            np_state[3],
            np_state[4],
        ],
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def validate(model, loader, device, limit=None):
    model.eval()
    scores = []
    for step, (mix, target, enrollment) in enumerate(
        tqdm(loader, desc="validation"), 1
    ):
        mix, target, enrollment = [x.to(device) for x in (mix, target, enrollment)]
        estimate = model(mix, enrollment)[:, 0]
        # Preserve the original float32 validation arithmetic and paired mean.
        score = (
            compute_si_snr(estimate, target).mean() - compute_si_snr(mix, target).mean()
        )
        if not torch.isfinite(score):
            raise FloatingPointError("Nonfinite validation SI-SDRi")
        scores.append(float(score))
        if limit is not None and step >= limit:
            break
    if not scores:
        raise ValueError("Empty validation set")
    return sum(scores) / len(scores)


def train(config, output, device, resume=None, smoke_steps=None):
    cfg = copy.deepcopy(config)
    out = Path(output)
    if out.exists() and any(out.iterdir()) and resume is None:
        raise FileExistsError("Use a new run directory or explicitly --resume")
    out.mkdir(parents=True, exist_ok=True)
    seed = int(cfg["trainer"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = APSRP()
    initialization = Path(cfg["trainer"]["initialization"])
    if seed != 43:
        raise ValueError("The released paper-initialization recipe is seed 43")
    if sha256(initialization) != cfg["trainer"]["initialization_sha256"]:
        raise ValueError("Paper initialization checksum mismatch")
    initial = torch.load(initialization, map_location="cpu", weights_only=True)
    model.load_state_dict(initial["model"])
    torch.set_rng_state(initial["torch_rng_state"])
    model.to(device)
    train_data = build_dataset(
        cfg["data"]["train_manifest"], True, cfg["data"]["dynamic_source_rebalance"]
    )
    val_data = build_dataset(cfg["data"]["validation_manifest"])
    train_gen = torch.Generator().manual_seed(seed)
    val_gen = torch.Generator().manual_seed(seed + 10000)
    workers = int(cfg["loader"]["workers"])
    val_workers = int(cfg["loader"]["validation_workers"])
    train_loader = DataLoader(
        train_data,
        batch_size=cfg["loader"]["batch_size"],
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=train_data.collate_fn,
        worker_init_fn=seed_dataloader_worker,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=val_workers > 0,
        collate_fn=val_data.collate_fn,
        worker_init_fn=seed_dataloader_worker,
        generator=val_gen,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["optimizer"]["lr"],
        weight_decay=cfg["optimizer"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg["scheduler"]["factor"],
        patience=cfg["scheduler"]["patience"],
    )
    objective = TrainingObjective(**cfg["loss"]).to(device)
    start, best, best_epoch, bad = 1, -float("inf"), None, 0
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            saved.get("format") != "apsr-p-training-v1"
            or saved.get("smoke_steps") != smoke_steps
        ):
            raise ValueError(
                "Resume requires a matching public trainer checkpoint and run scope"
            )
        previous = saved["config"]
        # Extending total epochs is permitted; all numerical recipe fields must match.
        current_check = copy.deepcopy(cfg)
        previous_check = copy.deepcopy(previous)
        current_check["trainer"].pop("epochs")
        previous_check["trainer"].pop("epochs")
        if current_check != previous_check:
            raise ValueError("Resume recipe differs from saved checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start = saved["epoch"] + 1
        best = saved["best_score"]
        best_epoch = saved["best_epoch"]
        bad = saved["no_improve_count"]
        train_gen.set_state(saved["train_generator"])
        val_gen.set_state(saved["validation_generator"])
        restore_rng(saved["rng"])
    protocol = {
        "scope": "SMOKE_ONLY" if smoke_steps else "TRAINING",
        "loader_batch_size": cfg["loader"]["batch_size"],
        "target_query_batch_size": 2 * cfg["loader"]["batch_size"],
        "manifest_sha256": {
            split: {
                name: sha256(Path(cfg["data"][split + "_manifest"]) / name)
                for name in ["mix.scp", "ref.scp", "aux.scp"]
            }
            for split in ["train", "validation"]
        },
    }
    if resume and saved.get("manifest_sha256") != protocol["manifest_sha256"]:
        raise ValueError("Resume dataset manifests differ from the saved checkpoint")
    write_json(out / "resolved_config.json", cfg)
    write_json(out / "run_manifest.json", protocol)
    stop_reason = "epoch_limit"
    if bad >= cfg["trainer"]["early_stop_patience"]:
        raise ValueError("Checkpoint already met the configured stopping rule")
    for epoch in range(start, int(cfg["trainer"]["epochs"]) + 1):
        model.train()
        total = 0.0
        count = 0
        for step, (mix, target, enrollment) in enumerate(
            tqdm(train_loader, desc=f"train {epoch}"), 1
        ):
            mix, target, enrollment = [x.to(device) for x in (mix, target, enrollment)]
            optimizer.zero_grad(set_to_none=True)
            estimate = model(mix, enrollment)
            loss, components = objective(model, estimate, target)
            loss.backward()
            clip_grad_norm_with_diagnostics(
                model, cfg["trainer"]["clip_grad_norm"], error_if_nonfinite=True
            )
            optimizer.step()
            total += float(loss.detach())
            count += 1
            if smoke_steps is not None and step >= smoke_steps:
                break
        score = validate(model, val_loader, device, smoke_steps)
        scheduler.step(score)
        improved = score > best
        if improved:
            best, best_epoch, bad = score, epoch, 0
        else:
            bad += 1
        payload = dict(
            format="apsr-p-training-v1",
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            epoch=epoch,
            score=score,
            best_score=best,
            best_epoch=best_epoch,
            no_improve_count=bad,
            config=cfg,
            rng=capture_rng(),
            train_generator=train_gen.get_state(),
            validation_generator=val_gen.get_state(),
            smoke_steps=smoke_steps,
            manifest_sha256=protocol["manifest_sha256"],
        )
        atomic_save(payload, out / "latest.pt")
        if improved:
            atomic_save(payload, out / "best.pt")
        record = dict(
            epoch=epoch,
            train_loss=total / count,
            validation_si_sdri=score,
            best_epoch=best_epoch,
            best_si_sdri=best,
            learning_rate=optimizer.param_groups[0]["lr"],
            no_improve_count=bad,
            scope=protocol["scope"],
        )
        with (out / "epochs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if bad >= cfg["trainer"]["early_stop_patience"]:
            stop_reason = "validation_patience"
            break
    write_json(
        out / "training_status.json",
        dict(
            status="STOPPED",
            reason=stop_reason,
            best_epoch=best_epoch,
            best_score=best,
            scope=protocol["scope"],
        ),
    )
