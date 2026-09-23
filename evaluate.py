"""Evaluate validation-selected APSR checkpoints on paired target queries."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apsr import APSR
from apsr.data.loading import build_dataset
from apsr.metrics import objective as metrics
from apsr.utils.runtime import get_device, sha256, write_json


def evaluate(
    checkpoint,
    manifest,
    output,
    device="cpu",
    workers=0,
    primary_only=False,
    limit=None,
):
    if not primary_only and any(
        x is None for x in (metrics.mir_eval, metrics.pesq, metrics.stoi)
    ):
        raise RuntimeError(
            "Full metrics require requirements-eval.txt; use --primary-only for SI-SDRi/TCR"
        )
    out = Path(output)
    if (out / "summary.json").exists() or (out / "metrics.csv").exists():
        raise FileExistsError("Choose a new evaluation output directory")
    dataset = build_dataset(manifest)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        collate_fn=dataset.collate_fn,
    )
    model = APSR.from_pretrained(checkpoint, device)
    rows = []
    seen = set()
    for idx, (mix, target, enrollment) in enumerate(
        tqdm(loader, total=min(len(dataset), limit or len(dataset)))
    ):
        if limit is not None and idx >= limit:
            break
        if mix.shape[0] != 2 or not torch.equal(mix[0], mix[1]):
            raise ValueError("Each mixture must have exactly two paired target queries")
        with torch.inference_mode():
            estimate = model(mix.to(device), enrollment.to(device))[:, 0].cpu().numpy()
        for case_idx, entry_idx in enumerate(dataset.entry_groups[idx]):
            key, mix_path, _, _ = dataset.entries[entry_idx]
            if key in seen:
                raise ValueError(f"Duplicate query: {key}")
            seen.add(key)
            est, ref, mixture = (
                estimate[case_idx],
                target[case_idx].numpy(),
                mix[case_idx].numpy(),
            )
            if not all(np.isfinite(x).all() for x in (est, ref, mixture)):
                raise FloatingPointError(f"Nonfinite waveform in query {key}")
            si = metrics.compute_si_sdr_np(est, ref)
            baseline = metrics.compute_si_sdr_np(mixture, ref)
            other = metrics.compute_si_sdr_np(est, target[1 - case_idx].numpy())
            row = dict(
                idx=idx,
                case_idx=case_idx,
                key=key,
                mixture_id=Path(mix_path).stem,
                seconds=len(est) / 8000,
                si_sdr=si,
                si_sdr_mix=baseline,
                si_sdr_i=si - baseline,
                interferer_si_sdr=other,
                target_margin_db=si - other,
                target_confusion_rate=float(si - other <= 0),
            )
            row.update(
                sdr="PENDING",
                sdr_mix="PENDING",
                sdr_i="PENDING",
                pesq="PENDING",
                estoi="PENDING",
            )
            if not primary_only:
                row["sdr"] = metrics.compute_bss_eval_sdr_np(est, ref)
                row["sdr_mix"] = metrics.compute_bss_eval_sdr_np(mixture, ref)
                row["sdr_i"] = row["sdr"] - row["sdr_mix"]
                row["pesq"] = metrics.safe_pesq(8000, ref, est)
                row["estoi"] = metrics.safe_estoi(8000, ref, est)
            if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
                raise FloatingPointError(
                    f"Nonfinite metric for query {key}; no complete summary written"
                )
            rows.append(row)
    if not rows:
        raise ValueError("No evaluation queries")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [
        "si_sdr",
        "si_sdr_mix",
        "si_sdr_i",
        "target_margin_db",
        "target_confusion_rate",
        "sdr",
        "sdr_mix",
        "sdr_i",
        "pesq",
        "estoi",
    ]
    means = {
        name: float(np.mean([r[name] for r in rows]))
        if rows[0][name] != "PENDING"
        else "PENDING"
        for name in metric_names
    }
    summary = dict(
        model="APSR",
        sample_rate=8000,
        num_mixtures=len(rows) // 2,
        num_queries=len(rows),
        metrics=means,
        confusion_count=int(sum(r["target_confusion_rate"] for r in rows)),
        checkpoint_sha256=sha256(checkpoint),
        checkpoint_info=model._checkpoint_info,
        metrics_sha256=sha256(out / "metrics.csv"),
        manifest_sha256={
            n: sha256(Path(manifest) / n) for n in ["mix.scp", "ref.scp", "aux.scp"]
        },
        scope="SUBSET" if limit is not None else "ALL_MANIFEST_QUERIES",
        torch_version=str(torch.__version__),
        device=str(device),
        loader_batch_size=1,
        model_batch_size=2,
        enrollment="Full enrollment, zero-padded within each paired mixture before model normalization",
        selection="External validation selection; this evaluator does not select checkpoints",
        aggregation="Unweighted mean over queries; TCR and eSTOI are fractions",
    )
    write_json(out / "summary.json", summary)
    print(means)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--limit", type=int, help="Mixture count; labels output SUBSET")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    torch.set_num_threads(args.threads)
    evaluate(
        args.checkpoint,
        args.manifest,
        args.output,
        get_device(args.device),
        args.workers,
        args.primary_only,
        args.limit,
    )


if __name__ == "__main__":
    main()
