"""Measure P whole-utterance throughput and neural MACs with explicit scope."""

import argparse
import platform
import time
from pathlib import Path

import torch
from apsr import APSR
from apsr.utils.profile import estimate_tse_macs
from apsr.utils.runtime import get_device, sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--mixture-seconds", type=float, default=4.0)
    p.add_argument("--enrollment-seconds", type=float, default=3.0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if (
        min(a.mixture_seconds, a.enrollment_seconds) <= 0.016
        or a.repeats < 1
        or a.warmup < 0
    ):
        p.error("Invalid durations/counts")
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(a.threads)
    device = get_device(a.device)
    model = APSR.from_pretrained(a.checkpoint, device)
    generator = torch.Generator().manual_seed(43)
    mix = torch.randn(1, round(8000 * a.mixture_seconds), generator=generator) * 0.05
    enrollment = (
        torch.randn(1, round(8000 * a.enrollment_seconds), generator=generator) * 0.05
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def call():
        # Deliberately include input host->device and output device->host transfer.
        return model(mix.to(device), enrollment.to(device)).cpu()

    elapsed = []
    with torch.inference_mode():
        for _ in range(a.warmup):
            call()
        synchronize()
        for _ in range(a.repeats):
            synchronize()
            begin = time.perf_counter()
            call()
            synchronize()
            elapsed.append(time.perf_counter() - begin)
    macs = estimate_tse_macs(
        model,
        sample_rate=8000,
        mix_seconds=a.mixture_seconds,
        enroll_seconds=a.enrollment_seconds,
        device=device,
    )
    actual_seconds = mix.shape[-1] / 8000
    report = dict(
        model="APSR",
        refinement_rounds=4,
        batch_size=1,
        dtype="float32",
        checkpoint_sha256=sha256(a.checkpoint),
        device=str(device),
        device_name=torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        host_threads=a.threads,
        torch_version=str(torch.__version__),
        mixture_seconds=actual_seconds,
        enrollment_seconds=enrollment.shape[-1] / 8000,
        warmup=a.warmup,
        repeats=a.repeats,
        seconds_per_pass=elapsed,
        rtf_per_pass=[v / actual_seconds for v in elapsed],
        rtf_mean=sum(elapsed) / len(elapsed) / actual_seconds,
        scope="Whole utterance, enrollment processing and input/output transfers included; excludes disk I/O and model loading",
        input="Deterministic synthetic noise; not the paper duration-figure sample set",
        interpretation="Throughput, not frame deadline or end-to-end streaming latency",
        macs=macs,
    )
    write_json(a.output, report)
    print(f"RTF {report['rtf_mean']:.6f}; MAC/s {macs['macs_per_second'] / 1e9:.6f} G")


if __name__ == "__main__":
    main()
