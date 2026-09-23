"""Extract a target speaker from a mixture and a reference utterance."""

import argparse
from pathlib import Path

import soundfile as sf
import torch

from apsr import APSR
from apsr.utils.runtime import get_device, read_audio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixture", required=True)
    parser.add_argument("--enrollment", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = get_device(args.device)
    model = APSR.from_pretrained(args.checkpoint, device)
    mix = read_audio(args.mixture).unsqueeze(0).to(device)
    enrollment = read_audio(args.enrollment).unsqueeze(0).to(device)
    with torch.inference_mode():
        estimate = model(mix, enrollment)[0, 0].cpu().numpy()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, estimate, 8000, subtype="FLOAT")
    print(f"Saved {len(estimate) / 8000:.3f} seconds to {destination}")


if __name__ == "__main__":
    main()
