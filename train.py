"""Train the APSR-P main model using the released paper recipe."""

import argparse
import torch
from apsr.training.trainer import train
from apsr.utils.runtime import get_device, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wsj0_2mix.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume")
    parser.add_argument(
        "--smoke-steps",
        type=int,
        help="Limit train/validation batches and label checkpoints SMOKE_ONLY",
    )
    args = parser.parse_args()
    if args.smoke_steps is not None and args.smoke_steps < 1:
        parser.error("--smoke-steps must be positive")
    torch.set_num_threads(args.threads)
    train(
        load_config(args.config),
        args.output,
        get_device(args.device),
        args.resume,
        args.smoke_steps,
    )


if __name__ == "__main__":
    main()
