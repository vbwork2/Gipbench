"""Extract the sampler state of a checkpoint produced by this training pipeline."""

import argparse
import json
from pathlib import Path


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=False)
    sampler = checkpoint.get("sampler")
    if not isinstance(sampler, dict) or not sampler:
        parser.error(f"Missing sampler state in {args.checkpoint}")
    data = {"checkpoint": args.checkpoint.name, "batch_idx_train": checkpoint["batch_idx_train"], "state": sampler}
    args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
