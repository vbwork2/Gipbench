"""Read the pinned downsampling bias and emit its CPU FP32 softmax bytes."""

import json
import sys
from pathlib import Path

import torch


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from gipbench.model.vietnote import load_checkpoint

    torch.set_num_threads(1)
    state = load_checkpoint(sys.argv[1])
    key = "encoder.encoders.3.downsample.bias"
    bias = state["model"][key]
    if not bias.is_floating_point() or tuple(bias.shape) != (8,) or not torch.isfinite(bias).all():
        raise ValueError(f"Unexpected checkpoint bias: {key}")
    weights = bias.softmax(dim=0, dtype=torch.float32)
    if len(sys.argv) == 3:
        torch.save({"model": state["model"]}, sys.argv[2])
    print(json.dumps({"key": key, "torch": str(torch.__version__), "platform": sys.platform,
                      "fp32_le_hex": weights.numpy().astype("<f4").tobytes().hex()}))
