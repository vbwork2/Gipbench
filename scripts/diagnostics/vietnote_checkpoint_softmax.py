"""Read the pinned downsampling bias and emit its CPU FP32 softmax bytes."""

import json
import sys

import torch


if __name__ == "__main__":
    torch.set_num_threads(1)
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=True, mmap=True)
    key = "encoder.encoders.3.downsample.bias"
    bias = state["model"][key]
    if bias.dtype != torch.float32 or tuple(bias.shape) != (8,):
        raise ValueError(f"Unexpected checkpoint bias: {key}")
    weights = bias.softmax(dim=0, dtype=torch.float32)
    print(json.dumps({"key": key, "torch": str(torch.__version__), "platform": sys.platform,
                      "fp32_le_hex": weights.numpy().astype("<f4").tobytes().hex()}))
