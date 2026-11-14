#!/usr/bin/env python3
"""Bọc model.pt của Gipformer thành dạng {"model": state_dict} mà finetune.py của icefall cần.

  python ft/wrap_ckpt.py models/gip15/model.pt models/gip15/init_ft.pt
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
if not isinstance(sd, dict):
    sys.exit(f"Không nhận ra định dạng checkpoint: {type(sd)}")
sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
torch.save({"model": sd}, dst)
print(f"{len(sd)} tensor, {n / 1e6:.1f}M tham số -> {dst}")
for k in ("decoder.embedding.weight", "joiner.output_linear.weight", "simple_lm_proj.weight"):
    if k in sd:
        print(f"  {k}: {tuple(sd[k].shape)}  (chiều đầu = vocab size, phải khớp bpe.model)")
