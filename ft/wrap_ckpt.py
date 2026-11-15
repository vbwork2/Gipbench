#!/usr/bin/env python3
"""Bọc model.pt của Gipformer thành dạng {"model": state_dict} mà finetune.py của icefall cần.

  python ft/wrap_ckpt.py models/gip15/model.pt models/gip15/init_ft.pt [--drop-ctc]

--drop-ctc: bỏ head CTC (ctc_output.*) để FT với --use-ctc 0. Gipformer 1.5 có head CTC; FT fp16 batch nhỏ với CTC
  (icefall: reduction="sum", không zero_infinity) làm grad_scale tụt về 0 (A4, 16/11/2026). Decode sherpa-onnx chỉ
  dùng transducer nên bỏ CTC không ảnh hưởng đường chạy production.
"""
import sys

import torch

args = [a for a in sys.argv[1:] if not a.startswith("--")]
drop_ctc = "--drop-ctc" in sys.argv[1:]
if len(args) != 2:
    sys.exit(__doc__)
src, dst = args
ck = torch.load(src, map_location="cpu", weights_only=False)
sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
if not isinstance(sd, dict):
    sys.exit(f"Không nhận ra định dạng checkpoint: {type(sd)}")
sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
if drop_ctc:
    gone = [k for k in sd if k.startswith("ctc_output.")]
    sd = {k: v for k, v in sd.items() if not k.startswith("ctc_output.")}
    print(f"--drop-ctc: bỏ {len(gone)} tensor {gone}")
n = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
torch.save({"model": sd}, dst)
print(f"{len(sd)} tensor, {n / 1e6:.1f}M tham số -> {dst}")
for k in ("decoder.embedding.weight", "joiner.output_linear.weight", "simple_lm_proj.weight"):
    if k in sd:
        print(f"  {k}: {tuple(sd[k].shape)}  (chiều đầu = vocab size, phải khớp bpe.model)")
