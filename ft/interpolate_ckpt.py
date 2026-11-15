#!/usr/bin/env python3
"""Trộn trọng số model gốc và model fine-tune (WiSE-FT): theta = (1 - alpha) * gốc + alpha * FT.

  python ft/interpolate_ckpt.py --base models/gip15/init_ft.pt --ft exp/e3_ft/epoch-8.pt \
      --alphas 0.3 0.5 0.7 --out-dir exp/e3_ft_wise

alpha = 0 là model gốc, 1 là model FT. Mỗi alpha ghi một checkpoint epoch-<N>.pt riêng trong thư mục con
alpha_<a>/ để export-onnx.py dùng được ngay (--epoch N --avg 1 --use-averaged-model 0).
Chọn alpha trên dev: điểm có CS-WER thấp nhất mà WER tiếng Việt thuần không tăng.
Chỉ dùng khi hai model cùng vocab (không dùng cho E6 mở rộng BPE).
"""
import argparse
from pathlib import Path

import torch


def state(path, keys=("model_avg", "model")):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for key in keys:  # mặc định ưu tiên bản trung bình nếu có
        if isinstance(ck, dict) and key in ck and isinstance(ck[key], dict):
            return {k[7:] if k.startswith("module.") else k: v for k, v in ck[key].items()}
    return ck


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--ft", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epoch", type=int, default=1, help="số epoch dùng trong tên file xuất")
    ap.add_argument("--ft-key", choices=["auto", "model", "model_avg"], default="auto",
                    help="khoá state dict của checkpoint FT (auto: model_avg nếu có, không thì model). Dùng model để "
                         "khớp với export-onnx --use-averaged-model 0")
    args = ap.parse_args()
    b = state(args.base)
    f = state(args.ft, ("model_avg", "model") if args.ft_key == "auto" else (args.ft_key,))
    miss = [k for k in f if k not in b]
    bad = [k for k in f if k in b and hasattr(f[k], "shape") and b[k].shape != f[k].shape]
    if bad:
        raise SystemExit(f"Khác kích thước ở {len(bad)} tensor (vd {bad[0]}): hai model khác vocab/kiến trúc.")
    for a in args.alphas:
        out = {}
        for k, v in f.items():
            if k in b and torch.is_tensor(v) and v.is_floating_point():
                out[k] = (1 - a) * b[k].to(v.dtype) + a * v
            else:
                out[k] = v  # bộ đếm, tham số chỉ có ở FT
        d = Path(args.out_dir) / f"alpha_{a:g}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save({"model": out}, d / f"epoch-{args.epoch}.pt")
        print(f"alpha={a:g}: {len(out)} tensor -> {d}/epoch-{args.epoch}.pt")
    if miss:
        print(f"Lưu ý: {len(miss)} tensor chỉ có ở model FT, giữ nguyên.")


if __name__ == "__main__":
    main()
