#!/usr/bin/env python3
"""Dựng dev CS của E3 v1 và bản ViMedCSS train có audio tuyệt đối (trước đây là heredoc stage 0 của ft/run_e3_v1*.sh).

  python -m gipbench.dataset.release.build_dev_cs --vimedcss-dev data/raw/vimedcss_full/dev_vimedcss.jsonl \
      --vss-dev data/e3_v1/dev_cs_vss.jsonl --vimedcss-train data/raw/vimedcss_full/train_vimedcss.jsonl \
      --audio-base /mnt/d/code/ZAIC/gipbench/data/raw/vimedcss_full/ --out-dir data/e3_v1

dev_cs.jsonl = --n-vimedcss câu ViMedCSS validation (xáo theo --seed) + vss_cs held-out (prep_e3_v1), bỏ câu ngắn hơn
--min-dur giây (ViMedCSS validation có câu audio rỗng 0 s làm sherpa-onnx lỗi). train_vimedcss_abs.jsonl = ViMedCSS train
với audio = --audio-base + đường dẫn gốc (nối chuỗi, như bản cũ). Logic giữ nguyên bản heredoc.
"""
import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vimedcss-dev", required=True)
    ap.add_argument("--vss-dev", required=True)
    ap.add_argument("--vimedcss-train", required=True)
    ap.add_argument("--audio-base", required=True, help="tiền tố nối vào audio của ViMedCSS (kết thúc bằng /)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-vimedcss", type=int, default=600)
    ap.add_argument("--min-dur", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = args.audio_base
    rows = [json.loads(l) for l in open(args.vimedcss_dev, encoding="utf-8")]
    random.Random(args.seed).shuffle(rows)
    o = [dict(r, audio=base + r["audio"]) for r in rows[:args.n_vimedcss]]
    o += [json.loads(l) for l in open(args.vss_dev, encoding="utf-8")]
    o = [r for r in o if float(r["duration"]) >= args.min_dur]
    with open(out / "dev_cs.jsonl", "w", encoding="utf-8") as f:
        for r in o:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("dev_cs:", len(o), "câu", round(sum(float(r["duration"]) for r in o) / 3600, 2), "h")
    with open(out / "train_vimedcss_abs.jsonl", "w", encoding="utf-8") as f:
        for l in open(args.vimedcss_train, encoding="utf-8"):
            r = json.loads(l)
            f.write(json.dumps(dict(r, audio=base + r["audio"]), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
