#!/usr/bin/env python3
"""Đường học từ log finetune_cs.py (có thể gồm nhiều đoạn resume của ft/run_e3_v1.sh).

  python ft/learning_curve.py --log data/e3_v1/finetune.log --out reports/e3_v1_learning_curve.md

Step toàn cục = --start-batch của đoạn (dòng "== đoạn mới: ... --start-batch N", 0 nếu không có) + "batch K" của icefall
(đếm lại từ 0 sau mỗi lần resume). Ghi bảng: step, tot_loss train (trung bình trượt của icefall), loss valid từng tập.
"""
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--every", type=int, default=1000, help="lấy một dòng train mỗi N step")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    seg, step, rows, valid = 0, 0, {}, {}
    r_tr = re.compile(r"batch (\d+), .*tot_loss\[loss=([\d.]+)")
    r_va = re.compile(r"Validation on (\S+?)(?:_cuts_raw)?\.jsonl\.gz: .*validation: loss=([\d.]+)")
    for line in open(args.log, encoding="utf-8", errors="replace"):
        if "== đoạn mới:" in line:
            m = re.search(r"--start-batch (\d+)", line)
            seg = int(m.group(1)) if m else 0
            continue
        m = r_tr.search(line)
        if m:
            step = seg + int(m.group(1))
            rows.setdefault(step - step % args.every, float(m.group(2)))
            continue
        m = r_va.search(line)
        if m:
            valid.setdefault(step, {})[m.group(1)] = float(m.group(2))
    sets = sorted({k for v in valid.values() for k in v})
    L = ["| Step | tot_loss train | " + " | ".join(f"loss valid {s}" for s in sets) + " |",
         "|" + " --- |" * (2 + len(sets))]
    for s in sorted(set(rows) | set(valid)):
        tr = f"{rows[s]:.4f}" if s in rows else ""
        L.append(f"| {s} | {tr} | " + " | ".join(str(valid.get(s, {}).get(k, "")) for k in sets) + " |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
