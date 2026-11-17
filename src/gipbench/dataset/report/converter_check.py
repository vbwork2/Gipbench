#!/usr/bin/env python3
"""Tóm tắt manifest do converter ghi ra: số câu, số giờ, người nói, phân bố auto_tag.confidence, cờ, câu mẫu.

  python -m gipbench.dataset.report.converter_check --manifests data/raw/ggroup/*.jsonl data/raw/vimedcss/*.jsonl \
      --samples 5 --seed 0 --out reports/converter_check_tables.md

Chỉ đọc file, không cần mạng. Câu mẫu lấy ngẫu nhiên (cố định theo --seed) trong mỗi manifest.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path



def summarize(path: Path, n_samples: int, rnd: random.Random):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    conf = Counter((r.get("auto_tag") or {}).get("confidence", "(không có)") for r in rows)
    flags = Counter(f for r in rows for f in (r.get("auto_tag") or {}).get("flags", []))
    hours = sum(float(r.get("duration", 0)) for r in rows) / 3600
    text_key = "text_tagged" if any("text_tagged" in r for r in rows) else "text"
    samples = rnd.sample(rows, min(n_samples, len(rows)))
    return {"file": str(path), "rows": len(rows), "hours": hours, "speakers": len({r.get("speaker") for r in rows}),
            "sources": len({r.get("source") for r in rows}), "domains": Counter(r.get("domain", "unk") for r in rows),
            "conf": conf, "flags": flags, "text_key": text_key,
            "samples": [(r["id"], r.get(text_key, "")) for r in samples]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rnd = random.Random(args.seed)
    res = [summarize(Path(p), args.samples, rnd) for p in args.manifests]

    def pct(c, n):
        return ", ".join(f"{k} {v} ({100 * v / n:.0f}%)" for k, v in c.most_common()) if n else "–"
    lines = ["| Manifest | Câu | Giờ | Người nói | Source | Domain | auto_tag.confidence |", "|" + " --- |" * 7]
    for r in res:
        lines.append(f"| `{r['file']}` | {r['rows']} | {r['hours']:.3f} | {r['speakers']} | {r['sources']} | "
                     f"{', '.join(f'{k} {v}' for k, v in r['domains'].most_common())} | {pct(r['conf'], r['rows'])} |")
    lines.append("")
    for r in res:
        lines += [f"### `{r['file']}`", ""]
        if r["flags"]:
            lines += ["Cờ: " + ", ".join(f"`{k}` {v}" for k, v in r["flags"].most_common(12)), ""]
        lines += [f"{len(r['samples'])} câu mẫu (`{r['text_key']}`):", ""]
        lines += [f"- `{i}`: {t}" for i, t in r["samples"]]
        lines.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã ghi {args.out} ({len(res)} manifest)")


if __name__ == "__main__":
    main()
