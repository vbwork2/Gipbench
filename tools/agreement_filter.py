#!/usr/bin/env python3
"""Lọc manifest CS có nhãn máy theo đồng thuận với hyp của các model ASR khác (chỉ ĐỌC hyps, không chạy model).

  python tools/agreement_filter.py --manifest data/raw/vss/vss_cs_candidates.jsonl \
      --hyps results/fw_large_v3/vss_cs/hyps.jsonl results/qwen3_asr/vss_cs/hyps.jsonl \
      --catalog configs/catalog_v0.tsv --min-agree 1 --out data/raw/vss/vss_cs_agreed.jsonl

Dùng chung cho SEA và vss_cs. Hyps do tools/asr_hyps.py ghi (định dạng SCHEMA.md mục 3).
Một span <en> của nhãn được xác nhận khi nó xuất hiện (sau normalize_hyp, strict) trong >= --min-agree file hyps.
Giữ câu khi tỉ lệ span được xác nhận >= --min-span-frac (mặc định 1.0: mọi span). Câu thiếu hyp ở một file thì
coi file đó là hyp rỗng. Ghi: <out> (câu giữ lại, thêm trường agreement) và <out>.report.md.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.agreement import agree  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyps", nargs="+", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--min-agree", type=int, default=1, help="số file hyps tối thiểu phải chứa một span")
    ap.add_argument("--min-span-frac", type=float, default=1.0, help="tỉ lệ span tối thiểu phải được xác nhận")
    ap.add_argument("--only-with-hyps", action="store_true", help="chỉ xét câu có hyp ở mọi file (chạy thử một phần)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    hyps = [{h["id"]: h.get("hyp", "") for h in read_jsonl(p)} for p in args.hyps]
    rows = read_jsonl(args.manifest)
    if args.only_with_hyps:
        rows = [r for r in rows if all(r["id"] in h for h in hyps)]
    kept, n_hours, k_hours = [], 0.0, 0.0
    span_tot, span_ok = Counter(), Counter()
    examples = {"giữ": [], "bỏ": []}
    for r in rows:
        res = agree(r["text_tagged"], [h.get(r["id"], "") for h in hyps], cat, args.min_agree, args.min_span_frac)
        n_hours += float(r["duration"]) / 3600
        for s, v in zip(res["spans"], res["confirmed"]):
            span_tot[s] += 1
            span_ok[s] += v >= args.min_agree
        key = "giữ" if res["keep"] else "bỏ"
        if len(examples[key]) < 8:
            examples[key].append((r["id"], r["text_tagged"], [h.get(r["id"], "") for h in hyps], res["confirmed"]))
        if res["keep"]:
            r["agreement"] = {"confirmed": res["confirmed"], "n_hyps": len(hyps), "min_agree": args.min_agree}
            kept.append(r)
            k_hours += float(r["duration"]) / 3600
    write_jsonl(args.out, kept)
    md = [f"# Lọc đồng thuận — {args.manifest}", "",
          f"Hyps: {', '.join(f'`{p}`' for p in args.hyps)}. --min-agree {args.min_agree}, --min-span-frac {args.min_span_frac}.", "",
          f"Giữ **{len(kept)} / {len(rows)}** câu ({k_hours:.2f} / {n_hours:.2f} h, "
          f"{100 * len(kept) / max(1, len(rows)):.1f}%).", "",
          "## Tỉ lệ xác nhận theo thuật ngữ (20 thuật ngữ nhiều nhất)", "", "| Thuật ngữ | Số span | Được xác nhận |", "| --- | --- | --- |"]
    md += [f"| {t} | {c} | {span_ok[t]} |" for t, c in span_tot.most_common(20)]
    for key, ex in examples.items():
        md += ["", f"## Ví dụ câu {key}", "", "| id | nhãn | hyps | số hyp xác nhận từng span |", "| --- | --- | --- | --- |"]
        md += [f"| {i} | {t} | {' ‖ '.join(h)} | {c} |" for i, t, h, c in ex]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Giữ {len(kept)}/{len(rows)} câu ({k_hours:.2f}/{n_hours:.2f} h) -> {args.out}; báo cáo {args.out}.report.md")


if __name__ == "__main__":
    main()
