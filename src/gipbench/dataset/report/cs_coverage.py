#!/usr/bin/env python3
"""Độ phủ domain của dữ liệu CODE-SWITCH (câu có span <en>) trong các pool train, so với mục tiêu (TODO 2e, audit R10).

  python -m gipbench.dataset.report.cs_coverage --pools pool/vimedcss_train@v2 pool/vss_cs@e3_v1 \
      --out reports/data/cs_coverage.md

Mỗi pool × domain: số câu CS, giờ CS, số nguồn (video / người nói theo core.decontam.source_key, không có thì speaker),
số thuật ngữ khác nhau, tỉ lệ giờ CS. Domain lấy từ trường `domain` (nhãn dataset hoặc suy từ text: `domain_source`).
Bảng "thiếu hụt" so với mục tiêu: --target-domains (mặc định 4 domain ngoài y khoa của C1 / G-Group) cần tối thiểu
--min-hours giờ CS và --min-sources nguồn mỗi domain (ngưỡng là đề xuất của dự án, không phải chuẩn ngành; ghi trong báo cáo).
KHÔNG đọc bench: chỉ thống kê dữ liệu train (thuật ngữ bench không được dùng để chọn dữ liệu, audit R05).
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.decontam import source_key
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import basic_tokens, parse_tagged


def cs_terms(text_tagged: str):
    return [" ".join(basic_tokens(s)) for s, en in parse_tagged(text_tagged or "") if en and basic_tokens(s)]


def coverage(rows):
    """{domain: {n, h, src, terms, dsrc}} trên các câu CS."""
    agg = defaultdict(lambda: {"n": 0, "h": 0.0, "src": set(), "terms": Counter(), "dsrc": Counter()})
    for r in rows:
        ts = cs_terms(r.get("text_tagged"))
        if not ts:
            continue
        a = agg[r.get("domain") or "other"]
        a["n"] += 1
        a["h"] += float(r.get("duration") or 0) / 3600
        a["src"].add(source_key(r) or r.get("speaker") or r.get("source"))
        a["terms"].update(ts)
        a["dsrc"][r.get("domain_source") or "dataset"] += 1
    return agg


def load(ref: str):
    p = Path(ref)
    return read_jsonl(str(p)) if p.exists() else read_jsonl(str(R.resolve(ref).manifest))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pools", nargs="+", required=True, help="release hoặc manifest train")
    ap.add_argument("--target-domains", nargs="+", default=["tech", "finance", "edu", "pubadmin"])
    ap.add_argument("--min-hours", type=float, default=2.0, help="giờ CS tối thiểu mỗi domain mục tiêu (đề xuất)")
    ap.add_argument("--min-sources", type=int, default=30, help="số nguồn tối thiểu mỗi domain mục tiêu (đề xuất)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    md = ["# Độ phủ domain dữ liệu code-switch trong train", "",
          "Chỉ tính câu có span `<en>`. Nguồn = video YouTube / người nói (`source_key`), không có thì `speaker`.", ""]
    total = defaultdict(lambda: {"n": 0, "h": 0.0, "src": set(), "terms": set()})
    res = {}
    for ref in args.pools:
        agg = coverage(load(ref))
        th = sum(a["h"] for a in agg.values()) or 1
        md += [f"## `{ref}`", "", "| Domain | Câu CS | Giờ CS | % giờ | Nguồn | Thuật ngữ | Nguồn domain | Top 5 thuật ngữ |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        res[ref] = {}
        for d, a in sorted(agg.items(), key=lambda x: -x[1]["h"]):
            md.append(f"| {d} | {a['n']} | {a['h']:.2f} | {100 * a['h'] / th:.1f}% | {len(a['src'])} | {len(a['terms'])} | "
                      + ", ".join(f"{k} {v}" for k, v in a["dsrc"].most_common()) + " | "
                      + ", ".join(f"{t} {c}" for t, c in a["terms"].most_common(5)) + " |")
            t = total[d]
            t["n"] += a["n"]
            t["h"] += a["h"]
            t["src"] |= a["src"]
            t["terms"] |= set(a["terms"])
            res[ref][d] = {"cs_rows": a["n"], "cs_hours": round(a["h"], 3), "sources": len(a["src"]), "terms": len(a["terms"])}
        md.append("")
    md += ["## Thiếu hụt so với mục tiêu", "",
           f"Mục tiêu (đề xuất, chưa phải chuẩn): mỗi domain trong {', '.join(args.target_domains)} có >= {args.min_hours:g} h CS "
           f"và >= {args.min_sources} nguồn khác nhau, gộp mọi pool trên.", "",
           "| Domain | Câu CS | Giờ CS | Nguồn | Thiếu giờ | Thiếu nguồn | Đạt |", "| --- | --- | --- | --- | --- | --- | --- |"]
    gap = {}
    for d in args.target_domains:
        t = total.get(d, {"n": 0, "h": 0.0, "src": set()})
        mh, ms = max(0.0, args.min_hours - t["h"]), max(0, args.min_sources - len(t["src"]))
        gap[d] = {"cs_rows": t["n"], "cs_hours": round(t["h"], 3), "sources": len(t["src"]), "missing_hours": round(mh, 3),
                  "missing_sources": ms}
        md.append(f"| {d} | {t['n']} | {t['h']:.2f} | {len(t['src'])} | {mh:.2f} | {ms} | {'có' if not (mh or ms) else 'KHÔNG'} |")
    other = {d: t for d, t in total.items() if d not in args.target_domains}
    md += ["", "Ngoài mục tiêu: " + ", ".join(f"{d} {t['h']:.2f} h / {len(t['src'])} nguồn"
                                              for d, t in sorted(other.items(), key=lambda x: -x[1]["h"])) + "."]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    Path(str(out.with_suffix("")) + ".json").write_text(json.dumps({"pools": res, "targets": gap,
                                                                    "min_hours": args.min_hours,
                                                                    "min_sources": args.min_sources},
                                                                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
