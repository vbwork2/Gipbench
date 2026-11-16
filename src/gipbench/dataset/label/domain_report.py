#!/usr/bin/env python3
"""Báo cáo phân bố domain (câu, giờ, số thuật ngữ, top thuật ngữ) theo nhóm dữ liệu (decisions §12.6).

  python -m gipbench.dataset.label.domain_report --catalog data/raw/_proxy/catalog_proxy.tsv --out reports/history/domain_report.md \
      --group B1=data/raw/ggroup/b1_candidates.jsonl B2=data/raw/consensus/b2_consensus_candidates.jsonl \
              B3=data/dl/vimedcss_meta/hard_set.csv cs_real=data/dl/vimedcss_meta/train_set.csv \
              cs_pseudo=data/raw/vss_full/vss_cs_clean_dom.jsonl

Nguồn .jsonl: manifest (span <en> của text_tagged, domain + domain_source của dòng). Nguồn .csv: meta ViMedCSS
(segment_text + cs_terms_list, duration_seconds; domain = medical), tag theo đúng cách conv_vimedcss gắn nhãn Spec
(auto_tag với thuật ngữ dataset + vi_words + term_decisions), không cần audio.
Thuật ngữ = span <en> sau chuẩn hoá (lib.textnorm.normalize_ref, gộp biến thể theo catalog).
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.textnorm import Catalog, normalize_ref


def rows_of(path, cat):
    """-> (domain, domain_source, giờ, [thuật ngữ])"""
    p = Path(path)
    if p.suffix == ".csv":
        from gipbench.dataset.ingest.common import auto_tag, load_vi_words
        vw = load_vi_words()
        for r in csv.DictReader(open(p, encoding="utf-8")):
            terms = [t.strip() for t in (r.get("cs_terms_list") or "").split(";") if t.strip()]
            tg = auto_tag(r["segment_text"], cat, terms, vi_words=vw)
            yield "medical", "dataset", float(r["duration_seconds"]) / 3600, terms_of(tg["text_tagged"], cat)
        return
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            yield (r.get("domain") or "other", r.get("domain_source") or "converter", float(r["duration"]) / 3600,
                   terms_of(r.get("text_tagged", ""), cat))


def terms_of(tagged, cat):
    _, spans = normalize_ref(tagged or "", cat)
    return [" ".join(s) for s in spans]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", nargs="+", required=True, metavar="TÊN=FILE[,FILE]")
    ap.add_argument("--catalog")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog) if args.catalog else Catalog()
    md = ["# Phân bố domain theo nhóm dữ liệu", "",
          "Thuật ngữ = span `<en>` sau chuẩn hoá (gộp biến thể catalog). `nguồn domain`: converter = domain mặc định "
          "của converter / dataset, inferred = suy từ text (`src/gipbench/dataset/label/infer_domain.py`, `configs/dataset/labeling/domain_keywords.tsv`), "
          "dataset = nhãn của dataset.", ""]
    summary = []
    for g in args.group:
        name, _, paths = g.partition("=")
        n, h, src = Counter(), Counter(), Counter()
        terms = defaultdict(Counter)
        for p in paths.split(","):
            for dom, ds, hrs, ts in rows_of(p, cat):
                n[dom] += 1
                h[dom] += hrs
                src[ds] += 1
                terms[dom].update(ts)
        allt = Counter()
        for c in terms.values():
            allt.update(c)
        summary.append((name, sum(n.values()), sum(h.values()), len(allt)))
        md += [f"## {name}", "", f"Nguồn: {', '.join(f'`{p}`' for p in paths.split(','))}. "
               f"{sum(n.values())} câu, {sum(h.values()):.2f} h, {len(allt)} thuật ngữ khác nhau; nguồn domain: "
               + ", ".join(f"{k} {v}" for k, v in src.most_common()) + ".", "",
               f"Top {args.top} thuật ngữ (số lần): " + " · ".join(f"{t} {c}" for t, c in allt.most_common(args.top)), "",
               "| Domain | Câu | Giờ | % giờ | Thuật ngữ khác nhau | Top 5 thuật ngữ |", "| --- | --- | --- | --- | --- | --- |"]
        tot = sum(h.values()) or 1
        for d, c in n.most_common():
            md.append(f"| {d} | {c} | {h[d]:.2f} | {100 * h[d] / tot:.1f}% | {len(terms[d])} | "
                      + ", ".join(f"{t} {k}" for t, k in terms[d].most_common(5)) + " |")
        md.append("")
    md[4:4] = ["| Nhóm | Câu | Giờ | Thuật ngữ khác nhau |", "| --- | --- | --- | --- |"] + \
        [f"| {a} | {b} | {c:.2f} | {d} |" for a, b, c, d in summary] + [""]
    Path(args.out).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:8 + len(summary)]))


if __name__ == "__main__":
    main()
