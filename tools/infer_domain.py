#!/usr/bin/env python3
"""Suy domain từ text cho manifest train chưa có domain thật (decisions §10, §12.6).

  python tools/infer_domain.py --manifest data/raw/vss_full/vss_cs_clean.jsonl \
      --out data/raw/vss_full/vss_cs_clean_dom.jsonl --catalog data/raw/_proxy/catalog_proxy.tsv \
      --report reports/domain_infer_vss_cs.md

Chỉ suy cho dòng có domain thuộc --infer-domains (mặc định conversation, other, rỗng: domain mặc định của converter,
không phải nhãn thật); dòng khác giữ nguyên. Dòng được suy: domain = kết quả (lib/domain.py), domain_source = inferred,
domain_orig = domain cũ, domain_scores = điểm từng domain. Báo cáo: phân bố domain (câu, giờ) + --samples câu mẫu được
suy ra là --sample-domain (mặc định finance, để kiểm tra trước khi dùng --holdout-domain finance).
Tham số luật (--min-score, --margin) chưa kiểm chứng: chỉnh theo câu mẫu.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.domain import DEFAULT_KEYWORDS, DomainInferer, load_keywords  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keywords", default=str(DEFAULT_KEYWORDS))
    ap.add_argument("--catalog")
    ap.add_argument("--infer-domains", nargs="*", default=["conversation", "other", ""])
    ap.add_argument("--min-score", type=float, default=2)
    ap.add_argument("--margin", type=float, default=1)
    ap.add_argument("--sample-domain", default="finance")
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report")
    args = ap.parse_args()
    inf = DomainInferer(load_keywords(args.keywords), Catalog.load(args.catalog) if args.catalog else None,
                        args.min_score, args.margin)
    rows, out = list(read_jsonl(args.manifest)), []
    dist, kept = Counter(), Counter()
    hours = Counter()
    for r in rows:
        o = {k: v for k, v in r.items() if not k.startswith("_")}
        if (r.get("domain") or "") in args.infer_domains:
            d, sc = inf.infer(r.get("text_tagged") or r.get("text") or "")
            o.update(domain=d, domain_source="inferred", domain_orig=r.get("domain") or "",
                     domain_scores=dict(sc.most_common()))
            dist[d] += 1
            hours[d] += float(r.get("duration") or 0) / 3600
        else:
            kept[r.get("domain") or ""] += 1
        out.append(o)
    write_jsonl(args.out, out)
    pool = [o for o in out if o.get("domain_source") == "inferred" and o["domain"] == args.sample_domain]
    rnd = random.Random(args.seed)
    sample = rnd.sample(pool, min(args.samples, len(pool)))
    n = sum(dist.values())
    md = [f"# Suy domain: `{args.manifest}`", "",
          f"{len(rows)} câu; suy cho {n} câu (domain thuộc {args.infer_domains}), giữ nguyên {sum(kept.values())}"
          + (f" ({dict(kept)})" if kept else "") + f". Luật: min-score {args.min_score:g}, margin {args.margin:g}, "
          f"keywords `{Path(args.keywords).name}`.", "", "| Domain | Câu | Giờ | % câu |", "| --- | --- | --- | --- |"]
    md += [f"| {d} | {c} | {hours[d]:.2f} | {100 * c / max(n, 1):.1f}% |" for d, c in dist.most_common()]
    md += ["", f"## {len(sample)} câu mẫu được suy ra là {args.sample_domain} (seed {args.seed}, trên {len(pool)} câu)", "",
           "| id | điểm | text |", "| --- | --- | --- |"]
    md += [f"| {o['id']} | {o['domain_scores']} | {(o.get('text_tagged') or o.get('text') or '')[:160]} |" for o in sample]
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:6 + len(dist)]))
    print(f"Đã ghi {args.out}")


if __name__ == "__main__":
    main()
