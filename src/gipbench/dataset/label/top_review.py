#!/usr/bin/env python3
"""Phiếu duyệt nhanh cho configs/dataset/labeling/vi_abbrev.txt và configs/dataset/labeling/vi_loanwords.txt: --top mục xuất hiện nhiều nhất của
mỗi file trong các corpus có sẵn, kèm 1 câu ví dụ. Người duyệt chỉ XOÁ dòng sai trong CSV; phần còn lại khoá thành v1.

  python -m gipbench.dataset.label.top_review --out reports/history/top_review.csv

Corpus (bỏ qua nguồn nào không có): G-Group (--ggroup, parquet cột id/text), ViMedCSS (--vimedcss-meta, *_set.csv
cột segment_id/segment_text), VietSuperSpeech (--vss, train.json + dev.json). Đếm theo token sau lib.textnorm.basic_tokens
(viết thường); mục nhiều từ đếm theo chuỗi token liền. Câu ví dụ: câu ngắn nhất có mục đó, ưu tiên G-Group, rồi
ViMedCSS, rồi VSS.
"""
import argparse
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

from gipbench.core.textnorm import basic_tokens

from gipbench.core.paths import REPO as G


def read_list(path):
    """[(mục, cột 2)] của các dòng đang dùng (bỏ '#', '#-' và header)."""
    out = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.rstrip("\n").split("\t")
        if cols[0] in ("abbr", "word"):
            continue
        out.append((cols[0].strip().lower(), cols[1] if len(cols) > 1 else "", cols[-1] if len(cols) > 2 else ""))
    return out


def corpora(args):
    for f in sorted(glob.glob(args.ggroup)):
        import pyarrow.parquet as pq
        t = pq.read_table(f, columns=["id", "text"]).to_pydict()
        name = Path(f).stem
        for i, x in zip(t["id"], t["text"]):
            yield "ggroup", f"gg_{name}_{i}", str(x)
    if args.vimedcss_meta:
        for f in sorted(Path(args.vimedcss_meta).glob("*_set.csv")):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                yield "vimedcss", r["segment_id"], r["segment_text"]
    if args.vss:
        for jf in ("train.json", "dev.json"):
            p = Path(args.vss) / jf
            if p.exists():
                raw = p.read_text(encoding="utf-8").strip()
                data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
                for k, d in enumerate(data):
                    yield "vss", f"vss_{jf[:-5]}_{k}", str(d["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abbrev", default=str(G / "configs/dataset/labeling/vi_abbrev.txt"))
    ap.add_argument("--loan", default=str(G / "configs/dataset/labeling/vi_loanwords.txt"))
    ap.add_argument("--ggroup", default=str(G / "data/dl/ggroup/*.parquet"))
    ap.add_argument("--vimedcss-meta", default=str(G / "data/dl/vimedcss_meta"))
    ap.add_argument("--vss", default=str(G / "data/dl/VietSuperSpeech"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    lists = {"vi_abbrev.txt": read_list(args.abbrev), "vi_loanwords.txt": read_list(args.loan)}
    want = {e for L in lists.values() for e, _, _ in L}
    multi = {e for e in want if " " in e}
    cnt = {s: Counter() for s in ("ggroup", "vimedcss", "vss")}
    ex = {}
    prio = {"ggroup": 0, "vimedcss": 1, "vss": 2}
    n_sent = Counter()
    for src, uid, text in corpora(args):
        n_sent[src] += 1
        toks = basic_tokens(text)
        joined = " " + " ".join(toks) + " "
        hits = Counter(t for t in toks if t in want)
        for m in multi:
            c = joined.count(" " + m + " ")
            if c:
                hits[m] += c
        for e, c in hits.items():
            cnt[src][e] += c
            key = (prio[src], len(text))
            if e not in ex or key < ex[e][0]:
                ex[e] = (key, src, uid, text.strip())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "rank", "entry", "group_or_kind", "count", "count_ggroup", "count_vimedcss", "count_vss",
                    "example_source", "example_id", "example_text", "list_source"])
        for fname, L in lists.items():
            tot = sorted(((sum(cnt[s][e] for s in cnt), e, g, so) for e, g, so in L), key=lambda x: (-x[0], x[1]))
            for rank, (c, e, g, so) in enumerate(tot[:args.top], 1):
                _, src, uid, text = ex.get(e, (None, "", "", ""))
                w.writerow([fname, rank, e, g, c, cnt["ggroup"][e], cnt["vimedcss"][e], cnt["vss"][e], src, uid, text, so])
    print(f"Đã ghi {args.out}: top {args.top} mỗi file; corpus: "
          + ", ".join(f"{s} {n} câu" for s, n in n_sent.items()))


if __name__ == "__main__":
    main()
