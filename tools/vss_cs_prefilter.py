#!/usr/bin/env python3
"""Lọc rẻ vss_cs_candidates TRƯỚC khi chạy ASR (không mở audio).

  python tools/vss_cs_prefilter.py --manifest data/raw/vss_full/vss_cs_candidates.jsonl \
      --catalog data/raw/_proxy/catalog_proxy.tsv --bench data/raw/_proxy/bench_proxy.jsonl \
      --out data/raw/vss_full/vss_cs_pre.jsonl --sample 1000 --sample-out data/raw/vss_full/vss_cs_pilot.jsonl

Thứ tự các bước (mỗi bước báo số câu và số giờ còn lại trong <out>.report.md):
  1. độ dài trong [--min-dur, --max-dur] (1–20 s);
  2. --max-en-frac (0.5): tỉ lệ token nằm trong <en> trên tổng token (sau chuẩn hoá strict) phải <= ngưỡng; bỏ câu
     tiếng Anh nguyên câu / gần nguyên câu (lời bài hát, câu nhận dạng sai) trước khi tốn GPU;
  3. rò rỉ: speaker / source / text trùng bench (--bench, split --bench-splits B1 B2 B3) và thuật ngữ reserved_b3
     của --catalog (lib/leak.py, cùng luật với converter);
  4. trần thuật ngữ --term-cap (50): mỗi thuật ngữ (span <en> sau chuẩn hoá) có mặt trong tối đa 50 câu. Câu được xét
     theo độ hiếm: câu có thuật ngữ hiếm nhất (ít câu nhất sau bước 3) trước, cùng độ hiếm thì thứ tự ngẫu nhiên
     (--seed). Câu được nhận khi MỌI thuật ngữ của nó còn dưới trần.
--sample N: lấy ngẫu nhiên (--seed) N câu từ kết quả để chạy pilot, ghi ra --sample-out.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.leak import LeakGuard  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog, normalize_ref, plain_text  # noqa: E402


def en_frac_and_terms(text_tagged, cat):
    toks, spans = normalize_ref(text_tagged, cat, "strict")
    n_en = sum(1 for t in toks if t.span >= 0)
    return (n_en / len(toks) if toks else 1.0), sorted({" ".join(s) for s in spans})


def term_cap(rows, terms, cap, seed):
    """Trả về chỉ số câu được nhận (ưu tiên thuật ngữ hiếm, ngẫu nhiên có seed)."""
    freq = Counter(t for ts in terms for t in ts)
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    order.sort(key=lambda i: min((freq[t] for t in terms[i]), default=0))  # sort ổn định: giữ thứ tự ngẫu nhiên
    used, keep = Counter(), []
    for i in order:
        if terms[i] and all(used[t] < cap for t in terms[i]):
            keep.append(i)
            used.update(terms[i])
    return sorted(keep), used, freq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--bench", nargs="*", default=[])
    ap.add_argument("--bench-splits", nargs="+", default=["B1", "B2", "B3"])
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--max-en-frac", type=float, default=0.5)
    ap.add_argument("--term-cap", type=int, default=50, help="0 = không đặt trần")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sample-out")
    args = ap.parse_args()
    if args.sample and not args.sample_out:
        sys.exit("--sample cần --sample-out")
    cat = Catalog.load(args.catalog)
    if args.bench:
        guard = LeakGuard((r for b in args.bench for r in read_jsonl(b)), cat, args.bench_splits)
        bench_note = f"{guard.n_bench} câu bench ({' '.join(args.bench_splits)}), {len(guard.reserved)} thuật ngữ reserved_b3"
    else:
        guard = LeakGuard([], cat)
        bench_note = f"KHÔNG có --bench (chỉ kiểm reserved_b3: {len(guard.reserved)} thuật ngữ)"
    rows = read_jsonl(args.manifest)
    steps = [("vss_cs_candidates", len(rows), sum(float(r["duration"]) for r in rows) / 3600)]
    why = Counter()

    def step(name, rows_):
        steps.append((name, len(rows_), sum(float(r["duration"]) for r in rows_) / 3600))
        return rows_

    rows = step(f"độ dài {args.min_dur:g}–{args.max_dur:g} s",
                [r for r in rows if args.min_dur <= float(r["duration"]) <= args.max_dur])
    kept, terms = [], []
    fracs = Counter()
    for r in rows:
        f, ts = en_frac_and_terms(r["text_tagged"], cat)
        fracs[min(int(f * 10), 9)] += 1
        if f > args.max_en_frac:
            continue
        r["prefilter"] = {"en_frac": round(f, 3)}
        kept.append(r)
        terms.append(ts)
    rows = step(f"--max-en-frac {args.max_en_frac:g}", kept)
    kept2, terms2 = [], []
    for r, ts in zip(rows, terms):
        w = guard.reason(r, r.get("text") or plain_text(r["text_tagged"]))
        if w:
            why[w] += 1
            continue
        kept2.append(r)
        terms2.append(ts)
    rows, terms = step("rò rỉ B1–B3 + reserved_b3", kept2), terms2
    used = freq = Counter()
    if args.term_cap:
        idx, used, freq = term_cap(rows, terms, args.term_cap, args.seed)
        rows = step(f"--term-cap {args.term_cap}", [rows[i] for i in idx])
    write_jsonl(args.out, rows)
    sample = []
    if args.sample:
        sample = random.Random(args.seed).sample(rows, min(args.sample, len(rows)))
        sample.sort(key=lambda r: r["id"])
        write_jsonl(args.sample_out, sample)

    md = [f"# Lọc rẻ trước ASR — {args.manifest}", "", f"Chặn rò rỉ: {bench_note}. Seed {args.seed}.", "",
          "| Bước | Câu còn lại | Giờ còn lại |", "| --- | --- | --- |"]
    md += [f"| {n} | {c} | {h:.2f} |" for n, c, h in steps]
    md += ["", "Phân bố tỉ lệ token tiếng Anh (sau bước độ dài):", "", "| en_frac | Số câu |", "| --- | --- |"]
    md += [f"| {k / 10:.1f}–{(k + 1) / 10:.1f} | {fracs[k]} |" for k in range(10)]
    if why:
        md += ["", "Lý do rò rỉ:", "", "| Lý do | Số câu |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in why.most_common()]
    if args.term_cap:
        capped = [t for t, c in freq.items() if c > args.term_cap]
        md += ["", f"Trần thuật ngữ: {len(freq)} thuật ngữ khác nhau trước trần, {len(capped)} thuật ngữ vượt trần "
               f"{args.term_cap}; sau trần còn {len(used)} thuật ngữ.", "",
               "| Thuật ngữ | Số câu trước trần | Sau trần |", "| --- | --- | --- |"]
        md += [f"| {t} | {c} | {used[t]} |" for t, c in freq.most_common(25)]
        long_spans = sum(1 for t in used if len(t.split()) >= 4)
        md += ["", f"Thuật ngữ dài >= 4 từ còn lại sau trần: {long_spans} (thường là cụm tiếng Anh nhận dạng sai; "
               "bộ lọc đồng thuận sẽ loại)."]
    if args.sample:
        sh = sum(float(r["duration"]) for r in sample) / 3600
        md += ["", f"Mẫu pilot: {len(sample)} câu, {sh:.2f} h -> `{args.sample_out}`."]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for n, c, h in steps:
        print(f"  {n}: {c} câu, {h:.2f} h")
    if args.sample:
        print(f"  mẫu pilot: {len(sample)} câu -> {args.sample_out}")
    print(f"Đã ghi {args.out} và {args.out}.report.md")


if __name__ == "__main__":
    main()
