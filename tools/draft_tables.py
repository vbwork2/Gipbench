#!/usr/bin/env python3
"""Bảng gọn cho một hoặc nhiều hệ thống đã chấm (score.py): số theo split + CI95, top lỗi thuật ngữ ref -> hyp.

  python tools/draft_tables.py --bench data/bench_v0_draft/bench_v0_draft.jsonl \
      --catalog data/bench_v0_draft/catalog_v0_draft.tsv \
      --systems results/v0_draft/e0_baseline results/v0_draft/e1_hw1.5 --baseline e0_baseline \
      --title "NHÁP v0-draft" --out reports/v0_draft_tables.md

Mỗi thư mục hệ thống cần summary.json (score.py) và hyps.jsonl (decode.py).
Cặp lỗi thuật ngữ: căn chỉnh lại ref/hyp (strict) như score.py; hyp của một span = token hyp của các op quy về span
đó (C/S theo token ref, chèn theo luật kề bên trái rồi phải của lib/align.attribute); xoá hết = "∅".
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.align import align  # noqa: E402
from lib.manifest import CS_SPLITS, read_jsonl  # noqa: E402
from lib.textnorm import Catalog, normalize_hyp, normalize_ref  # noqa: E402

SPLITS = ["B1", "B2", "B3", "B5"]


def span_hyps(ref_text, hyp_text, cat):
    """[(term, hyp_segment)] cho mọi span của câu (strict)."""
    rt, spans = normalize_ref(ref_text, cat, "strict")
    ht = normalize_hyp(hyp_text, cat, "strict")
    ref = [t.tok for t in rt]
    sp = [t.span for t in rt]
    ops = align(ref, ht)
    nxt, next_ref = None, [None] * len(ops)
    for k in range(len(ops) - 1, -1, -1):
        next_ref[k] = nxt
        if ops[k][1] is not None:
            nxt = ops[k][1]
    seg = defaultdict(list)
    last = None
    for k, (t, ri, hj) in enumerate(ops):
        if t == "I":
            left = sp[last] if last is not None else -1
            right = sp[next_ref[k]] if next_ref[k] is not None else -1
            s = left if left >= 0 else right
        else:
            s, last = sp[ri], ri
        if s >= 0 and hj is not None:
            seg[s].append(ht[hj])
    return [(" ".join(term), " ".join(seg[i]) or "∅") for i, term in enumerate(spans)]


def term_errors(bench, hyps, cat):
    pairs, by_split = Counter(), defaultdict(Counter)
    for r in bench:
        if r["split"] not in CS_SPLITS:
            continue
        for term, h in span_hyps(r["text_tagged"], hyps.get(r["id"], ""), cat):
            if h != term:
                pairs[(term, h)] += 1
                by_split[(term, h)][r["split"]] += 1
    return pairs, by_split


def f(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def ci(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]" if c else "–"


def system_table(summ, base=None):
    L = ["| Split | Utt | WER | CI95 | CS-WER strict | CI95 | CS-WER lenient | N-WER | CI95 | Term R | Term P | Chèn sai /100 utt |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for g in SPLITS + ["main_B1+B2"]:
        a = summ["groups"].get(g)
        if not a:
            continue
        cs = g != "B5"

        def dl(m):
            if not base or base["groups"].get(g, {}).get(m) is None or a.get(m) is None:
                return ""
            return f" ({a[m] - base['groups'][g][m]:+.2f})"
        L.append(f"| {g} | {a['utt']} | {f(a['wer'])}{dl('wer')} | {ci(a.get('wer_ci95'))} | "
                 f"{f(a['cs_wer']) if cs else '–'}{dl('cs_wer') if cs else ''} | {ci(a.get('cs_wer_ci95')) if cs else '–'} | "
                 f"{f(a['cs_wer_lenient']) if cs else '–'} | {f(a['n_wer'])}{dl('n_wer')} | {ci(a.get('n_wer_ci95'))} | "
                 f"{f(a['term_recall']) if cs else '–'}{dl('term_recall') if cs else ''} | {f(a['term_precision'])} | "
                 f"{f(a['false_terms_per_100utt'])}{dl('false_terms_per_100utt')} |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--baseline", help="tên hệ thống (thư mục) để ghi delta trong ngoặc")
    ap.add_argument("--title", default="")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cat = Catalog.load(args.catalog)
    bench = read_jsonl(args.bench)
    S = {Path(p).name: (Path(p), json.loads((Path(p) / "summary.json").read_text(encoding="utf-8"))) for p in args.systems}
    base = S[args.baseline][1] if args.baseline in S else None
    L = [f"# {args.title}".rstrip(), ""]
    if "draft" in args.bench or "NHÁP" in args.title:
        L += ["> **NHÁP.** Bench v0-draft, B1–B3 CHƯA được người duyệt (nhãn auto_tag / converter). Chỉ để xem xu hướng, "
              "không dùng để kết luận.", ""]
    if len(S) > 1:
        L += ["## Leaderboard (delta so với baseline trong ngoặc)", "",
              "| Hệ thống | CS-WER B1+B2 | CI95 | N-WER B1+B2 | Term R B1+B2 | Term P B1+B2 | CS-WER B3 | WER B5 | Chèn sai B5 /100 utt | RTF |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name, (p, s) in S.items():
            def d(g, k):
                v = s["groups"].get(g, {}).get(k)
                if v is None:
                    return "–"
                bv = None if base is None or name == args.baseline else base["groups"].get(g, {}).get(k)
                return f"{v:.2f}" + ("" if bv is None else f" ({v - bv:+.2f})")
            ds = p / "decode_summary.json"
            rtf = json.loads(ds.read_text(encoding="utf-8")).get("rtf") if ds.exists() else None
            L.append(f"| {name} | {d('main_B1+B2', 'cs_wer')} | {ci(s['groups'].get('main_B1+B2', {}).get('cs_wer_ci95'))} | "
                     f"{d('main_B1+B2', 'n_wer')} | {d('main_B1+B2', 'term_recall')} | {d('main_B1+B2', 'term_precision')} | "
                     f"{d('B3', 'cs_wer')} | {d('B5', 'wer')} | {d('B5', 'false_terms_per_100utt')} | {f(rtf, 3)} |")
        L.append("")
    for name, (p, s) in S.items():
        hyps = {h["id"]: h.get("hyp", "") for h in read_jsonl(str(p / "hyps.jsonl"))}
        L += [f"## {name}", ""] + system_table(s, base if name != args.baseline else None) + [""]
        pairs, by_split = term_errors(bench, hyps, cat)
        L += [f"Top {args.top} lỗi thuật ngữ (B1–B3, strict; {sum(pairs.values())} span sai, {len(pairs)} cặp khác nhau)", "",
              "| # | ref → hyp | Lần | Theo split |", "| --- | --- | --- | --- |"]
        for i, ((t, h), c) in enumerate(pairs.most_common(args.top), 1):
            sp = ", ".join(f"{k} {v}" for k, v in sorted(by_split[(t, h)].items()))
            L.append(f"| {i} | `{t}` → `{h}` | {c} | {sp} |")
        main = Counter({k: v["B1"] + v["B2"] for k, v in by_split.items() if v["B1"] + v["B2"]})
        L += ["", f"Top {args.top} lỗi thuật ngữ riêng B1+B2 ({sum(main.values())} span sai)", "",
              "| # | ref → hyp | Lần | Theo split |", "| --- | --- | --- | --- |"]
        for i, ((t, h), c) in enumerate(main.most_common(args.top), 1):
            sp = ", ".join(f"{k} {v}" for k, v in sorted(by_split[(t, h)].items()) if k in ("B1", "B2"))
            L.append(f"| {i} | `{t}` → `{h}` | {c} | {sp} |")
        L.append("")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
