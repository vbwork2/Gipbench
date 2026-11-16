#!/usr/bin/env python3
"""Bảng gọn cho một hoặc nhiều hệ thống đã chấm (score.py): số theo split + CI95, top lỗi thuật ngữ ref -> hyp.

  python -m gipbench.bench.report.draft_tables --bench data/bench_v0_draft/bench_v0_draft.jsonl \
      --catalog data/bench_v0_draft/catalog_v0_draft.tsv \
      --systems results/v0_draft/e0_baseline results/v0_draft/e1_hw1.5 --baseline e0_baseline \
      --title "NHÁP v0-draft" --out reports/history/v0_draft_tables.md

Mỗi thư mục hệ thống cần summary.json (score.py) và hyps.jsonl (decode.py).
Cặp lỗi thuật ngữ: căn chỉnh lại ref/hyp (strict) như score.py; hyp của một span = token hyp của các op quy về span
đó (C/S theo token ref, chèn theo luật kề bên trái rồi phải của lib/align.attribute); xoá hết = "∅".
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from gipbench.core.paths import REPO

from gipbench.core.align import align
from gipbench.core.manifest import CS_SPLITS, read_jsonl
from gipbench.core.textnorm import Catalog, normalize_hyp, normalize_ref

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


LEX = REPO / "data/dl/lexicons"
TYPES = {"a": "chuẩn hoá / biến thể", "b": "phiên âm Việt", "c": "xoá", "d": "sai từ khác"}


def load_lexicons():
    import unicodedata
    vi, en = set(), set()
    p = LEX / "Viet74K.txt"
    if p.exists():
        for l in open(p, encoding="utf-8"):
            vi.update(unicodedata.normalize("NFC", l.strip().lower()).split())
    p = LEX / "google-10000-english-no-swears.txt"
    if p.exists():
        en = {l.strip().lower() for l in open(p, encoding="utf-8") if l.strip()}
    return vi, en


def error_type(ref, hyp, vi, en):
    """a) chuẩn hoá / biến thể · b) phiên âm Việt · c) xoá · d) sai từ khác (luật ghi dưới bảng phân loại)."""
    if hyp == "∅":
        return "c"
    sq = lambda s: s.replace(" ", "").replace("-", "").replace("'", "")  # noqa: E731
    r, h = sq(ref), sq(hyp)
    last = ref.split()[-1] if ref.split() else ref
    plural = h == r + "s" or (r == h + "s" and not r.endswith("ss") and len(last) > 5)  # ielts -> ielt: lỗi thật
    if r == h or plural or any(ch.isdigit() for ch in ref + hyp):
        return "a"
    toks = hyp.split()
    if any(not t.isascii() for t in toks):  # chữ có dấu tiếng Việt (kể cả đ)
        return "b"
    if toks and all(t in vi for t in toks) and not all(t in en for t in toks):
        return "b"
    return "d"


def classify_section(name, pairs, by_split, vi, en, n_ex=5):
    groups = defaultdict(Counter)
    for (t, h), c in pairs.items():
        groups[error_type(t, h, vi, en)][(t, h)] += c
    tot = sum(pairs.values())
    L = [f"### Phân loại lỗi thuật ngữ — {name} (B1–B3, strict)", "",
         "| Loại | Span sai | % | B1 | B2 | B3 | 5 ví dụ hay gặp nhất (ref → hyp ×lần) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for k in "abcd":
        g = groups[k]
        n = sum(g.values())
        sp = Counter()
        for key, c in g.items():
            for s, v in by_split[key].items():
                sp[s] += v
        ex = "; ".join(f"`{t}` → `{h}` ×{c}" for (t, h), c in g.most_common(n_ex))
        L.append(f"| ({k}) {TYPES[k]} | {n} | {100 * n / max(tot, 1):.1f} | {sp['B1']} | {sp['B2']} | {sp['B3']} | {ex} |")
    L += ["", "Luật: (c) hyp rỗng ở span; (a) giống ref sau khi bỏ dấu cách / gạch nối / nháy, hoặc số ít / số nhiều "
          "(hyp thêm `s`; hoặc ref bỏ `s` khi từ cuối > 5 chữ và không tận cùng `ss`: `ielts` → `ielt` là lỗi thật), hoặc có "
          "chữ số; (b) hyp có chữ có dấu tiếng Việt, hoặc mọi âm tiết có trong Viet74K và không phải từ tiếng Anh thông "
          "dụng (google-10000); (d) còn lại (từ Anh khác, chính tả sai kiểu Anh).", ""]
    prop = defaultdict(Counter)
    for (t, h), c in groups["a"].items():
        if not any(ch.isdigit() for ch in t + h):
            prop[t][h] += c
    if prop:
        L += [f"#### ĐỀ XUẤT thêm `variants` vào catalog (loại a, CHƯA áp dụng; {len(prop)} thuật ngữ)", "",
              "| term | variants đề xuất (hyp ×lần) |", "| --- | --- |"]
        for t, hs in sorted(prop.items(), key=lambda x: -sum(x[1].values())):
            L.append(f"| {t} | {' · '.join(f'`{h}` ×{c}' for h, c in hs.most_common())} |")
        L.append("")
    num = Counter({k: c for k, c in groups["a"].items() if any(ch.isdigit() for ch in k[0] + k[1])})
    if num:
        L += ["Loại (a) có chữ số (để chuẩn hoá số, không đề xuất variants): "
              + "; ".join(f"`{t}` → `{h}` ×{c}" for (t, h), c in num.most_common(8)), ""]
    return L


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
    ap.add_argument("--classify", nargs="*", default=[], help="hệ thống cần phân loại lỗi thuật ngữ (vd e0_baseline)")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="bảng so sánh theo split giữa hai hệ thống")
    ap.add_argument("--by-domain", action="store_true", help="bảng B5 theo domain (WER) và B3 theo topic (CS-WER)")
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
    if args.compare:
        a_, b_ = (S[x][1]["groups"] for x in args.compare)
        L += [f"## So sánh theo split: {args.compare[0]} → {args.compare[1]}", "",
              "Hotwords của E1 = thuật ngữ B1/B2 (catalog nháp, bỏ reserved_b3): thuật ngữ B3 KHÔNG có trong hotwords, "
              "nên term recall B3 đo khả năng tổng quát và mức chèn nhầm.", "",
              "| Split | Utt | Đơn vị bootstrap | WER | CS-WER strict | CS-WER lenient | N-WER | Term recall | Term precision | Chèn sai /100 utt |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

        def ab(g, k):
            x, y = a_.get(g, {}).get(k), b_.get(g, {}).get(k)
            if x is None or y is None:
                return "–"
            return f"{x:.2f} → {y:.2f} ({y - x:+.2f})"
        for g in SPLITS + ["main_B1+B2"]:
            if g not in a_:
                continue
            cs = g != "B5"
            L.append(f"| {g} | {a_[g]['utt']} | {a_[g].get('boot_units', '–')} | {ab(g, 'wer')} | "
                     f"{ab(g, 'cs_wer') if cs else '–'} | {ab(g, 'cs_wer_lenient') if cs else '–'} | {ab(g, 'n_wer')} | "
                     f"{ab(g, 'term_recall') if cs else '–'} | {ab(g, 'term_precision') if cs else '–'} | "
                     f"{ab(g, 'false_terms_per_100utt')} |")
        L.append("")
    if args.by_domain:
        from gipbench.bench.score import agg
        meta = {r["id"]: r for r in bench}
        U = {n: [json.loads(l) for l in open(p / "utt_scores.jsonl", encoding="utf-8")] for n, (p, _) in S.items()}
        for split, key, metric, title in (("B5", "domain", "wer", "B5 theo domain: WER"),
                                          ("B3", "topic", "cs_wer", "B3 theo topic ViMedCSS: CS-WER")):
            vals = sorted({str(meta[u["id"]].get(key) or "?") for u in next(iter(U.values())) if u["split"] == split})
            L += [f"## {title}", "", "| " + " | ".join([key, "Utt"] + list(U)) + " |", "|" + " --- |" * (len(U) + 2)]
            for v in vals:
                cells, nutt = [], 0
                for n, us in U.items():
                    R = [u for u in us if u["split"] == split and str(meta[u["id"]].get(key) or "?") == v]
                    nutt = len(R)
                    a = agg(R)[metric]
                    cells.append(f(a))
                L.append(f"| {v} | {nutt} | " + " | ".join(cells) + " |")
            L.append("")
    vi, en = load_lexicons() if args.classify else (set(), set())
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
        if name in args.classify:
            L += classify_section(name, pairs, by_split, vi, en)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
