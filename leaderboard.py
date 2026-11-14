#!/usr/bin/env python3
"""Gom kết quả mọi hệ thống, so cặp với baseline bằng bootstrap theo người nói, áp luật chọn.

  python leaderboard.py --results results --baseline e0_baseline [--out reports/leaderboard]

Mỗi thư mục results/<system>/ cần utt_scores.jsonl (từ score.py); decode_summary.json (từ decode.py) là tuỳ chọn.

Luật chọn (sửa ngưỡng bằng tham số):
  1. Loại nếu WER trên B5 tăng > --max-b5-delta điểm tuyệt đối so với baseline.
  2. Loại nếu N-WER (B1–B4) tăng > --max-nwer-delta điểm.
  2b. Loại nếu WER của bất kỳ domain nào trong B5 tăng > --max-b5-domain-delta (chặn suy giảm cục bộ).
  3. Loại nếu RTF > --max-rtf-ratio × RTF baseline (khi có decode_summary.json).
  4. Chỉ coi là cải thiện khi CI 95% của ΔCS-WER (B1+B2) nằm hoàn toàn dưới 0.
  5. Xếp hạng hệ thống đủ điều kiện theo CS-WER (B1+B2), hoà thì CS-WER B3 thấp hơn thắng.
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda import CSS  # noqa: E402
from lib.manifest import CS_SPLITS  # noqa: E402
from lib.svgcharts import table  # noqa: E402
from score import agg  # noqa: E402

METRICS = [  # (khoá, nhóm split, metric của agg)
    ("cs_main", ["B1", "B2"], "cs_wer"),
    ("cs_main_len", ["B1", "B2"], "cs_wer_lenient"),
    ("nwer", CS_SPLITS, "n_wer"),
    ("wer_main", ["B1", "B2"], "wer"),
    ("cs_b3", ["B3"], "cs_wer"),
    ("cs_b4", ["B4"], "cs_wer"),
    ("b5_wer", ["B5"], "wer"),
    ("term_r", CS_SPLITS, "term_recall"),
    ("term_p", CS_SPLITS, "term_precision"),
    ("b5_false", ["B5"], "false_terms_per_100utt"),
]


def load(d: Path):
    recs = [json.loads(l) for l in open(d / "utt_scores.jsonl", encoding="utf-8")]
    ds = d / "decode_summary.json"
    rtf = json.loads(ds.read_text()).get("rtf") if ds.exists() else None
    return {r["id"]: r for r in recs}, rtf


def metric(recs, splits, m):
    R = [r for r in recs if r["split"] in splits]
    return agg(R)[m] if R else None


def paired_ci(base, other, splits, m, iters, seed=0):
    ids = [i for i, r in base.items() if r["split"] in splits and i in other]
    by = defaultdict(list)
    for i in ids:
        by[base[i]["speaker"]].append(i)
    spk = list(by)
    if len(spk) < 2:
        return None
    rnd = random.Random(seed)
    diffs = []
    for _ in range(iters):
        sid = [i for s in (rnd.choice(spk) for _ in spk) for i in by[s]]
        a = agg([other[i] for i in sid])[m]
        b = agg([base[i] for i in sid])[m]
        if a is not None and b is not None:
            diffs.append(a - b)
    diffs.sort()
    return [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if diffs else None


def dotplot(rows, base_val, width=640, label_w=150, row_h=30):
    """Điểm + whisker CI 95% của CS-WER (B1+B2) từng hệ thống; vạch dọc = baseline."""
    vals = [v for _, v, lo, hi in rows for v in (v, lo, hi) if v is not None]
    vmin, vmax = 0, max(vals + [1]) * 1.08
    pw = width - label_w - 60
    h = row_h * len(rows) + 30

    def X(v):
        return label_w + pw * (v - vmin) / (vmax - vmin)
    o = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" font-family="inherit">']
    step = next(st for st in (1, 2, 5, 10, 20, 25, 50, 100) if vmax / st <= 6)
    for t in range(0, int(vmax) + 1, step):
        o.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="4" y2="{h - 22}" class="axis"/>'
                 f'<text x="{X(t):.1f}" y="{h - 6}" text-anchor="middle" class="tick">{t}%</text>')
    if base_val is not None:
        o.append(f'<line x1="{X(base_val):.1f}" x2="{X(base_val):.1f}" y1="2" y2="{h - 22}" class="target" stroke-dasharray="4 3"/>')
    for i, (name, v, lo, hi) in enumerate(rows):
        y = 6 + i * row_h + row_h / 2
        o.append(f'<text x="{label_w - 8}" y="{y + 4}" text-anchor="end" class="lab">{escape(name)}</text>')
        if lo is not None:
            o.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y}" y2="{y}" style="stroke:var(--s1);stroke-width:2"/>')
        o.append(f'<circle cx="{X(v):.1f}" cy="{y}" r="5" style="fill:var(--s1);stroke:var(--surface);stroke-width:2">'
                 f'<title>{escape(name)}: {v:.2f}%' + (f' (CI {lo:.2f}–{hi:.2f})' if lo is not None else '') + '</title></circle>')
        o.append(f'<text x="{X(hi if hi is not None else v) + 8:.1f}" y="{y + 4}" class="val">{v:.2f}</text>')
    o.append("</svg>")
    return "".join(o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", default="reports/leaderboard")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--max-b5-delta", type=float, default=0.3)
    ap.add_argument("--max-nwer-delta", type=float, default=0.3)
    ap.add_argument("--max-rtf-ratio", type=float, default=1.2)
    ap.add_argument("--max-b5-domain-delta", type=float, default=0.5,
                    help="loại nếu WER của BẤT KỲ domain nào trong B5 tăng quá ngưỡng này (domain >= --min-domain-tokens)")
    ap.add_argument("--min-domain-tokens", type=int, default=500)
    args = ap.parse_args()

    root = Path(args.results)
    systems = sorted(d.name for d in root.iterdir() if (d / "utt_scores.jsonl").exists())
    if args.baseline not in systems:
        sys.exit(f"Không thấy baseline {args.baseline} trong {root}")
    data = {s: load(root / s) for s in systems}
    base, base_rtf = data[args.baseline]

    rows = []
    for s in systems:
        recs, rtf = data[s]
        R = list(recs.values())
        row = {"system": s, "rtf": rtf}
        for k, splits, m in METRICS:
            row[k] = metric(R, splits, m)
        if s != args.baseline:
            row["d_cs_main_ci"] = paired_ci(base, recs, ["B1", "B2"], "cs_wer", args.iters)
            row["d_b5_ci"] = paired_ci(base, recs, ["B5"], "wer", args.iters)
        rows.append(row)
    b = next(r for r in rows if r["system"] == args.baseline)

    def b5_by_domain(recs):
        g = defaultdict(list)
        for x in recs.values():
            if x["split"] == "B5":
                g[x.get("domain", "unk")].append(x)
        return {d: agg(R) for d, R in g.items()}
    base_dom = b5_by_domain(base)
    for r in rows:
        r["d_cs_main"] = r["cs_main"] - b["cs_main"]
        r["d_b5"] = (r["b5_wer"] - b["b5_wer"]) if r["b5_wer"] is not None and b["b5_wer"] is not None else None
        r["d_nwer"] = r["nwer"] - b["nwer"]
        reasons = []
        if r["system"] != args.baseline:
            if r["d_b5"] is not None and r["d_b5"] > args.max_b5_delta:
                reasons.append(f"B5 WER +{r['d_b5']:.2f}")
            if r["d_nwer"] > args.max_nwer_delta:
                reasons.append(f"N-WER +{r['d_nwer']:.2f}")
            worst = None
            for d, a in b5_by_domain(data[r["system"]][0]).items():
                bd = base_dom.get(d)
                if bd and bd["n"] >= args.min_domain_tokens and a["wer"] is not None:
                    dl = a["wer"] - bd["wer"]
                    if worst is None or dl > worst[1]:
                        worst = (d, dl)
            r["b5_worst_domain"] = worst
            if worst and worst[1] > args.max_b5_domain_delta:
                reasons.append(f"B5 domain {worst[0]} +{worst[1]:.2f}")
            if r["rtf"] and base_rtf and r["rtf"] > args.max_rtf_ratio * base_rtf:
                reasons.append(f"RTF ×{r['rtf'] / base_rtf:.2f}")
            ci = r.get("d_cs_main_ci")
            r["significant"] = bool(ci and ci[1] < 0)
            if not r["significant"]:
                reasons.append("ΔCS-WER chưa có ý nghĩa (CI chứa 0)")
        r["reasons"] = reasons
        r["eligible"] = r["system"] != args.baseline and not reasons

    elig = sorted([r for r in rows if r["eligible"]], key=lambda r: (r["cs_main"], r["cs_b3"] or 0))
    for i, r in enumerate(elig, 1):
        r["rank"] = i
    rows.sort(key=lambda r: (r["system"] != args.baseline, not r["eligible"], r["cs_main"]))
    winner = elig[0] if elig else None

    def f(x, d=2):
        return "–" if x is None else f"{x:.{d}f}"

    def ci(c):
        return "" if not c else f" [{c[0]:+.2f}, {c[1]:+.2f}]"

    def d_b5_txt(r):
        return "–" if r["d_b5"] is None else f"{r['d_b5']:+.2f}"

    hdr = ["Hệ thống", "CS-WER B1+B2", "Δ vs baseline [CI95]", "CS-WER lenient", "N-WER B1–B4", "CS-WER B3", "CS-WER B4",
           "B5 WER (Δ)", "Term R / P", "B5 chèn sai /100 utt", "RTF", "Kết luận"]
    trows = []
    for r in rows:
        verdict = ("baseline" if r["system"] == args.baseline else
                   f"#{r['rank']} đủ điều kiện" if r["eligible"] else "loại: " + "; ".join(r["reasons"]))
        trows.append([r["system"], f(r["cs_main"]),
                      "–" if r["system"] == args.baseline else f"{r['d_cs_main']:+.2f}{ci(r.get('d_cs_main_ci'))}",
                      f(r["cs_main_len"]), f(r["nwer"]), f(r["cs_b3"]), f(r["cs_b4"]),
                      f"{f(r['b5_wer'])} ({d_b5_txt(r)})",
                      f"{f(r['term_r'], 1)} / {f(r['term_p'], 1)}", f(r["b5_false"]), f(r["rtf"], 3), verdict])

    lead = (f"Phương án tốt nhất hiện tại: {winner['system']} — CS-WER B1+B2 {winner['cs_main']:.2f}% "
            f"({winner['d_cs_main']:+.2f} điểm so với baseline, CI95{ci(winner['d_cs_main_ci'])}), B5 WER Δ {f(winner['d_b5'])}."
            if winner else "Chưa có hệ thống nào vượt baseline mà vẫn giữ các ràng buộc.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Leaderboard (baseline: {args.baseline})", "", lead, "",
          "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)]
    md += ["| " + " | ".join(str(x) for x in t) + " |" for t in trows]
    md += ["", f"Luật: loại nếu B5 WER tăng > {args.max_b5_delta}, N-WER tăng > {args.max_nwer_delta}, RTF > {args.max_rtf_ratio}× baseline, "
           f"một domain B5 tăng > {args.max_b5_domain_delta}, hoặc CI95 của ΔCS-WER chứa 0. Bootstrap theo người nói, "
           f"{args.iters} lần. CS-WER lenient − strict = lỗi chính tả (nghe đúng, viết sai) mà hậu xử lý có thể sửa."]
    (out / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "leaderboard.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["system", "cs_wer_b1b2", "d_cs", "d_cs_ci_lo", "d_cs_ci_hi", "cs_wer_lenient", "n_wer", "cs_wer_b3",
                    "cs_wer_b4", "b5_wer", "d_b5", "term_recall", "term_precision", "b5_false_per100", "rtf", "eligible", "rank", "reasons"])
        for r in rows:
            c = r.get("d_cs_main_ci") or [None, None]
            w.writerow([r["system"], r["cs_main"], r["d_cs_main"], c[0], c[1], r["cs_main_len"], r["nwer"], r["cs_b3"], r["cs_b4"],
                        r["b5_wer"], r["d_b5"], r["term_r"], r["term_p"], r["b5_false"], r["rtf"], r["eligible"], r.get("rank"),
                        "; ".join(r["reasons"])])
    plot_rows = []
    for r in rows:
        c = r.get("d_cs_main_ci")
        lo, hi = (b["cs_main"] + c[0], b["cs_main"] + c[1]) if c else (None, None)
        plot_rows.append((r["system"], r["cs_main"], lo, hi))
    html = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Leaderboard bench CS</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">'
            f'<h1>Leaderboard bench code-switching</h1><p class="lead">{escape(lead)}</p>'
            f'<h2>CS-WER B1+B2 (strict)</h2><p class="lead">Chấm = CS-WER; thanh ngang = CI95 từ bootstrap cặp theo người nói; '
            f'vạch đứt = baseline {escape(args.baseline)}.</p><div class="card">{dotplot(plot_rows, b["cs_main"])}</div>'
            f'<h2>Bảng đầy đủ</h2><div class="card" style="overflow-x:auto">{table(hdr, trows)}</div>'
            f'<p class="note">{escape(md[-1])}</p></div></div></body></html>')
    (out / "leaderboard.html").write_text(html, encoding="utf-8")
    print(lead)
    print(f"Đã ghi {out}/leaderboard.md, .csv, .html")


if __name__ == "__main__":
    main()
