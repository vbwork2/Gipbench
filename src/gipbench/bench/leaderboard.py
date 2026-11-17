#!/usr/bin/env python3
"""Gom kết quả mọi hệ thống, so cặp với baseline bằng bootstrap theo người nói, áp luật chọn.

  python -m gipbench.bench.leaderboard --results results --baseline e0_baseline [--out reports/leaderboard]

Mỗi thư mục results/<system>/ cần utt_scores.jsonl (từ score.py); decode_summary.json (từ decode.py) là tuỳ chọn.

Luật chọn (sửa ngưỡng bằng tham số):
  1. Kiểm định không kém hơn trên B5: loại nếu CẬN TRÊN CI95 (bootstrap cặp theo người nói) của ΔWER B5 > --b5-margin
     (điểm tuyệt đối, mặc định 0.5). Không còn so điểm ước lượng với ngưỡng nhỏ hơn độ rộng CI.
  2. Tương tự cho ΔN-WER (B1–B4) với --nwer-margin (mặc định 0.5).
  2b. Loại nếu WER của bất kỳ domain nào trong B5 tăng > --max-b5-domain-delta (chặn suy giảm cục bộ).
  3. Loại nếu RTF > --max-rtf-ratio × RTF baseline (khi có decode_summary.json).
  4. Chỉ coi là cải thiện khi CI 95% ghép cặp của Δ metric chính (B1+B2) không chứa 0 và đúng chiều.
     Metric chính chọn bằng --main-metric: term_acc_macro (mặc định, accuracy theo thuật ngữ rồi trung bình đều),
     span_acc (accuracy theo span) hoặc cs_wer. Accuracy cao hơn là tốt, CS-WER thấp hơn là tốt.
  5. Xếp hạng hệ thống đủ điều kiện theo metric chính (B1+B2), hoà thì CS-WER B3 thấp hơn thắng.
Nếu utt_scores có lát tần suất train (score.py --train-manifest), bảng thêm hai cột span accuracy: unseen và real_5+
(B1+B2).
"""
import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from html import escape
from pathlib import Path

from gipbench.bench.report.eda import CSS
from gipbench.core.manifest import CS_SPLITS
from gipbench.core.svgcharts import table
from gipbench.core.scoring import SPAN_METRICS, agg, freq_slices, term_weights

MAIN_KEY = {"term_acc_macro": "acc_macro", "span_acc": "span_acc", "cs_wer": "cs_main"}  # --main-metric -> khoá METRICS
MAIN_LABEL = {"term_acc_macro": "Term acc (macro)", "span_acc": "Span acc", "cs_wer": "CS-WER"}

METRICS = [  # (khoá, nhóm split, metric của agg)
    ("cs_main", ["B1", "B2"], "cs_wer"),
    ("cs_main_len", ["B1", "B2"], "cs_wer_lenient"),
    ("nwer", CS_SPLITS, "n_wer"),
    ("wer_main", ["B1", "B2"], "wer"),
    ("cs_b3", ["B3"], "cs_wer"),
    ("cs_b4", ["B4"], "cs_wer"),
    ("b5_wer", ["B5"], "wer"),
    ("acc_macro", ["B1", "B2"], "term_acc_macro"),
    ("span_acc", ["B1", "B2"], "span_acc"),
    ("span_cer", ["B1", "B2"], "span_cer"),
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
        by[base[i].get("boot_unit") or base[i]["speaker"]].append(i)  # speaker_cluster nếu bench có (score.py)
    spk = list(by)
    if len(spk) < 2:
        return None
    rnd = random.Random(seed)
    ws = m in SPAN_METRICS
    # trọng số term_acc_macro cố định theo tập đầy đủ (baseline, các id chung), dùng chung cho hai hệ: macro tính lại
    # theo mẫu thì lệch lên vì thuật ngữ hiếm rơi khỏi mẫu
    tw = term_weights([base[i] for i in ids]) if m == "term_acc_macro" else None
    diffs = []
    for _ in range(iters):
        sid = [i for s in (rnd.choice(spk) for _ in spk) for i in by[s]]
        a = agg([other[i] for i in sid], ws, tw)[m]
        b = agg([base[i] for i in sid], ws, tw)[m]
        if a is not None and b is not None:
            diffs.append(a - b)
    diffs.sort()
    return [diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]] if diffs else None


def dotplot(rows, base_val, width=640, label_w=150, row_h=30):
    """Điểm + whisker CI 95% của metric chính (B1+B2) từng hệ thống; vạch dọc = baseline."""
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
    ap.add_argument("--main-metric", choices=list(MAIN_KEY), default="term_acc_macro",
                    help="metric chính (B1+B2) để xếp hạng và kiểm định: accuracy cao hơn là tốt, cs_wer thấp hơn là tốt")
    ap.add_argument("--b5-margin", type=float, default=0.5,
                    help="loại nếu cận trên CI95 ghép cặp của ΔWER B5 > ngưỡng này (kiểm định không kém hơn)")
    ap.add_argument("--nwer-margin", type=float, default=0.5,
                    help="như --b5-margin, cho ΔN-WER (B1–B4)")
    ap.add_argument("--max-rtf-ratio", type=float, default=1.2)
    ap.add_argument("--max-b5-domain-delta", type=float, default=0.5,
                    help="loại nếu WER của BẤT KỲ domain nào trong B5 tăng quá ngưỡng này (domain >= --min-domain-tokens)")
    ap.add_argument("--min-domain-tokens", type=int, default=500)
    ap.add_argument("--systems", nargs="*", default=None,
                    help="chỉ xếp các system này (tên thư mục trong --results); mặc định: mọi thư mục có utt_scores.jsonl")
    ap.add_argument("--rules", default=None,
                    help="YAML ngưỡng luật chọn (configs/bench/rules.yaml): đặt mặc định cho các tham số cùng tên; "
                         "tham số truyền trên dòng lệnh vẫn thắng")
    pre, _ = ap.parse_known_args()
    if pre.rules:
        import yaml
        ap.set_defaults(**{k.replace("-", "_"): v for k, v in
                           (yaml.safe_load(Path(pre.rules).read_text(encoding="utf-8")) or {}).items()})
    args = ap.parse_args()

    root = Path(args.results)
    systems = sorted(d.name for d in root.iterdir() if (d / "utt_scores.jsonl").exists())
    if args.systems is not None:
        missing = [s for s in args.systems if s not in systems]
        if missing:
            sys.exit(f"Thiếu kết quả đã chấm của {missing} trong {root}")
        systems = sorted(args.systems)
    if args.baseline not in systems:
        sys.exit(f"Không thấy baseline {args.baseline} trong {root}")
    data = {s: load(root / s) for s in systems}
    base, base_rtf = data[args.baseline]

    mm = args.main_metric
    up = mm != "cs_wer"  # accuracy: cao hơn là tốt; cs_wer: thấp hơn là tốt
    label = MAIN_LABEL[mm]
    rows = []
    for s in systems:
        recs, rtf = data[s]
        R = list(recs.values())
        row = {"system": s, "rtf": rtf}
        for k, splits, m in METRICS:
            row[k] = metric(R, splits, m)
        row["main"] = row[MAIN_KEY[mm]]
        row["fs"] = freq_slices([x for x in R if x["split"] in ("B1", "B2")])
        if s != args.baseline:
            row["d_main_ci"] = paired_ci(base, recs, ["B1", "B2"], mm, args.iters)
            row["d_cs_main_ci"] = row["d_main_ci"] if mm == "cs_wer" else paired_ci(base, recs, ["B1", "B2"], "cs_wer", args.iters)
            row["d_b5_ci"] = paired_ci(base, recs, ["B5"], "wer", args.iters)
            row["d_nwer_ci"] = paired_ci(base, recs, CS_SPLITS, "n_wer", args.iters)
        rows.append(row)
    b = next(r for r in rows if r["system"] == args.baseline)
    if b["main"] is None:
        sys.exit(f"Baseline thiếu chỉ số theo span ({mm}): chấm lại bằng score.py bản mới")
    has_fs = any(r["fs"] for r in rows)

    def b5_by_domain(recs):
        g = defaultdict(list)
        for x in recs.values():
            if x["split"] == "B5":
                g[x.get("domain", "unk")].append(x)
        return {d: agg(R) for d, R in g.items()}
    base_dom = b5_by_domain(base)
    for r in rows:
        r["d_main"] = (r["main"] - b["main"]) if r["main"] is not None else None
        r["d_cs_main"] = r["cs_main"] - b["cs_main"]
        r["d_b5"] = (r["b5_wer"] - b["b5_wer"]) if r["b5_wer"] is not None and b["b5_wer"] is not None else None
        r["d_nwer"] = r["nwer"] - b["nwer"]
        reasons = []
        if r["system"] != args.baseline:
            # kiểm định không kém hơn: cận trên CI95 ghép cặp của Δ (tăng = xấu) phải <= margin
            for key, d, name, margin in (("d_b5_ci", r["d_b5"], "B5 WER", args.b5_margin),
                                         ("d_nwer_ci", r["d_nwer"], "N-WER", args.nwer_margin)):
                if d is None:
                    continue
                c = r.get(key)
                if not c:
                    reasons.append(f"{name}: thiếu CI (ít hơn 2 người nói)")
                elif c[1] > margin:
                    reasons.append(f"{name} không chứng minh được không kém hơn (cận trên CI95 Δ {c[1]:+.2f} > {margin:g})")
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
            ci = r.get("d_main_ci")
            r["significant"] = bool(ci and (ci[0] > 0 if up else ci[1] < 0))
            if not r["significant"]:
                reasons.append(f"Δ{label} chưa có ý nghĩa (CI chứa 0 hoặc sai chiều)")
        r["reasons"] = reasons
        r["eligible"] = r["system"] != args.baseline and not reasons

    def sort_main(r):  # nhỏ hơn = xếp trên
        v = r["main"] if r["main"] is not None else (float("-inf") if up else float("inf"))
        return -v if up else v
    elig = sorted([r for r in rows if r["eligible"]], key=lambda r: (sort_main(r), r["cs_b3"] or 0))
    for i, r in enumerate(elig, 1):
        r["rank"] = i
    rows.sort(key=lambda r: (r["system"] != args.baseline, not r["eligible"], sort_main(r)))
    winner = elig[0] if elig else None

    def f(x, d=2):
        return "–" if x is None else f"{x:.{d}f}"

    def ci(c):
        return "" if not c else f" [{c[0]:+.2f}, {c[1]:+.2f}]"

    def d_b5_txt(r):
        return "–" if r["d_b5"] is None else f"{r['d_b5']:+.2f}"

    def fs_txt(r, k):
        v = r["fs"].get(k)
        return "–" if not v or v["span_acc"] is None else f"{v['span_acc']:.2f} (n={v['n_span']})"

    hdr = ["Hệ thống", f"{label} B1+B2", "Δ vs baseline [CI95]"] + (["CS-WER B1+B2"] if up else []) + [
           "CS-WER lenient", "N-WER B1–B4", "CS-WER B3", "CS-WER B4", "B5 WER (Δ)", "Term R / P",
           "B5 chèn sai /100 utt"] + (["Span acc unseen", "Span acc real_5+"] if has_fs else []) + ["RTF", "Kết luận"]
    trows = []
    for r in rows:
        verdict = ("baseline" if r["system"] == args.baseline else
                   f"#{r['rank']} đủ điều kiện" if r["eligible"] else "loại: " + "; ".join(r["reasons"]))
        trows.append([r["system"], f(r["main"]),
                      "–" if r["system"] == args.baseline or r["d_main"] is None else f"{r['d_main']:+.2f}{ci(r.get('d_main_ci'))}"]
                     + ([f(r["cs_main"])] if up else []) +
                     [f(r["cs_main_len"]), f(r["nwer"]), f(r["cs_b3"]), f(r["cs_b4"]),
                      f"{f(r['b5_wer'])} ({d_b5_txt(r)})",
                      f"{f(r['term_r'], 1)} / {f(r['term_p'], 1)}", f(r["b5_false"])]
                     + ([fs_txt(r, "unseen"), fs_txt(r, "real_5+")] if has_fs else []) + [f(r["rtf"], 3), verdict])

    lead = (f"Phương án tốt nhất hiện tại: {winner['system']} — {label} B1+B2 {winner['main']:.2f}% "
            f"({winner['d_main']:+.2f} điểm so với baseline, CI95{ci(winner['d_main_ci'])}), B5 WER Δ {f(winner['d_b5'])}."
            if winner else "Chưa có hệ thống nào vượt baseline mà vẫn giữ các ràng buộc.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# Leaderboard (baseline: {args.baseline})", "", lead, "",
          "| " + " | ".join(hdr) + " |", "|" + " --- |" * len(hdr)]
    md += ["| " + " | ".join(str(x) for x in t) + " |" for t in trows]
    md += ["", f"Luật: loại nếu cận trên CI95 ghép cặp của ΔWER B5 > {args.b5_margin:g} hoặc của ΔN-WER (B1–B4) > {args.nwer_margin:g} "
           f"(kiểm định không kém hơn), RTF > {args.max_rtf_ratio}× baseline, một domain B5 tăng > {args.max_b5_domain_delta}, "
           f"hoặc CI95 ghép cặp của Δ{label} (B1+B2) chứa 0 / sai chiều ({'cao hơn' if up else 'thấp hơn'} là tốt). "
           f"Xếp hạng theo {label} B1+B2, hoà thì CS-WER B3 thấp hơn. Bootstrap theo người nói, {args.iters} lần. "
           f"CS-WER lenient − strict = lỗi chính tả (nghe đúng, viết sai) mà hậu xử lý có thể sửa; chỉ có nghĩa khi catalog có "
           f"cột phonetic (nếu không, lenient = strict)."]
    (out / "leaderboard.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out / "leaderboard.csv", "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["system", "cs_wer_b1b2", "d_cs", "d_cs_ci_lo", "d_cs_ci_hi", "cs_wer_lenient", "n_wer", "cs_wer_b3",
                    "cs_wer_b4", "b5_wer", "d_b5", "term_recall", "term_precision", "b5_false_per100", "rtf", "eligible", "rank", "reasons",
                    "main_metric", "main_value", "d_main", "d_main_ci_lo", "d_main_ci_hi", "term_acc_macro", "span_acc", "span_cer",
                    "d_b5_ci_hi", "d_nwer", "d_nwer_ci_hi"])
        for r in rows:
            c = r.get("d_cs_main_ci") or [None, None]
            cm = r.get("d_main_ci") or [None, None]
            w.writerow([r["system"], r["cs_main"], r["d_cs_main"], c[0], c[1], r["cs_main_len"], r["nwer"], r["cs_b3"], r["cs_b4"],
                        r["b5_wer"], r["d_b5"], r["term_r"], r["term_p"], r["b5_false"], r["rtf"], r["eligible"], r.get("rank"),
                        "; ".join(r["reasons"]), mm, r["main"], r["d_main"], cm[0], cm[1], r["acc_macro"], r["span_acc"],
                        r["span_cer"], (r.get("d_b5_ci") or [None, None])[1], r["d_nwer"], (r.get("d_nwer_ci") or [None, None])[1]])
    plot_rows = []
    for r in rows:
        c = r.get("d_main_ci")
        lo, hi = (b["main"] + c[0], b["main"] + c[1]) if c else (None, None)
        plot_rows.append((r["system"], r["main"], lo, hi))
    html = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Leaderboard bench CS</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">'
            f'<h1>Leaderboard bench code-switching</h1><p class="lead">{escape(lead)}</p>'
            f'<h2>{label} B1+B2 (strict)</h2><p class="lead">Chấm = {label}; thanh ngang = CI95 từ bootstrap cặp theo người nói; '
            f'vạch đứt = baseline {escape(args.baseline)}.</p><div class="card">{dotplot(plot_rows, b["main"])}</div>'
            f'<h2>Bảng đầy đủ</h2><div class="card" style="overflow-x:auto">{table(hdr, trows)}</div>'
            f'<p class="note">{escape(md[-1])}</p></div></div></body></html>')
    (out / "leaderboard.html").write_text(html, encoding="utf-8")
    print(lead)
    print(f"Đã ghi {out}/leaderboard.md, .csv, .html")


if __name__ == "__main__":
    main()
