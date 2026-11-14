#!/usr/bin/env python3
"""EDA cho bench code-switching: xuất report HTML + summary JSON + bảng Markdown để dán vào report.

  python eda.py --bench data/bench.jsonl --catalog configs/catalog.tsv \
      [--train data/train_cs.jsonl ...] [--audio-stats] --out reports/eda_v1.0

Report gồm: tổng quan theo split, kiểm tra mục tiêu (giờ, người nói, số span, số thuật ngữ),
phân bố thời lượng, miền/giới/kênh, độ tập trung người nói, hồ sơ code-switching
(mức CS, vị trí, độ dài span, mức Việt hoá), thuật ngữ (category, đuôi dài, top 30, độ phủ catalog),
mức trùng với train (nếu có --train), chất lượng audio (nếu có --audio-stats).
"""
import argparse
import datetime as dt
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import svgcharts as C  # noqa: E402
from lib.manifest import BENCH_SPLITS, CS_SPLITS, TARGET_HOURS, read_jsonl, span_meta  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, normalize_ref, plain_text  # noqa: E402

CAT4 = ["--s1", "--s2", "--s3", "--s4"]
LABELS = {"north": "miền Bắc", "central": "miền Trung", "south": "miền Nam", "other": "khác", "unk": "chưa rõ",
          "f": "nữ", "m": "nam", "u": "chưa rõ", "wideband": "wideband (≥ 16 kHz)", "phone8k": "điện thoại 8 kHz",
          "strong": "Việt hoá mạnh", "mid": "trung gian", "native": "gần bản ngữ"}
ORD4 = ["--q1", "--q2", "--q3", "--q4"]
TARGETS = {"b1_speakers": 150, "cs_spans": 4000, "distinct_terms": 1000}

CSS = """
.viz-root{color-scheme:light;--surface:#fcfcfb;--page:#f4f3f0;--text:#0b0b0b;--text2:#52514e;--muted:#8a8984;
--grid:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--q1:#9ec5f4;--q2:#5598e7;--q3:#256abf;--q4:#0d366b;
--good:#0b7a3e;--bad:#c62828}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;--surface:#1a1a19;
--page:#111110;--text:#f0efec;--text2:#c3c2b7;--muted:#8f8e86;--grid:#2e2e2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;
--q1:#184f95;--q2:#256abf;--q3:#5598e7;--q4:#9ec5f4;--good:#4cc27e;--bad:#ef6b6b}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface:#1a1a19;--page:#111110;--text:#f0efec;--text2:#c3c2b7;
--muted:#8f8e86;--grid:#2e2e2b;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--q1:#184f95;--q2:#256abf;--q3:#5598e7;
--q4:#9ec5f4;--good:#4cc27e;--bad:#ef6b6b}
body{margin:0;background:var(--page);}
.viz-root{background:var(--page);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
padding:24px 16px 64px}
.wrap{max-width:980px;margin:0 auto}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 6px}h3{font-size:15px;margin:18px 0 6px}
.sub{color:var(--text2);margin:0 0 16px}.lead{margin:4px 0 12px;color:var(--text2)}
.card{background:var(--surface);border:1px solid var(--grid);border-radius:10px;padding:14px 16px;margin:10px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.kpi .v{font-size:26px;font-weight:600}.kpi .l{color:var(--text2);font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--grid);text-align:left}
th{color:var(--text2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
details{margin-top:6px}summary{cursor:pointer;color:var(--text2);font-size:13px}
svg .lab{fill:var(--text2);font-size:12px}svg .strong{fill:var(--text);font-weight:600}svg .val{fill:var(--text);font-size:12px}
svg .tick{fill:var(--muted);font-size:10px}svg .inbar{fill:#fff;font-size:11px;font-weight:600}svg .inbar.dark{fill:#0b0b0b}
svg .axis{stroke:var(--grid);stroke-width:1}svg .target{stroke:var(--text);stroke-width:2}
svg path:hover,svg rect:hover{opacity:.8}
.ok{color:var(--good);font-weight:600}.no{color:var(--bad);font-weight:600}
.note{font-size:13px;color:var(--text2)}code{font-size:12px}
@media (max-width:600px){.viz-root{padding:16px}}
"""


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600


def audio_stats(path):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if len(x) < sr // 10:
        return None
    clip = float(np.mean(np.abs(x) >= 0.999))
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    fl = int(0.025 * sr)
    n = len(x) // fl
    e = (x[: n * fl].reshape(n, fl) ** 2).mean(axis=1) + 1e-12
    p95, p10 = np.percentile(e, 95), np.percentile(e, 10)
    return {"sr": sr, "clip_pct": 100 * clip, "rms_dbfs": 20 * np.log10(rms),
            "snr_db": float(10 * np.log10(p95 / p10)), "silence_pct": float(100 * np.mean(e < p10 * 4))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--train", nargs="*", default=[])
    ap.add_argument("--audio-stats", action="store_true")
    ap.add_argument("--out", default="reports/eda")
    ap.add_argument("--title", default="EDA bộ benchmark code-switching Việt–Anh")
    args = ap.parse_args()

    cat = Catalog.load(args.catalog)
    rows = read_jsonl(args.bench)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    bench_hash = hashlib.sha256(Path(args.bench).read_bytes()).hexdigest()[:12]
    splits = [s for s in BENCH_SPLITS if any(r["split"] == s for r in rows)]
    by = defaultdict(list)
    for r in rows:
        by[r["split"]].append(r)

    # ---------------------------------------------------------------- tách span
    span_recs = []  # mỗi span: split, term, category, viet_level, position, length, utt id
    utt_info = {}
    for r in rows:
        toks, spans = normalize_ref(r["text_tagged"], cat)
        n_en = sum(t.span >= 0 for t in toks)
        utt_info[r["id"]] = {"n_tok": len(toks), "n_en": n_en, "n_spans": len(spans)}
        for si, sp in enumerate(spans):
            idx = [k for k, t in enumerate(toks) if t.span == si]
            pos = "đầu câu" if idx[0] == 0 else ("cuối câu" if idx[-1] == len(toks) - 1 else "giữa câu")
            term = cat.terms.get(sp)
            span_recs.append({"split": r["split"], "term": " ".join(sp), "category": term.category if term else "(ngoài catalog)",
                              "viet_level": span_meta(r, si).get("viet_level", "chưa gán"),
                              "position": pos, "len": len(sp), "reserved": bool(term and term.reserved_b3), "id": r["id"]})

    # ---------------------------------------------------------------- tổng quan theo split
    overview = []
    for s in splits:
        R = by[s]
        sp = [x for x in span_recs if x["split"] == s]
        ntok = sum(utt_info[r["id"]]["n_tok"] for r in R)
        nen = sum(utt_info[r["id"]]["n_en"] for r in R)
        spk_h = Counter()
        for r in R:
            spk_h[r["speaker"]] += float(r["duration"])
        tot = sum(spk_h.values())
        overview.append({
            "split": s, "utt": len(R), "hours": hours(R), "target_hours": TARGET_HOURS.get(s),
            "speakers": len(spk_h), "sources": len({r["source"] for r in R}),
            "mean_dur": statistics.mean(float(r["duration"]) for r in R),
            "spans": len(sp), "distinct_terms": len({x["term"] for x in sp}),
            "en_token_pct": pct(nen, ntok),
            "multi_span_pct": pct(sum(utt_info[r["id"]]["n_spans"] >= 2 for r in R), len(R)),
            "top_speaker_pct": pct(max(spk_h.values()), tot),
        })
    cs_spans = [x for x in span_recs if x["split"] in CS_SPLITS]
    term_freq = Counter(x["term"] for x in cs_spans)

    # ---------------------------------------------------------------- trùng với train
    train_freq = None
    if args.train:
        bench_terms = {tuple(t.split()) for t in term_freq}
        maxn = max([len(t) for t in bench_terms] + [1])
        train_freq = Counter()
        for tp in args.train:
            for r in read_jsonl(tp):
                toks = cat.apply(basic_tokens(plain_text(r.get("text_tagged") or r.get("text") or "")))
                for n in range(1, maxn + 1):
                    for i in range(len(toks) - n + 1):
                        k = tuple(toks[i:i + n])
                        if k in bench_terms:
                            train_freq[" ".join(k)] += 1

    # ---------------------------------------------------------------- audio
    astats = {}
    if args.audio_stats:
        for r in rows:
            try:
                a = audio_stats(r["_audio_path"])
                if a:
                    astats[r["id"]] = a
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------------- kiểm tra mục tiêu
    ov = {o["split"]: o for o in overview}
    checks = []
    for s in splits:
        if TARGET_HOURS.get(s):
            checks.append((f"{s}: số giờ", f"{ov[s]['hours']:.2f} h", f"≥ {TARGET_HOURS[s]} h", ov[s]["hours"] >= TARGET_HOURS[s]))
    if "B1" in ov:
        checks.append(("B1: số người nói", ov["B1"]["speakers"], f"≥ {TARGETS['b1_speakers']}", ov["B1"]["speakers"] >= TARGETS["b1_speakers"]))
    checks.append(("B1–B4: tổng span tiếng Anh", len(cs_spans), f"≥ {TARGETS['cs_spans']}", len(cs_spans) >= TARGETS["cs_spans"]))
    checks.append(("B1–B4: số thuật ngữ khác nhau", len(term_freq), f"≥ {TARGETS['distinct_terms']}", len(term_freq) >= TARGETS["distinct_terms"]))
    for s in ("B1", "B4"):
        if s in by:
            h = Counter()
            for r in by[s]:
                h[r.get("region", "unk")] += float(r["duration"])
            mn = min((h.get(k, 0) for k in ("north", "central", "south")))
            checks.append((f"{s}: miền ít nhất (B/T/N)", f"{pct(mn, sum(h.values())):.0f}%", "≥ 20%", pct(mn, sum(h.values())) >= 20))
    for s in ("B2", "B5"):
        if s in by:
            p8 = pct(sum(float(r["duration"]) for r in by[s] if r.get("channel") == "phone8k"), hours(by[s]) * 3600)
            checks.append((f"{s}: tỉ lệ audio điện thoại 8 kHz", f"{p8:.0f}%", "> 0% (có kênh thoại)", p8 > 0))
    for s, o in ov.items():
        if o["top_speaker_pct"] > 10 and s != "B4":
            checks.append((f"{s}: người nói chiếm nhiều nhất", f"{o['top_speaker_pct']:.0f}%", "≤ 10% số giờ", False))
    if train_freq is not None and "B3" in by:
        b3 = {x["term"] for x in span_recs if x["split"] == "B3"}
        seen = sum(1 for t in b3 if train_freq.get(t, 0) > 0)
        checks.append(("B3: thuật ngữ đã xuất hiện trong train", seen, "= 0", seen == 0))
    n_unk = sum(1 for x in cs_spans if x["category"] == "(ngoài catalog)")
    if cat.terms:
        checks.append(("Span ngoài catalog", f"{pct(n_unk, len(cs_spans)):.1f}%", "≤ 5%", pct(n_unk, len(cs_spans)) <= 5))
    nvl = sum(1 for x in cs_spans if x["viet_level"] == "chưa gán")
    checks.append(("Span chưa gán mức Việt hoá", f"{pct(nvl, len(cs_spans)):.1f}%", "≤ 5%", pct(nvl, len(cs_spans)) <= 5))

    # ---------------------------------------------------------------- dựng HTML
    H = []
    total_h = hours(rows)
    H.append(f'<h1>{escape(args.title)}</h1><p class="sub">Manifest <code>{escape(Path(args.bench).name)}</code> · '
             f'sha256 {bench_hash} · tạo lúc {dt.datetime.now():%Y-%m-%d %H:%M}</p>')
    n_fail = sum(not c[3] for c in checks)
    H.append(f'<p class="lead">Bench có {len(rows):,} utterance, {total_h:.2f} giờ, {len({r["speaker"] for r in rows})} người nói, '
             f'{len(cs_spans):,} span tiếng Anh thuộc {len(term_freq):,} thuật ngữ; '
             f'{"đạt mọi mục tiêu" if not n_fail else f"còn {n_fail}/{len(checks)} mục tiêu chưa đạt"}.</p>')
    H.append('<div class="kpis">' + "".join(
        f'<div class="card kpi"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in [
            (f"{total_h:.2f} h", "tổng thời lượng"), (f"{len(rows):,}", "utterance"),
            (f"{len({r['speaker'] for r in rows})}", "người nói"), (f"{len(cs_spans):,}", "span tiếng Anh (B1–B4)"),
            (f"{len(term_freq):,}", "thuật ngữ khác nhau")]) + "</div>")

    H.append("<h2>Kiểm tra mục tiêu</h2>")
    H.append('<div class="card">' + C.table(["Tiêu chí", "Hiện có", "Mục tiêu", "Trạng thái"],
             [[a, b, c, '<span class="ok">✓ Đạt</span>' if ok else '<span class="no">✗ Chưa đạt</span>'] for a, b, c, ok in checks],
             num_from=1) + "</div>")

    H.append("<h2>Tổng quan theo split</h2>")
    H.append('<p class="lead">Vạch đen là số giờ mục tiêu của từng split.</p>')
    H.append('<div class="card">' + C.hbar([(o["split"], round(o["hours"], 2)) for o in overview], " h",
             targets=[o["target_hours"] for o in overview]) + "</div>")
    H.append('<div class="card">' + C.table(
        ["Split", "Utt", "Giờ", "Người nói", "Nguồn", "TB (s)", "Span EN", "Thuật ngữ", "% token EN", "% utt ≥ 2 span", "Người nói lớn nhất (% giờ)"],
        [[o["split"], o["utt"], round(o["hours"], 2), o["speakers"], o["sources"], round(o["mean_dur"], 1), o["spans"],
          o["distinct_terms"], round(o["en_token_pct"], 1), round(o["multi_span_pct"], 1), round(o["top_speaker_pct"], 1)]
         for o in overview]) + "</div>")

    H.append("<h2>Phân bố thời lượng utterance</h2>")
    edges = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
    hs = []
    for s in splits:
        cnt = [0] * (len(edges) - 1)
        for r in by[s]:
            d = float(r["duration"])
            for i in range(len(edges) - 1):
                if edges[i] <= d < edges[i + 1] or (i == len(edges) - 2 and d >= edges[-1]):
                    cnt[i] += 1
                    break
        hs.append(f'<div class="card">{C.hist(cnt, edges, s)}</div>')
    H.append('<div class="grid">' + "".join(hs) + "</div>")

    def dist_section(field, title, order, lead):
        H.append(f"<h2>{title}</h2><p class='lead'>{lead}</p>")
        vals = order + sorted({r.get(field, "unk") for r in rows} - set(order))
        vals = vals[:4] if len(vals) <= 4 else vals[:3] + ["khác"]
        data, trows = [], []
        for s in splits:
            h = Counter()
            for r in by[s]:
                v = r.get(field, "unk")
                h[v if v in vals else "khác"] += float(r["duration"]) / 3600
            data.append((s, [h.get(v, 0) for v in vals]))
            trows.append([s] + [round(h.get(v, 0), 2) for v in vals])
        H.append('<div class="card">' + C.stacked100(data, vals, CAT4, unit=" h", labels=LABELS) +
                 "<details><summary>Bảng số liệu (giờ)</summary>" + C.table(["Split"] + [LABELS.get(v, v) for v in vals], trows) + "</details></div>")

    dist_section("region", "Miền", ["north", "central", "south"], "Tỉ lệ số giờ theo miền của người nói.")
    dist_section("gender", "Giới tính", ["f", "m", "u"], "Tỉ lệ số giờ theo giới tính.")
    dist_section("channel", "Kênh thu", ["wideband", "phone8k"], "Wideband so với âm thanh điện thoại 8 kHz.")

    H.append("<h2>Hồ sơ code-switching</h2>")
    H.append("<p class='lead'>Mức CS = tỉ lệ token tiếng Anh trong utterance: thấp &lt; 15%, vừa 15–35%, cao &gt; 35%.</p>")
    data, trows = [], []
    for s in [x for x in splits if x in CS_SPLITS]:
        b = Counter()
        for r in by[s]:
            u = utt_info[r["id"]]
            p = pct(u["n_en"], u["n_tok"])
            b["thấp" if p < 15 else "vừa" if p <= 35 else "cao"] += 1
        vals = [b["thấp"], b["vừa"], b["cao"]]
        data.append((s, vals))
        trows.append([s] + vals)
    H.append('<div class="card"><h3>Mức code-switching theo split</h3>' + C.stacked100(data, ["thấp", "vừa", "cao"], ORD4[1:]) +
             "<details><summary>Bảng (số utterance)</summary>" + C.table(["Split", "thấp", "vừa", "cao"], trows) + "</details></div>")

    for field, title, order, colors in [("position", "Vị trí span trong câu", ["đầu câu", "giữa câu", "cuối câu"], CAT4),
                                        ("viet_level", "Mức Việt hoá cách phát âm", ["strong", "mid", "native", "chưa gán"], ORD4[::-1])]:
        data, trows = [], []
        for s in [x for x in splits if x in CS_SPLITS]:
            b = Counter(x[field] for x in cs_spans if x["split"] == s)
            vals = [b.get(v, 0) for v in order]
            data.append((s, vals))
            trows.append([s] + vals)
        H.append(f'<div class="card"><h3>{title}</h3>' + C.stacked100(data, order, colors, labels=LABELS) +
                 "<details><summary>Bảng (số span)</summary>" + C.table(["Split"] + [LABELS.get(v, v) for v in order], trows) + "</details></div>")
    ln = Counter(min(x["len"], 3) for x in cs_spans)
    H.append(f'<p class="note">Độ dài span: {pct(ln[1], len(cs_spans)):.0f}% một từ, {pct(ln[2], len(cs_spans)):.0f}% hai từ, '
             f'{pct(ln[3], len(cs_spans)):.0f}% từ ba từ trở lên.</p>')

    H.append("<h2>Thuật ngữ</h2>")
    catc = Counter(x["category"] for x in cs_spans)
    H.append('<div class="card"><h3>Số span theo category</h3>' + C.hbar(catc.most_common(), "") + "</div>")
    buckets = [("1 lần", 1, 1), ("2 lần", 2, 2), ("3–5", 3, 5), ("6–20", 6, 20), ("> 20", 21, 10 ** 9)]
    bk = [(lab, sum(1 for c in term_freq.values() if lo <= c <= hi)) for lab, lo, hi in buckets]
    H.append(f'<div class="card"><h3>Đuôi dài: số thuật ngữ khác nhau theo số lần xuất hiện trong B1–B4</h3>'
             f'<p class="note">{pct(bk[0][1], len(term_freq)):.0f}% thuật ngữ chỉ xuất hiện 1 lần — CS-WER trên nhóm này có độ bất định cao.</p>'
             + C.hbar(bk, "", color="--q3") + "</div>")
    top = term_freq.most_common(30)
    split_of_term = defaultdict(set)
    for x in cs_spans:
        split_of_term[x["term"]].add(x["split"])
    trows = []
    for t, c in top:
        term = cat.terms.get(tuple(t.split()))
        row = [t, term.category if term else "(ngoài catalog)", c, ", ".join(sorted(split_of_term[t]))]
        if train_freq is not None:
            row.append(train_freq.get(t, 0))
        trows.append(row)
    hdr = ["Thuật ngữ", "Category", "Số lần", "Split"] + (["Số lần trong train"] if train_freq is not None else [])
    H.append('<div class="card"><h3>30 thuật ngữ nhiều nhất</h3>' + C.table(hdr, trows, num_from=2) + "</div>")
    if cat.terms:
        cat_normal = [t for t, v in cat.terms.items() if not v.reserved_b3]
        seen = {tuple(t.split()) for t in term_freq}
        H.append(f'<p class="note">Độ phủ catalog: {sum(t in seen for t in cat_normal)}/{len(cat_normal)} thuật ngữ thường '
                 f'({pct(sum(t in seen for t in cat_normal), len(cat_normal)):.0f}%) đã có trong B1–B4; '
                 f'{n_unk} span ({pct(n_unk, len(cs_spans)):.1f}%) nằm ngoài catalog — nên bổ sung vào catalog.</p>')

    if train_freq is not None:
        H.append("<h2>Mức trùng thuật ngữ với dữ liệu train</h2>")
        H.append("<p class='lead'>Tỉ lệ span có thuật ngữ đã xuất hiện trong train bao nhiêu lần. B3 phải là 100% “0 lần”.</p>")
        order = ["0 lần", "1–5", "6–20", "> 20"]
        data, trows = [], []
        for s in [x for x in splits if x in CS_SPLITS]:
            b = Counter()
            for x in cs_spans:
                if x["split"] != s:
                    continue
                c = train_freq.get(x["term"], 0)
                b["0 lần" if c == 0 else "1–5" if c <= 5 else "6–20" if c <= 20 else "> 20"] += 1
            vals = [b.get(v, 0) for v in order]
            data.append((s, vals))
            trows.append([s] + vals)
        H.append('<div class="card">' + C.stacked100(data, order, ORD4) +
                 "<details><summary>Bảng (số span)</summary>" + C.table(["Split"] + order, trows) + "</details></div>")

    if astats:
        H.append("<h2>Chất lượng audio</h2>")
        H.append("<p class='lead'>SNR ước lượng = 10·log10(năng lượng khung p95 / p10); chỉ dùng để so sánh tương đối giữa các split.</p>")
        edges = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60]
        hs, trows = [], []
        for s in splits:
            A = [astats[r["id"]] for r in by[s] if r["id"] in astats]
            if not A:
                continue
            cnt = [0] * (len(edges) - 1)
            for a in A:
                v = min(a["snr_db"], edges[-1] - 0.01)
                for i in range(len(edges) - 1):
                    if edges[i] <= v < edges[i + 1]:
                        cnt[i] += 1
                        break
            hs.append(f'<div class="card">{C.hist(cnt, edges, s + " · SNR ước lượng", xunit=" dB")}</div>')
            srs = Counter(a["sr"] for a in A)
            trows.append([s, len(A), round(statistics.median(a["snr_db"] for a in A), 1),
                          round(statistics.median(a["rms_dbfs"] for a in A), 1),
                          round(pct(sum(a["clip_pct"] > 0.1 for a in A), len(A)), 1),
                          ", ".join(f"{k // 1000} kHz×{v}" for k, v in srs.most_common())])
        H.append('<div class="grid">' + "".join(hs) + "</div>")
        H.append('<div class="card">' + C.table(["Split", "File", "SNR trung vị (dB)", "RMS trung vị (dBFS)", "% file bị clipping", "Sample rate"], trows) + "</div>")

    html = (f'<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>EDA bench CS</title><style>{CSS}</style></head><body><div class="viz-root"><div class="wrap">{"".join(H)}'
            f'</div></div></body></html>')
    (out / "eda_report.html").write_text(html, encoding="utf-8")

    summary = {"bench": str(args.bench), "sha256_12": bench_hash, "generated": dt.datetime.now().isoformat(timespec="seconds"),
               "overview": overview, "checks": [{"criterion": a, "value": str(b), "target": c, "pass": ok} for a, b, c, ok in checks],
               "n_cs_spans": len(cs_spans), "n_distinct_terms": len(term_freq),
               "term_freq_buckets": dict(bk), "category_counts": dict(catc), "top_terms": top}
    (out / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    md = ["| Split | Utt | Giờ (mục tiêu) | Người nói | Span EN | Thuật ngữ | % token EN |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for o in overview:
        md.append(f"| {o['split']} | {o['utt']} | {o['hours']:.2f} ({o['target_hours']}) | {o['speakers']} | {o['spans']} | "
                  f"{o['distinct_terms']} | {o['en_token_pct']:.1f} |")
    md += ["", "| Tiêu chí | Hiện có | Mục tiêu | Đạt |", "| --- | --- | --- | --- |"]
    md += [f"| {a} | {b} | {c} | {'✓' if ok else '✗'} |" for a, b, c, ok in checks]
    (out / "eda_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Đã ghi {out}/eda_report.html, eda_summary.json, eda_summary.md — {n_fail}/{len(checks)} mục tiêu chưa đạt")


if __name__ == "__main__":
    main()
