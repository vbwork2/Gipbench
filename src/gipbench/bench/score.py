#!/usr/bin/env python3
"""Chấm điểm một hệ thống trên bench: WER, CS-WER (strict + lenient), N-WER, term recall/precision,
chèn sai thuật ngữ trên B5, khoảng tin cậy bootstrap theo người nói, và cắt lát theo metadata.

  python -m gipbench.bench.score --bench data/bench.jsonl --catalog configs/catalog.tsv \
      --hyps results/e0_baseline/hyps.jsonl [--out results/e0_baseline] [--train-manifest data/train.jsonl ...]

Ghi ra: utt_scores.jsonl (lỗi từng utterance, leaderboard.py dùng để so sánh cặp),
summary.json, slices.md.

--label-field chọn trường reference (mặc định text_tagged, nhãn theo Spec). B3 báo số chính theo Spec và số phụ
theo nhãn gốc ViMedCSS: chạy thêm một lần với --label-field text_tagged_orig --out <thư mục khác>.

Định nghĩa
  WER     = (S + D + I) / N trên mọi token
  CS-WER  = lỗi trên token thuộc span <en> / số token tiếng Anh trong reference.
            Chèn (I) tính cho span tiếng Anh nếu một token reference kề bên là tiếng Anh.
  N-WER   = lỗi trên token tiếng Việt / số token tiếng Việt
  strict  : hyp phải đúng chính tả Anh (sau bảng biến thể `variants` của catalog)
  lenient : chấp nhận thêm phiên âm Việt trong cột `phonetic` (chẩn đoán: lỗi "nghe đúng, viết sai")
  Chỉ số theo span (strict; bổ sung cho CS-WER, vì CS-WER tính 2 lỗi cho "bilirubin" -> "billy rubin" và coi
  "setup" / "set up" là sai). Hyp của một span = token hyp căn với span đó (C/S của token ref thuộc span + token chèn
  quy về span theo luật kề bên của lib/align.attribute). Mỗi span ghi thêm: hyp, exact, cer.
  exact          = err == 0, hoặc ghép token ref (bỏ dấu cách) == ghép token hyp (vd "setup" = "set up")
  cer            = Levenshtein theo ký tự giữa hai chuỗi đã ghép / số ký tự ref (lưu thô, có thể > 1)
  span_acc       = % span exact (micro, mỗi span một phiếu)
  term_acc_macro = accuracy tính riêng từng thuật ngữ rồi trung bình đều qua các thuật ngữ (thuật ngữ lặp nhiều
                   không chi phối); span_cer = trung bình min(cer, 1) trên các span (%)
--train-manifest (tuỳ chọn): đếm số lần chuỗi token của từng thuật ngữ bench trong text train (cùng pipeline chuẩn hoá,
  bỏ tag; text_tagged nếu có, không thì text), tách dòng thật (train_freq_real) và dòng `synthetic: true`
  (train_freq_synth). Lát cắt: unseen (cả hai = 0), synth_only (real = 0, synth > 0), real_1-4, real_5+.
Term recall    = số lần thuật ngữ reference xuất hiện đúng trong hyp / số span reference
  Term precision = số lần khớp / số thuật ngữ catalog mà hyp đưa ra
Category của span: term_category của câu (vd {"hugo lambert": "name"}, converter ghi cho tên người) trước, rồi
  category trong catalog, không có thì "(ngoài catalog)". slices.md có lát cắt theo category cho B1–B4 và riêng B2.
Bootstrap CI lấy mẫu lại theo speaker_cluster nếu câu có trường này (cụm người nói ước lượng bằng speaker embedding,
  vd bộ consensus không có speaker id), không thì theo speaker. Đơn vị ghi ở boot_unit trong utt_scores.jsonl.
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.align import align, attribute, char_edit_distance, span_hyps
from gipbench.core.manifest import CS_SPLITS, read_jsonl, span_meta
from gipbench.core.textnorm import Catalog, basic_tokens, normalize_hyp, normalize_ref, plain_text

MAIN = ["B1", "B2"]


def count_occ(toks, term):
    n, c, i = len(term), 0, 0
    while i <= len(toks) - n:
        if tuple(toks[i:i + n]) == term:
            c += 1
            i += n
        else:
            i += 1
    return c


def train_term_freq(paths, cat, terms):
    """Đếm số lần mỗi thuật ngữ (tuple token) xuất hiện trong text train, tách dòng thật / dòng synthetic.
    Trả về {"t1 t2": (real, synth)} cho mọi thuật ngữ trong `terms`."""
    first = defaultdict(list)
    for t in terms:
        first[t[0]].append(t)
    cnt = {t: [0, 0] for t in terms}
    for p in paths:
        for r in read_jsonl(p):
            toks = cat.apply(basic_tokens(plain_text(r.get("text_tagged") or r.get("text") or "")), "strict")
            col = 1 if r.get("synthetic") else 0
            for t in {t for tok in set(toks) for t in first.get(tok, ())}:
                cnt[t][col] += count_occ(toks, t)
    return {" ".join(t): tuple(v) for t, v in cnt.items()}


def score_utt(r, hyp_text, cat, label_field="text_tagged", freq=None):
    rec = {"id": r["id"], "split": r["split"], "speaker": r["speaker"],
           "boot_unit": r.get("speaker_cluster") or r["speaker"]}
    for k in ("region", "gender", "channel", "domain"):
        rec[k] = r.get(k, "unk")
    ref_text = r.get(label_field) or r["text_tagged"]
    tcat = r.get("term_category") or {}
    for mode in ("strict", "lenient"):
        rt, spans = normalize_ref(ref_text, cat, mode)
        ht = normalize_hyp(hyp_text, cat, mode)
        ref = [t.tok for t in rt]
        sp = [t.span for t in rt]
        ops = align(ref, ht)
        errs, S, D, I = attribute(ops, sp)
        n_en = sum(s >= 0 for s in sp)
        sfx = "" if mode == "strict" else "_len"
        rec["err_en" + sfx] = sum(v for k, v in errs.items() if k >= 0)
        if mode == "strict":
            rec.update(n=len(ref), n_en=n_en, n_vi=len(ref) - n_en, S=S, D=D, I=I,
                       err=S + D + I, err_vi=errs.get(-1, 0))
            rec["spans"] = []
            sh = span_hyps(ops, sp, ht, len(spans))
            for si, term in enumerate(spans):
                t = cat.terms.get(term)
                ts = " ".join(term)
                rj, hj, e = "".join(term), "".join(sh[si]), errs.get(si, 0)
                sd = {"term": ts, "n": sum(1 for s in sp if s == si), "err": e,
                      "category": tcat.get(ts) or (t.category if t and t.category else "(ngoài catalog)"),
                      "viet_level": span_meta(r, si).get("viet_level", "chưa gán"),
                      "hyp": " ".join(sh[si]), "exact": e == 0 or rj == hj,
                      "cer": char_edit_distance(rj, hj) / len(rj)}
                if freq is not None:
                    sd["train_freq_real"], sd["train_freq_synth"] = freq.get(ts, (0, 0))
                rec["spans"].append(sd)
        # thuật ngữ
        ref_terms = Counter(spans)
        matched = sum(min(c, count_occ(ht, t)) for t, c in ref_terms.items())
        hyp_terms = Counter(cat.find_terms(ht))
        for t in ref_terms:  # span ngoài catalog vẫn tính
            if t not in cat.terms:
                hyp_terms[t] += count_occ(ht, t)
        hyp_hits = sum(hyp_terms.values())
        rec["ref_terms" + sfx] = sum(ref_terms.values())
        rec["matched_terms" + sfx] = matched
        rec["hyp_terms" + sfx] = hyp_hits
        rec["false_terms" + sfx] = sum(max(0, c - ref_terms.get(t, 0)) for t, c in hyp_terms.items())
    return rec


def term_weights(recs):
    """Trọng số cố định theo thuật ngữ cho bootstrap: w = 1 / (số span của thuật ngữ trong TOÀN BỘ tập `recs`).
    Trên tập đầy đủ, Σ w·exact / Σ w = term_acc_macro (Σ w = số thuật ngữ)."""
    n = Counter(x["term"] for r in recs for x in r.get("spans", []) if "exact" in x)
    return {t: 1.0 / c for t, c in n.items()}


def span_stats(spans, term_w=None):
    """Chỉ số theo span trên một danh sách span (thang %). Span thiếu `exact` (utt_scores cũ) bị bỏ qua.
    term_w (từ term_weights của tập đầy đủ): tính term_acc_macro trên MẪU bootstrap bằng trọng số cố định thay vì
    trung bình đều qua các thuật ngữ có mặt trong mẫu (cách đó lệch lên vì thuật ngữ hiếm rơi khỏi mẫu)."""
    sp = [x for x in spans if "exact" in x]
    if not sp:
        return {"n_span": 0, "n_term": 0, "span_acc": None, "term_acc_macro": None, "span_cer": None, "cs_wer": None}
    per = defaultdict(lambda: [0, 0])  # thuật ngữ -> [số span exact, số span]
    for x in sp:
        per[x["term"]][0] += bool(x["exact"])
        per[x["term"]][1] += 1
    n_tok = sum(x["n"] for x in sp)
    if term_w:
        den = sum(term_w.get(x["term"], 0.0) for x in sp)
        macro = 100.0 * sum(term_w.get(x["term"], 0.0) * bool(x["exact"]) for x in sp) / den if den else None
    else:
        macro = 100.0 * sum(a / b for a, b in per.values()) / len(per)
    return {"n_span": len(sp), "n_term": len(per),
            "span_acc": 100.0 * sum(bool(x["exact"]) for x in sp) / len(sp),
            "term_acc_macro": macro,
            "span_cer": 100.0 * sum(min(x["cer"], 1.0) for x in sp) / len(sp),
            "cs_wer": 100.0 * sum(x["err"] for x in sp) / n_tok if n_tok else None}


FREQ_SLICES = ["unseen", "synth_only", "real_1-4", "real_5+"]


def freq_slice(x):
    """Lát tần suất train của một span (None nếu span không có train_freq_*)."""
    if "train_freq_real" not in x:
        return None
    real, synth = x["train_freq_real"], x.get("train_freq_synth", 0)
    if real == 0:
        return "synth_only" if synth > 0 else "unseen"
    return "real_1-4" if real < 5 else "real_5+"


def freq_slices(recs):
    """{lát: span_stats} theo tần suất thuật ngữ trong train; {} nếu bench chưa gắn train_freq."""
    g = defaultdict(list)
    for r in recs:
        for x in r.get("spans", []):
            k = freq_slice(x)
            if k:
                g[k].append(x)
    return {k: span_stats(g[k]) for k in FREQ_SLICES} if g else {}


SPAN_METRICS = ("span_acc", "term_acc_macro", "span_cer")


def agg(recs, with_spans=True, term_w=None):
    """Gộp chỉ số trên danh sách utt. with_spans=False bỏ qua chỉ số theo span (nhanh hơn trong bootstrap khi
    không cần chúng; khi đó các khoá span là None). term_w: xem span_stats (chỉ dùng trong bootstrap)."""
    s = Counter()
    for r in recs:
        for k in ("n", "err", "n_en", "err_en", "err_en_len", "n_vi", "err_vi", "ref_terms", "matched_terms",
                  "hyp_terms", "false_terms", "matched_terms_len", "S", "D", "I"):
            s[k] += r.get(k, 0)
    s["utt"] = len(recs)
    ss = span_stats([x for r in recs for x in r.get("spans", [])] if with_spans else [], term_w)

    def rate(a, b):
        return 100.0 * s[a] / s[b] if s[b] else None
    return {"utt": len(recs), "n": s["n"], "n_en": s["n_en"],
            "wer": rate("err", "n"), "cs_wer": rate("err_en", "n_en"), "cs_wer_lenient": rate("err_en_len", "n_en"),
            "n_wer": rate("err_vi", "n_vi"), "term_recall": rate("matched_terms", "ref_terms"),
            "term_recall_lenient": rate("matched_terms_len", "ref_terms"),
            "term_precision": rate("matched_terms", "hyp_terms"),
            "false_terms_per_100utt": 100.0 * s["false_terms"] / len(recs) if recs else None,
            "S": s["S"], "D": s["D"], "I": s["I"],
            "n_span": ss["n_span"], "n_term": ss["n_term"], "span_acc": ss["span_acc"],
            "term_acc_macro": ss["term_acc_macro"], "span_cer": ss["span_cer"]}


def bootstrap_many(recs, metrics, iters=1000, seed=0):
    """CI 95% nhiều metric một lượt bằng cách lấy mẫu lại NGƯỜI NÓI (boot_unit: speaker_cluster nếu có, không thì
    speaker; có hoàn lại). Mỗi mẫu tính lại agg từ đầu nên metric macro cũng được tính lại đúng trong mẫu."""
    by = defaultdict(list)
    for r in recs:
        by[r.get("boot_unit") or r["speaker"]].append(r)
    spk = list(by)
    if len(spk) < 2:
        return {m: None for m in metrics}
    rnd = random.Random(seed)
    vals = {m: [] for m in metrics}
    ws = any(m in SPAN_METRICS for m in metrics)
    tw = term_weights(recs) if "term_acc_macro" in metrics else None  # cố định theo tập đầy đủ, không tính lại theo mẫu
    for _ in range(iters):
        sample = [x for s in (rnd.choice(spk) for _ in spk) for x in by[s]]
        a = agg(sample, ws, tw)
        for m in metrics:
            if a[m] is not None:
                vals[m].append(a[m])
    out = {}
    for m, v in vals.items():
        v.sort()
        out[m] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else None
    return out


def bootstrap(recs, metric, iters=1000, seed=0):
    return bootstrap_many(recs, [metric], iters, seed)[metric]


GROUPS = {"main_B1+B2": MAIN, "B1": ["B1"], "B2": ["B2"], "B3": ["B3"], "B4": ["B4"], "B5": ["B5"], "cs_B1-B4": CS_SPLITS}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--out")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--label-field", default="text_tagged",
                    help="trường chứa reference (mặc định text_tagged = nhãn theo Spec). Vd text_tagged_orig: chấm "
                         "theo nhãn gốc của ViMedCSS (số phụ cho B3); câu không có trường này dùng text_tagged")
    ap.add_argument("--train-manifest", nargs="+", metavar="PATH",
                    help="manifest train (JSONL): đếm tần suất thuật ngữ trong train (thật / synthetic) cho lát "
                         "unseen / synth_only / real_1-4 / real_5+")
    args = ap.parse_args()
    out = Path(args.out or Path(args.hyps).parent)
    out.mkdir(parents=True, exist_ok=True)

    cat = Catalog.load(args.catalog)
    bench = read_jsonl(args.bench)
    hyps = {h["id"]: h.get("hyp", "") for h in read_jsonl(args.hyps)}
    missing = [r["id"] for r in bench if r["id"] not in hyps]
    if missing:
        print(f"CẢNH BÁO: {len(missing)} utterance không có hyp (tính như hyp rỗng), vd {missing[:3]}")

    if args.label_field != "text_tagged":
        has = sum(1 for r in bench if r.get(args.label_field))
        print(f"Reference = {args.label_field}: {has} câu có trường này, {len(bench) - has} câu dùng text_tagged")
    freq = None
    if args.train_manifest:
        terms = {t for r in bench for t in normalize_ref(r.get(args.label_field) or r["text_tagged"], cat)[1]}
        freq = train_term_freq(args.train_manifest, cat, terms)
        print(f"Tần suất train: {len(terms)} thuật ngữ bench, {sum(1 for v in freq.values() if v == (0, 0))} chưa thấy")
    recs = [score_utt(r, hyps.get(r["id"], ""), cat, args.label_field, freq) for r in bench]
    with open(out / "utt_scores.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lenient_ok = cat.lenient_map != cat.strict_map  # catalog không có cột phonetic nào thì lenient = strict
    summary = {"system": out.name, "hyps": str(args.hyps), "bench": str(args.bench), "label_field": args.label_field,
               "lenient_available": lenient_ok, "train_manifest": args.train_manifest, "groups": {}}
    for g, splits in GROUPS.items():
        R = [r for r in recs if r["split"] in splits]
        if not R:
            continue
        a = agg(R)
        ms = [m for m in ("wer", "cs_wer", "n_wer", "span_acc", "term_acc_macro", "span_cer") if a[m] is not None]
        for m, ci in bootstrap_many(R, ms, args.iters).items():
            a[m + "_ci95"] = ci
        a["boot_units"] = len({r.get("boot_unit") or r["speaker"] for r in R})
        if freq is not None:
            a["freq_slices"] = freq_slices(R)
        summary["groups"][g] = a
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- cắt lát
    lines = [f"# Cắt lát — {out.name}", ""]
    cs = [r for r in recs if r["split"] in CS_SPLITS]

    def pct(x):
        return "–" if x is None else f"{x:.2f}"
    sg = {g: a for g, a in summary["groups"].items() if a.get("span_acc") is not None}
    if sg:
        lines += ["## Chỉ số theo span (strict)", "",
                  "| Nhóm | Span | Thuật ngữ | span_acc | term_acc_macro | span_cer | CS-WER |", "| --- | --- | --- | --- | --- | --- | --- |"]
        lines += [f"| {g} | {a['n_span']} | {a['n_term']} | {pct(a['span_acc'])} | {pct(a['term_acc_macro'])} | "
                  f"{pct(a['span_cer'])} | {pct(a['cs_wer'])} |" for g, a in sg.items()]
        lines.append("")
    for g, a in sg.items():
        if a.get("freq_slices"):
            lines += [f"## Lát cắt theo tần suất thuật ngữ trong train ({g})", "",
                      "| Lát | Span | Thuật ngữ | span_acc | term_acc_macro | span_cer | CS-WER |", "| --- | --- | --- | --- | --- | --- | --- |"]
            lines += [f"| {k} | {v['n_span']} | {v['n_term']} | {pct(v['span_acc'])} | {pct(v['term_acc_macro'])} | "
                      f"{pct(v['span_cer'])} | {pct(v['cs_wer'])} |" for k, v in a["freq_slices"].items()]
            lines.append("")
    for key, title in (("region", "Miền"), ("channel", "Kênh"), ("gender", "Giới tính"), ("domain", "Domain")):
        lines += [f"## {title} (B1–B4)", "", "| Giá trị | Utt | Token EN | CS-WER | N-WER |", "| --- | --- | --- | --- | --- |"]
        g = defaultdict(list)
        for r in cs:
            g[r[key]].append(r)
        for k, R in sorted(g.items()):
            a = agg(R)
            lines.append(f"| {k} | {a['utt']} | {a['n_en']} | {a['cs_wer']:.2f} | {a['n_wer']:.2f} |" if a["cs_wer"] is not None else f"| {k} | {a['utt']} | 0 | – | – |")
        lines.append("")
    b2 = [r for r in recs if r["split"] == "B2"]
    for key, title, R, where in (("category", "Category thuật ngữ", cs, "B1–B4"), ("category", "Category thuật ngữ", b2, "B2"),
                                 ("viet_level", "Mức Việt hoá", cs, "B1–B4")):
        if where == "B2" and not R:
            continue
        lines += [f"## {title} ({where}, theo span)", "", "| Giá trị | Span | CS-WER |", "| --- | --- | --- |"]
        g = Counter()
        n = Counter()
        cnt = Counter()
        for r in R:
            for s in r["spans"]:
                g[s[key]] += s["err"]
                n[s[key]] += s["n"]
                cnt[s[key]] += 1
        for k in sorted(n):
            lines.append(f"| {k} | {cnt[k]} | {100 * g[k] / n[k]:.2f} |")
        lines.append("")
    terr = Counter()
    tn = Counter()
    for r in cs:
        for s in r["spans"]:
            terr[s["term"]] += s["err"]
            tn[s["term"]] += s["n"]
    worst = sorted(tn, key=lambda t: (-terr[t] / tn[t], -tn[t]))[:25]
    lines += ["## 25 thuật ngữ sai nhiều nhất (B1–B4)", "", "| Thuật ngữ | Token | CS-WER |", "| --- | --- | --- |"]
    lines += [f"| {t} | {tn[t]} | {100 * terr[t] / tn[t]:.1f} |" for t in worst]
    (out / "slices.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    m = summary["groups"].get("main_B1+B2", {})
    b5 = summary["groups"].get("B5", {})

    def f(x):
        return "–" if x is None else f"{x:.2f}"
    c = m.get("cs_wer_ci95")
    cs_ci = f"[{c[0]:.2f}, {c[1]:.2f}]" if c else ""
    len_txt = f"lenient {f(m.get('cs_wer_lenient'))} · " if lenient_ok else ""
    print(f"[{out.name}] CS-WER B1+B2 {f(m.get('cs_wer'))} {cs_ci} · {len_txt}"
          f"span acc {f(m.get('span_acc'))} · term acc macro {f(m.get('term_acc_macro'))} · span CER {f(m.get('span_cer'))} · "
          f"N-WER B1+B2 {f(m.get('n_wer'))} · term R/P {f(m.get('term_recall'))}/{f(m.get('term_precision'))} · "
          f"B3 CS-WER {f(summary['groups'].get('B3', {}).get('cs_wer'))} · B5 WER {f(b5.get('wer'))} · "
          f"B5 chèn sai {f(b5.get('false_terms_per_100utt'))}/100 utt")


if __name__ == "__main__":
    main()
