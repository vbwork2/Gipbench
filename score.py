#!/usr/bin/env python3
"""Chấm điểm một hệ thống trên bench: WER, CS-WER (strict + lenient), N-WER, term recall/precision,
chèn sai thuật ngữ trên B5, khoảng tin cậy bootstrap theo người nói, và cắt lát theo metadata.

  python score.py --bench data/bench.jsonl --catalog configs/catalog.tsv \
      --hyps results/e0_baseline/hyps.jsonl [--out results/e0_baseline]

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.align import align, attribute  # noqa: E402
from lib.manifest import CS_SPLITS, read_jsonl, span_meta  # noqa: E402
from lib.textnorm import Catalog, normalize_hyp, normalize_ref  # noqa: E402

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


def score_utt(r, hyp_text, cat, label_field="text_tagged"):
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
            for si, term in enumerate(spans):
                t = cat.terms.get(term)
                ts = " ".join(term)
                rec["spans"].append({"term": ts, "n": sum(1 for s in sp if s == si), "err": errs.get(si, 0),
                                     "category": tcat.get(ts) or (t.category if t and t.category else "(ngoài catalog)"),
                                     "viet_level": span_meta(r, si).get("viet_level", "chưa gán")})
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


def agg(recs):
    s = Counter()
    for r in recs:
        for k in ("n", "err", "n_en", "err_en", "err_en_len", "n_vi", "err_vi", "ref_terms", "matched_terms",
                  "hyp_terms", "false_terms", "matched_terms_len", "S", "D", "I"):
            s[k] += r.get(k, 0)
    s["utt"] = len(recs)

    def rate(a, b):
        return 100.0 * s[a] / s[b] if s[b] else None
    return {"utt": len(recs), "n": s["n"], "n_en": s["n_en"],
            "wer": rate("err", "n"), "cs_wer": rate("err_en", "n_en"), "cs_wer_lenient": rate("err_en_len", "n_en"),
            "n_wer": rate("err_vi", "n_vi"), "term_recall": rate("matched_terms", "ref_terms"),
            "term_recall_lenient": rate("matched_terms_len", "ref_terms"),
            "term_precision": rate("matched_terms", "hyp_terms"),
            "false_terms_per_100utt": 100.0 * s["false_terms"] / len(recs) if recs else None,
            "S": s["S"], "D": s["D"], "I": s["I"]}


def bootstrap(recs, metric, iters=1000, seed=0):
    """CI 95% bằng cách lấy mẫu lại NGƯỜI NÓI (boot_unit: speaker_cluster nếu có, không thì speaker; có hoàn lại)."""
    by = defaultdict(list)
    for r in recs:
        by[r.get("boot_unit") or r["speaker"]].append(r)
    spk = list(by)
    if len(spk) < 2:
        return None
    rnd = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [x for s in (rnd.choice(spk) for _ in spk) for x in by[s]]
        v = agg(sample)[metric]
        if v is not None:
            vals.append(v)
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]] if vals else None


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
    recs = [score_utt(r, hyps.get(r["id"], ""), cat, args.label_field) for r in bench]
    with open(out / "utt_scores.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {"system": out.name, "hyps": str(args.hyps), "bench": str(args.bench), "label_field": args.label_field,
               "groups": {}}
    for g, splits in GROUPS.items():
        R = [r for r in recs if r["split"] in splits]
        if not R:
            continue
        a = agg(R)
        for m in ("wer", "cs_wer", "n_wer"):
            if a[m] is not None:
                a[m + "_ci95"] = bootstrap(R, m, args.iters)
        a["boot_units"] = len({r.get("boot_unit") or r["speaker"] for r in R})
        summary["groups"][g] = a
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- cắt lát
    lines = [f"# Cắt lát — {out.name}", ""]
    cs = [r for r in recs if r["split"] in CS_SPLITS]
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
    print(f"[{out.name}] CS-WER B1+B2 {f(m.get('cs_wer'))} {cs_ci} · lenient {f(m.get('cs_wer_lenient'))} · "
          f"N-WER B1+B2 {f(m.get('n_wer'))} · term R/P {f(m.get('term_recall'))}/{f(m.get('term_precision'))} · "
          f"B3 CS-WER {f(summary['groups'].get('B3', {}).get('cs_wer'))} · B5 WER {f(b5.get('wer'))} · "
          f"B5 chèn sai {f(b5.get('false_terms_per_100utt'))}/100 utt")


if __name__ == "__main__":
    main()
