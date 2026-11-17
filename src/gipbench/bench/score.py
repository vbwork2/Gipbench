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

from gipbench.core.scoring import (  # noqa: F401  (import lại: code cũ gọi gipbench.bench.score.<hàm>)
    FREQ_SLICES, GROUPS, MAIN, SPAN_METRICS, agg, bootstrap, bootstrap_many, count_occ, freq_slice, freq_slices, score_utt, span_stats, term_weights, train_term_freq)


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
