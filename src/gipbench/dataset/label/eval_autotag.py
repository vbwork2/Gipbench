#!/usr/bin/env python3
"""Đo chất lượng bộ tự gắn tag (converters/common.auto_tag) so với một file vàng: precision / recall mức span.

  python -m gipbench.dataset.label.eval_autotag --gold reports/history/t02_gold.jsonl --catalog configs/dataset/labeling/catalog.example.tsv \
      --texts data/dl/ggroup/*.parquet --out reports/t02_eval [--spotcheck 30 --seed 0]

File vàng (JSONL, mỗi dòng {"id", "gold"}): text gốc kèm đánh dấu, bỏ đánh dấu phải ra đúng text gốc.
  [[...]]  span tiếng Anh / tiếng nước ngoài (từ, viết tắt Latin kiểu Anh, brand, tên riêng nước ngoài)
  {{...}}  viết tắt / mã tiếng Việt bằng chữ Latin (vn, gdt, bidv, mã chứng khoán): không phải tiếng Anh
  ((...))  từ Anh bị viết theo phiên âm Việt (chai đinh = trading): chỉ thống kê, auto_tag không thể thấy
Span được so theo vị trí token (tách khoảng trắng của text gốc); "khớp" = trùng hệt đầu và cuối.
Mặc định gộp các span kề nhau (cả vàng lẫn dự đoán) thành một dải tiếng Anh liền: auto_tag cố ý tách
`<en>bitcoin</en> <en>sideway</en>` theo thuật ngữ catalog, còn việc cần đo là có tìm ra vùng tiếng Anh không.
--exact-spans để so từng span như ghi.
Hai cách tính precision: strict (mọi span dự đoán không khớp đều là sai) và neutral (span dự đoán nằm gọn
trong một span {{...}} không bị tính, vì đó là chỗ người duyệt sẽ bỏ tag).
Ghi: <out>.md, <out>.json, và (nếu --spotcheck) <out>_spotcheck.csv để người duyệt kiểm lại file vàng.
"""
import argparse
import csv
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from gipbench.core.tagging import VI_WORD_FILES, auto_tag, load_vi_words
from gipbench.core.textnorm import Catalog

_MARK = re.compile(r"\[\[(.+?)\]\]|\{\{(.+?)\}\}|\(\((.+?)\)\)")
KINDS = {1: "en", 2: "vi_abbr", 3: "phonetic"}


def parse_gold(gold: str):
    """-> (text gốc, [(start, end, kind)]) với chỉ số token."""
    toks, spans, pos = [], [], 0
    for m in _MARK.finditer(gold):
        toks += gold[pos:m.start()].split()
        kind = next(k for k in (1, 2, 3) if m.group(k) is not None)
        inner = m.group(kind).split()
        spans.append((len(toks), len(toks) + len(inner), KINDS[kind]))
        toks += inner
        pos = m.end()
    toks += gold[pos:].split()
    return " ".join(toks), spans


def parse_pred(text_tagged: str):
    """Span <en>...</en> của auto_tag -> [(start, end)] theo token (text gốc không có dấu câu dính tag)."""
    spans, start = [], None
    for i, t in enumerate(text_tagged.split()):
        if t.startswith("<en>") or "<en>" in t:
            start = i
        if "</en>" in t:
            spans.append((start, i + 1))
            start = None
    return spans


def merge_adjacent(spans):
    out = []
    for s, e in sorted(spans):
        if out and out[-1][1] == s:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return set(out)


def evaluate(rows, cat, merge=True, vi_words=None):
    tp = fp = fp_neutral = fn = 0
    tok_tp = tok_fp = tok_fn = 0
    fns, fps, per = [], [], []
    for r in rows:
        toks = r["text"].split()
        gold_en = {(s, e) for s, e, k in r["spans"] if k == "en"}
        abbr = [(s, e) for s, e, k in r["spans"] if k == "vi_abbr"]
        pred = set(parse_pred(auto_tag(r["text"], cat, vi_words=vi_words)["text_tagged"]))
        if merge:
            gold_en, pred = merge_adjacent(gold_en), merge_adjacent(pred)
        hit = gold_en & pred
        tp += len(hit)
        fn += len(gold_en - pred)
        for s, e in sorted(pred - gold_en):
            inside_abbr = any(a <= s and e <= b for a, b in abbr)
            fp += 1
            fp_neutral += not inside_abbr
            fps.append((r["id"], " ".join(toks[s:e]), "vi_abbr" if inside_abbr else
                        ("chồng lấn" if any(not (e <= a or b <= s) for a, b in gold_en) else "thừa")))
        for s, e in sorted(gold_en - pred):
            fns.append((r["id"], " ".join(toks[s:e]), "chồng lấn" if any(not (e <= a or b <= s) for a, b in pred) else "bỏ sót"))
        g_tok = {i for s, e in gold_en for i in range(s, e)}
        p_tok = {i for s, e in pred for i in range(s, e)}
        tok_tp += len(g_tok & p_tok)
        tok_fp += len(p_tok - g_tok)
        tok_fn += len(g_tok - p_tok)
        per.append({"id": r["id"], "gold": sorted(gold_en), "pred": sorted(pred)})

    def div(a, b):
        return a / b if b else None
    p, rc = div(tp, tp + fp), div(tp, tp + fn)
    pn = div(tp, tp + fp_neutral)
    return {"utt": len(rows), "gold_spans": tp + fn, "pred_spans": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "precision_neutral": pn, "recall": rc,
            "f1": div(2 * p * rc, p + rc) if p and rc else None,
            "token_precision": div(tok_tp, tok_tp + tok_fp), "token_recall": div(tok_tp, tok_tp + tok_fn),
            "fn_list": fns, "fp_list": fps, "per_utt": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--texts", nargs="*", default=[], help="parquet gốc (cột id, text) để kiểm tra file vàng khớp text gốc")
    ap.add_argument("--id-prefix", default="gg_{name}_", help="id vàng = prefix + id gốc; {name} = tên file parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exact-spans", action="store_true", help="không gộp span kề nhau khi so")
    ap.add_argument("--spotcheck", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vi-words", nargs="*", default=None,
                    help="danh sách từ không gắn <en> (mặc định configs/dataset/labeling/vi_abbrev.txt + configs/dataset/labeling/vi_loanwords.txt; "
                         "truyền rỗng để tắt)")
    args = ap.parse_args()
    vi_files = VI_WORD_FILES if args.vi_words is None else [Path(p) for p in args.vi_words]
    vi_used = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()[:12] for p in vi_files if p.exists()}
    vi_words = load_vi_words(vi_files)

    rows = []
    for line in open(args.gold, encoding="utf-8"):
        if line.strip():
            g = json.loads(line)
            text, spans = parse_gold(g["gold"])
            rows.append({"id": g["id"], "gold": g["gold"], "text": text, "spans": spans})
    if args.texts:
        import pyarrow.parquet as pq
        orig = {}
        for f in sorted(x for p in args.texts for x in glob.glob(p)):
            d = pq.read_table(f, columns=["id", "text"]).to_pydict()
            pre = args.id_prefix.replace("{name}", Path(f).stem)
            orig.update({pre + str(i): t for i, t in zip(d["id"], d["text"])})
        bad = [r["id"] for r in rows if orig.get(r["id"]) != r["text"]]
        if bad:
            sys.exit(f"LỖI: {len(bad)} câu vàng không khớp text gốc sau khi bỏ đánh dấu, vd {bad[:5]}")
        print(f"File vàng khớp text gốc: {len(rows)} câu")

    cat = Catalog.load(args.catalog)
    with_en = [r for r in rows if any(k == "en" for _, _, k in r["spans"])]
    merge = not args.exact_spans
    res = {"merge_adjacent": merge, "vi_words_files": vi_used, "vi_words_count": len(vi_words),
           "subset_en": evaluate(with_en, cat, merge, vi_words), "all_read": evaluate(rows, cat, merge, vi_words),
           "subset_en_exact": {k: v for k, v in evaluate(with_en, cat, False, vi_words).items()
                               if not k.endswith("_list") and k != "per_utt"},
           "phonetic_spans": sum(k == "phonetic" for r in rows for _, _, k in r["spans"]),
           "phonetic_utts": sum(any(k == "phonetic" for _, _, k in r["spans"]) for r in rows),
           "vi_abbr_spans": sum(k == "vi_abbr" for r in rows for _, _, k in r["spans"])}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Path(str(out) + ".json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    def f(x):
        return "–" if x is None else f"{x:.3f}"
    vi_desc = ", ".join(f"`{p}` (sha256 {h})" for p, h in vi_used.items()) or "không dùng"
    md = [f"# Đánh giá auto_tag — {args.gold}", "",
          f"Danh sách từ không gắn <en>: {vi_desc}; {len(vi_words)} từ.", "",
          f"Catalog: `{args.catalog}`. {len(rows)} câu đã đọc, {len(with_en)} câu có span tiếng Anh. "
          f"Ngoài ra: {res['vi_abbr_spans']} span viết tắt tiếng Việt, {res['phonetic_spans']} span phiên âm Việt "
          f"trong {res['phonetic_utts']} câu (không tính vào recall). "
          f"So span: {'gộp span kề nhau thành dải liền' if merge else 'từng span như ghi'}; "
          f"không gộp thì P/R trên câu có từ Anh = {f(res['subset_en_exact']['precision'])}/{f(res['subset_en_exact']['recall'])}.", "",
          "| Tập | Span vàng | Span dự đoán | Precision | Precision (bỏ qua viết tắt VN) | Recall | F1 | Token P | Token R |",
          "|" + " --- |" * 9]
    for name, k in (("Câu có từ Anh", "subset_en"), ("Mọi câu đã đọc", "all_read")):
        e = res[k]
        md.append(f"| {name} ({e['utt']}) | {e['gold_spans']} | {e['pred_spans']} | {f(e['precision'])} | "
                  f"{f(e['precision_neutral'])} | {f(e['recall'])} | {f(e['f1'])} | {f(e['token_precision'])} | {f(e['token_recall'])} |")
    e = res["all_read"]
    md += ["", f"## Span vàng không khớp ({len(e['fn_list'])})", "", "| id | span vàng | loại |", "| --- | --- | --- |"]
    md += [f"| {i} | {s} | {k} |" for i, s, k in e["fn_list"]]
    md += ["", f"## Span dự đoán không khớp ({len(e['fp_list'])})", "", "| id | span dự đoán | loại |", "| --- | --- | --- |"]
    md += [f"| {i} | {s} | {k} |" for i, s, k in e["fp_list"]]
    Path(str(out) + ".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    s = res["subset_en"]
    print(f"[{len(with_en)} câu có từ Anh] span P {f(s['precision'])} (bỏ qua viết tắt VN {f(s['precision_neutral'])}) · "
          f"R {f(s['recall'])} · F1 {f(s['f1'])} · token P/R {f(s['token_precision'])}/{f(s['token_recall'])}")
    print(f"Đã ghi {out}.md, {out}.json")

    if args.spotcheck:
        pick = random.Random(args.seed).sample(with_en, min(args.spotcheck, len(with_en)))
        p = Path(str(out) + "_spotcheck.csv")
        with open(p, "w", newline="", encoding="utf-8-sig") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["id", "text", "gold (vàng do Claude gắn)", "dong_y (y/n)", "sua_lai", "ghi_chu"])
            for r in pick:
                w.writerow([r["id"], r["text"], r["gold"], "", "", ""])
        print(f"Đã ghi {p} ({len(pick)} câu để người duyệt)")


if __name__ == "__main__":
    main()
