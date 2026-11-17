#!/usr/bin/env python3
"""Chọn checkpoint / alpha WiSE-FT trên DEV (không dùng bench): term_acc_macro dev CS cao nhất (accuracy theo thuật
ngữ rồi trung bình đều; hoà: WER dev VI thấp hơn), với điều kiện WER dev VI tăng <= --max-vi-delta (điểm tuyệt đối)
so với model gốc. Bảng phụ tách dev CS theo `dataset` (tiền tố trước dấu ':') để thấy khi một nguồn tốt lên còn nguồn
khác xấu đi.

  python -m gipbench.model.select.select_ckpt --catalog data/bench_v0_draft/catalog_v0_draft.tsv \
      --dev-cs data/e3_v1/dev_cs.jsonl --dev-vi data/e3_v1/dev_vi.jsonl \
      --base results/e3_v1_dev/base --cand results/e3_v1_dev/ck4000 results/e3_v1_dev/wise_0.5 ... \
      --out results/e3_v1_dev/select

Mỗi thư mục (base và ứng viên) cần dev_cs/hyps.jsonl và dev_vi/hyps.jsonl (decode.py --system dev_cs / dev_vi).
Reference: text_tagged nếu có, không thì text (dev VI). Ghi <out>.md (bảng) và <out>.json (ứng viên được chọn).
CS-WER vẫn in để đối chiếu nhưng không dùng để chọn.
Không ứng viên nào đạt luật thì chọn model gốc và ghi rõ.
"""
import argparse
import json
import sys
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import Catalog
from gipbench.core.scoring import agg, score_utt


def evaluate(rows, hyp_path, cat):
    """Trả về (agg toàn bộ, {dataset: agg}); dataset = tiền tố trước dấu ':' của trường `dataset` ("?" nếu thiếu)."""
    hyps = {h["id"]: h.get("hyp", "") for h in read_jsonl(str(hyp_path))}
    recs, by_ds = [], {}
    for r in rows:
        r = dict(r)
        r.setdefault("text_tagged", r.get("text", ""))
        rec = score_utt(r, hyps.get(r["id"], ""), cat)
        recs.append(rec)
        by_ds.setdefault(str(r.get("dataset") or "?").split(":")[0], []).append(rec)
    return agg(recs), {d: agg(R) for d, R in sorted(by_ds.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--dev-cs", required=True)
    ap.add_argument("--dev-vi", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--max-vi-delta", type=float, default=0.3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    cs_rows, vi_rows = read_jsonl(args.dev_cs), read_jsonl(args.dev_vi)
    res = {}
    for d in [args.base] + args.cand:
        p = Path(d)
        if not (p / "dev_cs/hyps.jsonl").exists() or not (p / "dev_vi/hyps.jsonl").exists():
            print(f"BỎ {p.name}: thiếu hyps dev")
            continue
        (cs, cs_ds), (vi, _) = evaluate(cs_rows, p / "dev_cs/hyps.jsonl", cat), evaluate(vi_rows, p / "dev_vi/hyps.jsonl", cat)
        res[p.name] = {"term_acc_macro": cs["term_acc_macro"], "span_acc": cs["span_acc"], "span_cer": cs["span_cer"],
                       "cs_wer": cs["cs_wer"], "wer_cs": cs["wer"], "n_wer_cs": cs["n_wer"], "term_recall": cs["term_recall"],
                       "wer_vi": vi["wer"],
                       "by_dataset": {d: {"n_span": a["n_span"], "term_acc_macro": a["term_acc_macro"],
                                          "span_acc": a["span_acc"], "cs_wer": a["cs_wer"]} for d, a in cs_ds.items()}}
    base = res[Path(args.base).name]
    rows = []
    for name, m in res.items():
        dv = m["wer_vi"] - base["wer_vi"]
        ok = name == Path(args.base).name or dv <= args.max_vi_delta
        rows.append((name, m, dv, ok))
    elig = [(n, m) for n, m, dv, ok in rows if ok and n != Path(args.base).name]
    pick = max(elig, key=lambda x: (x[1]["term_acc_macro"] or 0.0, -x[1]["wer_vi"]))[0] if elig else Path(args.base).name

    def f(x):
        return "–" if x is None else f"{x:.2f}"

    def d(x, y):
        return "–" if x is None or y is None else f"{x - y:+.2f}"
    L = ["| Ứng viên | Term acc macro dev CS | Δ | Span acc dev CS | CS-WER dev CS | WER dev CS | N-WER dev CS | Term recall dev CS "
         "| WER dev VI | Δ VI | Đạt luật |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, m, dv, ok in rows:
        L.append(f"| {'**' + n + '**' if n == pick else n} | {f(m['term_acc_macro'])} | {d(m['term_acc_macro'], base['term_acc_macro'])} | "
                 f"{f(m['span_acc'])} | {f(m['cs_wer'])} | {f(m['wer_cs'])} | {f(m['n_wer_cs'])} | {f(m['term_recall'])} | "
                 f"{m['wer_vi']:.2f} | {dv:+.2f} | {'có' if ok else 'KHÔNG'} |")
    dss = sorted({k for _, m, _, _ in rows for k in m["by_dataset"]})
    if dss:
        L += ["", "Dev CS theo dataset (term acc macro / span acc / số span):", "",
              "| Ứng viên | " + " | ".join(dss) + " |", "|" + " --- |" * (len(dss) + 1)]
        for n, m, _, _ in rows:
            cells = []
            for k in dss:
                a = m["by_dataset"].get(k)
                cells.append("–" if not a else f"{f(a['term_acc_macro'])} / {f(a['span_acc'])} / {a['n_span']}")
            L.append(f"| {'**' + n + '**' if n == pick else n} | " + " | ".join(cells) + " |")
    L += ["", f"Luật: WER dev VI tăng <= {args.max_vi_delta} điểm so với {Path(args.base).name}; trong các ứng viên đạt, "
          f"chọn term_acc_macro dev CS cao nhất (hoà: WER dev VI thấp hơn). Được chọn: **{pick}**"
          + ("" if elig else " (KHÔNG ứng viên nào đạt luật -> giữ model gốc)")]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps({"pick": pick, "results": res}, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
