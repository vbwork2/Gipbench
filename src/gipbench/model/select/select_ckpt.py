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
CS-WER vẫn in để đối chiếu; chỉ dùng để chọn khi --rule cs_wer (CS-WER dev CS thấp nhất, hoà: WER dev VI thấp hơn).
Không ứng viên nào đạt luật thì chọn model gốc và ghi rõ.

Luật chọn phải đăng ký TRƯỚC khi chạy (experiment YAML: select.rule) và được ghi lại (audit R21: luật đã đổi sau khi chạy).
Khi truyền --dev-ref-cs / --dev-ref-vi (model.run luôn truyền), <out>.json ghi thêm rule, dev_refs, dev_flags, catalog,
catalog_flags; dev không có flags.independent: true (không chứng minh độc lập với train / bench) bị TỪ CHỐI (thoát mã 3), trừ
--allow-legacy-dev (chạy lại dev cũ, chỉ cảnh báo). Catalog có span bench (from_bench) bị từ chối trừ --allow-bench-catalog.
"""
import argparse
import json
import sys
from pathlib import Path

from gipbench.core import release as R
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
    ap.add_argument("--rule", choices=["term_acc_macro", "cs_wer"], default="term_acc_macro",
                    help="tiêu chí chọn trong các ứng viên đạt ràng buộc WER dev VI (đăng ký trước, ghi vào <out>.json)")
    ap.add_argument("--dev-ref-cs", help="release của dev CS (vd dev/dev_v2_cs@1): kiểm flags.independent")
    ap.add_argument("--dev-ref-vi", help="release của dev VI")
    ap.add_argument("--allow-legacy-dev", action="store_true", help="dev không có flags.independent chỉ bị cảnh báo")
    ap.add_argument("--allow-bench-catalog", action="store_true", help="cho phép catalog from_bench (chẩn đoán oracle)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    try:
        cat_flags = R.forbid_bench_catalog(args.catalog, "select_ckpt", allow=args.allow_bench_catalog)
    except R.ReleaseError as e:
        print("LỖI", e)
        sys.exit(3)
    dev_refs = {k: v for k, v in (("cs", args.dev_ref_cs), ("vi", args.dev_ref_vi)) if v}
    dev_flags = {}
    for k, ref in dev_refs.items():
        try:
            dev_flags[k] = dict(R.resolve(ref, verify=False).meta.get("flags") or {})
        except R.ReleaseError as e:
            print("LỖI", e)
            sys.exit(3)
        if not dev_flags[k].get("independent"):
            msg = (f"dev {k} ({ref}) không có flags.independent: true: chưa chứng minh độc lập theo video / người nói với "
                   "train và bench (audit R01: dev cũ chọn checkpoint ngược chiều bench)")
            if not args.allow_legacy_dev:
                print("LỖI", msg, "(--allow-legacy-dev chỉ để chạy lại dev cũ)")
                sys.exit(3)
            print("CẢNH BÁO", msg)
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
    if not elig:
        pick = Path(args.base).name
    elif args.rule == "cs_wer":
        pick = min(elig, key=lambda x: (x[1]["cs_wer"] if x[1]["cs_wer"] is not None else float("inf"), x[1]["wer_vi"]))[0]
    else:
        pick = max(elig, key=lambda x: (x[1]["term_acc_macro"] or 0.0, -x[1]["wer_vi"]))[0]

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
    crit = ("term_acc_macro dev CS cao nhất" if args.rule == "term_acc_macro" else "CS-WER dev CS thấp nhất")
    L += ["", f"Luật: WER dev VI tăng <= {args.max_vi_delta} điểm so với {Path(args.base).name}; trong các ứng viên đạt, "
          f"chọn {crit} (hoà: WER dev VI thấp hơn). Được chọn: **{pick}**"
          + ("" if elig else " (KHÔNG ứng viên nào đạt luật -> giữ model gốc)")]
    out = {"pick": pick, "results": res}
    if dev_refs or args.rule != "term_acc_macro":  # có tham chiếu dev (model.run) hoặc luật khác mặc định: ghi nguồn gốc luật
        out.update(rule={"name": args.rule, "max_vi_delta": args.max_vi_delta}, dev_refs=dev_refs, dev_flags=dev_flags,
                   catalog=str(args.catalog), catalog_flags=cat_flags)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out + ".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    Path(args.out + ".json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
