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
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.align import align
from gipbench.core.decontam import read_lines
from gipbench.core.evalstats import check_hyps, paired_bootstrap, ratio, sum_counts
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import Catalog, normalize_hyp, normalize_ref
from gipbench.core.scoring import agg, count_occ, cs_breakdown, score_utt
from gipbench.model.recipe.experiment import VI_WER_RULE


def evaluate(rows, hyp_path, cat, strict=False, reserved=()):
    hyp_rows = read_jsonl(str(hyp_path))
    if strict:
        ids = check_hyps([r["id"] for r in rows], [h["id"] for h in hyp_rows])
        if not rows or not ids["ok"]:
            raise SystemExit(f"{hyp_path}: expected exactly one hypothesis per dev utterance")
    hyps = {h["id"]: h.get("hyp", "") for h in hyp_rows}
    recs, by_ds = [], {}
    for r in rows:
        r = dict(r)
        r.setdefault("text_tagged", r.get("text", ""))
        rec = score_utt(r, hyps.get(r["id"], ""), cat)
        if strict or reserved:
            ref, spans = normalize_ref(r["text_tagged"], cat)
            ref_tokens = [token.tok for token in ref]
            hyp_tokens = normalize_hyp(hyps[r["id"]], cat)
            if strict:
                rec.update(cs_breakdown(align(ref_tokens, hyp_tokens), [token.span for token in ref], ref_tokens, hyp_tokens))
            if reserved:
                terms = Counter(term for term in spans if term in reserved)
                rec["ft_unseen_terms"] = sum(terms.values())
                rec["ft_unseen_matched"] = sum(min(count, count_occ(hyp_tokens, term)) for term, count in terms.items())
        recs.append(rec)
        by_ds.setdefault(str(r.get("dataset") or "?").split(":")[0], []).append(rec)
    return agg(recs), {d: agg(group) for d, group in sorted(by_ds.items())}, recs


def cs_by_domain(records):
    domains = {}
    for rec in records:
        domains.setdefault(rec["domain"] or "unknown", []).append(rec)
    return {name: ratio(sum_counts(group), "cs_wer") for name, group in sorted(domains.items())}


def paired_cs_by_domain(base, candidate):
    groups = defaultdict(lambda: ([], []))
    for before, after in zip(base, candidate):
        pair = groups[after["domain"]]
        pair[0].append(before)
        pair[1].append(after)
    return {domain: paired_bootstrap(before, after, "cs_wer", "boot_unit", iters=1000, seed=0)
            for domain, (before, after) in sorted(groups.items())}


def retention_checks(metrics, base, vi_rows, base_vi_rows, limits):
    ci = paired_bootstrap(base_vi_rows, vi_rows, "wer", "boot_unit", iters=1000, seed=0)["ci95"]
    hall = metrics["en_hallucination_per100"]
    base_hall = base["en_hallucination_per100"]
    domains = {}
    for name, baseline in base["cs_wer_by_domain"].items():
        value = metrics["cs_wer_by_domain"][name]
        domains[name] = value - baseline if value is not None and baseline is not None else None
    deltas = {"wer_vi_ci95": ci, "en_hallucination_delta": hall - base_hall
              if hall is not None and base_hall is not None else None, "cs_domain_deltas": domains}
    limit = limits["max_vi_ci_upper"]
    if limit is not None and (ci is None or ci[1] > limit + 1e-9):
        return deltas, False
    limit = limits["max_vi_hallucination_delta"]
    if limit is not None and (deltas["en_hallucination_delta"] is None or deltas["en_hallucination_delta"] > limit + 1e-9):
        return deltas, False
    limit = limits["max_cs_domain_delta"]
    return deltas, limit is None or all(d is not None and d <= limit + 1e-9 for d in domains.values())


def write_selection(path, lines, result):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    Path(path + ".json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines))


def select_vi_wer(args, cat, dev_ref, flags, cat_flags):
    rows = read_jsonl(args.dev_vi)
    base = Path(args.base)
    baseline, _, baseline_records = evaluate(rows, base / "dev_vi/hyps.jsonl", cat, strict=True)
    if baseline["wer"] is None:
        raise SystemExit(f"{args.dev_vi}: no reference tokens to score")
    ci_upper = args.max_vi_ci_upper if args.max_vi_ci_upper is not None else 0.0
    results = {base.name: {"wer_vi": baseline["wer"], "wer_vi_delta": 0.0, "eligible": True}}
    eligible = []
    table = ["| Model | WER dev VI | ΔWER | CI 95% của ΔWER | Đạt |", "| --- | ---: | ---: | --- | --- |",
             f"| {base.name} | {baseline['wer']:.2f} | 0.00 | – | có |"]
    for path in map(Path, args.cand):
        metrics, _, records = evaluate(rows, path / "dev_vi/hyps.jsonl", cat, strict=True)
        paired = paired_bootstrap(baseline_records, records, "wer", "boot_unit", iters=1000, seed=0)
        delta, ci = paired["delta"], paired["ci95"]
        ok = delta < 0 and delta <= args.max_vi_delta and ci is not None and ci[1] <= ci_upper
        results[path.name] = {"wer_vi": metrics["wer"], "wer_vi_delta": delta, "wer_vi_ci95": ci, "eligible": ok}
        if ok:
            step = int(path.name[2:]) if path.name.startswith("ck") and path.name[2:].isdigit() else float("inf")
            eligible.append((metrics["wer"], step, path.name))
        interval = f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "–"
        table.append(f"| {path.name} | {metrics['wer']:.2f} | {delta:+.2f} | {interval} | {'có' if ok else 'không'} |")
    pick = min(eligible)[2] if eligible else base.name
    table += ["", f"Được chọn: **{pick}**."]
    write_selection(args.out, table, {"pick": pick, "results": results, "dev_refs": {"vi": dev_ref},
                    "dev_flags": {"vi": flags}, "catalog": args.catalog, "catalog_flags": cat_flags,
                    "rule": {"name": VI_WER_RULE, "require_improvement": True, "max_vi_delta": args.max_vi_delta,
                             "max_vi_ci_upper": ci_upper, "bootstrap_iters": 1000, "bootstrap_seed": 0}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--dev-cs")
    ap.add_argument("--dev-vi", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--cand", nargs="+", required=True)
    ap.add_argument("--max-vi-delta", type=float, default=0.3)
    ap.add_argument("--max-vi-ci-upper", type=float)
    ap.add_argument("--max-vi-hallucination-delta", type=float)
    ap.add_argument("--max-cs-domain-delta", type=float)
    ap.add_argument("--reserved-terms", help="Frozen list for reporting FT-unseen term recall on CS development data")
    ap.add_argument("--rule", choices=["term_acc_macro", "cs_wer", VI_WER_RULE], default="term_acc_macro",
                    help="tiêu chí chọn trong các ứng viên đạt ràng buộc WER dev VI (đăng ký trước, ghi vào <out>.json)")
    ap.add_argument("--dev-ref-cs", help="release của dev CS (vd dev/dev_v2_cs@1): kiểm flags.independent")
    ap.add_argument("--dev-ref-vi", help="release của dev VI")
    ap.add_argument("--allow-legacy-dev", action="store_true", help="dev không có flags.independent chỉ bị cảnh báo")
    ap.add_argument("--allow-bench-catalog", action="store_true", help="cho phép catalog from_bench (chẩn đoán oracle)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.rule == VI_WER_RULE:
        if args.dev_cs or args.dev_ref_cs or not args.dev_ref_vi or args.allow_legacy_dev:
            ap.error(f"{VI_WER_RULE} requires an independent --dev-ref-vi and no CS development set")
    elif not args.dev_cs:
        ap.error(f"{args.rule} requires --dev-cs")
    limits = {key: getattr(args, key) for key in ("max_vi_ci_upper", "max_vi_hallucination_delta", "max_cs_domain_delta")}
    strict = any(limit is not None for limit in limits.values())
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
    if args.rule == VI_WER_RULE:
        select_vi_wer(args, cat, args.dev_ref_vi, dev_flags["vi"], cat_flags)
        return
    reserved = {tuple(normalize_hyp(term, cat)) for term in read_lines(args.reserved_terms)}
    cs_rows, vi_rows = read_jsonl(args.dev_cs), read_jsonl(args.dev_vi)
    res, vi_scores, cs_records = {}, {}, {}
    base_name = Path(args.base).name
    for d in [args.base] + args.cand:
        p = Path(d)
        if not (p / "dev_cs/hyps.jsonl").exists() or not (p / "dev_vi/hyps.jsonl").exists():
            if strict:
                raise SystemExit(f"{p}: missing development hypotheses")
            print(f"BỎ {p.name}: thiếu hyps dev")
            continue
        cs, cs_ds, cs_scores = evaluate(cs_rows, p / "dev_cs/hyps.jsonl", cat, strict, reserved)
        vi, _, vi_records = evaluate(vi_rows, p / "dev_vi/hyps.jsonl", cat, strict)
        res[p.name] = {"term_acc_macro": cs["term_acc_macro"], "span_acc": cs["span_acc"], "span_cer": cs["span_cer"],
                       "cs_wer": cs["cs_wer"], "wer_cs": cs["wer"], "n_wer_cs": cs["n_wer"], "term_recall": cs["term_recall"],
                       "wer_vi": vi["wer"],
                       "by_dataset": {d: {"n_span": a["n_span"], "term_acc_macro": a["term_acc_macro"],
                                          "span_acc": a["span_acc"], "cs_wer": a["cs_wer"]} for d, a in cs_ds.items()}}
        if strict:
            sd = ratio(sum_counts(cs_scores), "cs_wer_sd")
            res[p.name].update(cs_wer_by_domain=cs_by_domain(cs_scores),
                               cs_wer_sd=sd, cs_recall=100.0 - sd if sd is not None else None,
                               en_hallucination_per100=ratio(sum_counts(vi_records), "en_hallucination_per100"))
            vi_scores[p.name] = vi_records
            cs_records[p.name] = cs_scores
        if reserved:
            count = sum(row["ft_unseen_terms"] for row in cs_scores)
            matched = sum(row["ft_unseen_matched"] for row in cs_scores)
            res[p.name]["ft_unseen"] = {"term_recall": 100.0 * matched / count if count else None,
                                        "ref_terms": count, "matched_terms": matched}
    base = res[base_name]
    rows = []
    for name, m in res.items():
        dv = m["wer_vi"] - base["wer_vi"]
        ok = name == base_name or dv <= args.max_vi_delta
        if strict and name != base_name:
            deltas, within_limits = retention_checks(m, base, vi_scores[name], vi_scores[base_name], limits)
            ok = dv <= args.max_vi_delta + 1e-9 and within_limits and m["cs_wer"] is not None
            m.update(retention=deltas, eligible=ok,
                     paired_cs_wer_by_domain=paired_cs_by_domain(cs_records[base_name], cs_records[name]))
        rows.append((name, m, dv, ok))
    elig = [(n, m) for n, m, dv, ok in rows if ok and n != Path(args.base).name]
    if not elig:
        pick = Path(args.base).name
    elif args.rule == "cs_wer":
        pick = min(elig, key=lambda x: (
            x[1]["cs_wer"] if x[1]["cs_wer"] is not None else float("inf"), x[1]["wer_vi"],
            int(x[0][2:]) if strict and x[0].startswith("ck") and x[0][2:].isdigit() else float("inf")))[0]
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
    if dev_refs or args.rule != "term_acc_macro" or strict:
        out.update(rule={"name": args.rule, "max_vi_delta": args.max_vi_delta}, dev_refs=dev_refs, dev_flags=dev_flags,
                   catalog=str(args.catalog), catalog_flags=cat_flags)
    if strict:
        out["rule"].update({key: value for key, value in limits.items() if value is not None})
        out["rule"].update(bootstrap_iters=1000, bootstrap_seed=0)
        L += ["", "Additional retention thresholds", "",
              *[f"- {key} ≤ {value:g}" for key, value in limits.items() if value is not None]]
    write_selection(args.out, L, out)


if __name__ == "__main__":
    main()
