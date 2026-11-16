#!/usr/bin/env python3
"""Chấm một hệ thống trên một benchmark của registry (G0 / C0 / R0 / E0), nhiều bộ dữ liệu, có phiên bản.

  python -m gipbench.bench.evaluate --benchmark general_v1_draft --hyps results/general_v1/gip15/hyps.jsonl \
      [--out DIR] [--mode diagnostic|final] [--allow-draft] [--baseline-hyps results/.../hyps.jsonl] [--iters 1000]
  python -m gipbench.bench.evaluate --manifest M.jsonl --profile gipbench_strict_v1 --hyps H.jsonl   # không qua registry

Bổ sung cho gipbench.bench.score (không thay thế; score.py giữ nguyên hành vi cho v0-draft). Khác biệt:
  * --profile / benchmark.profile: hồ sơ chuẩn hoá có phiên bản (core.normprofiles); cùng lúc chấm thêm các hồ sơ
    trong `also_report_profiles` để thấy độ nhạy theo chuẩn hoá.
  * Định nghĩa metric ghi rõ trong summary.json:
      wer (space_token_v1) = (S + D + I) / N trên token tách bằng khoảng trắng sau chuẩn hoá. Tiếng Việt viết cách
                             theo âm tiết nên token ≈ âm tiết, KHÔNG phải "từ" theo tách từ.
      cer (char_v1)        = Levenshtein ký tự giữa chuỗi token nối bằng 1 dấu cách / số ký tự ref.
      cs_wer, n_wer, term_recall, span_acc, term_acc_macro: như gipbench.bench.score (core.scoring), tính trên span <en>
                             của reference. CS-WER có thể > 100% (1 từ Anh -> nhiều token chèn).
  * --mode final: thiếu / thừa / trùng id hyps -> thoát mã 2; benchmark draft -> từ chối trừ khi --allow-draft;
    inference mode phải biết (run_meta.json cạnh hyps hoặc --inference-mode) và thuộc allowed_inference_modes;
    hồ sơ không xếp hạng được (lenient) không được làm hồ sơ chính. --mode diagnostic: chỉ cảnh báo, hyp thiếu = rỗng.
  * Nhóm theo `group_by` (vd dataset, domain, region), `split_groups` (vd main_B1+B2), và `aggregate: macro_by_dataset`
    (trung bình đều WER từng bộ, không có CI, luôn kèm số từng bộ).
  * CI 95%: bootstrap theo cụm `boot_unit` (speaker / session / source / utt / boot_unit_legacy), seed cố định.
  * --baseline-hyps: Δ ghép cặp trên cùng id câu (bootstrap cặp theo cụm).
Ghi ra: utt_scores.jsonl (hồ sơ chính), summary.json, report.md.
"""
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from gipbench.bench import registry as R
from gipbench.core.align import align, char_edit_distance
from gipbench.core.evalstats import RATIOS, check_hyps, cluster_bootstrap, paired_bootstrap, ratio, sum_counts
from gipbench.core.manifest import read_jsonl
from gipbench.core.normprofiles import Profile, get_profile
from gipbench.core.scoring import agg, score_utt
from gipbench.core.textnorm import Catalog, normalize_hyp, normalize_ref

SCORER = {"name": "gipbench.bench.evaluate", "version": "1.0.0"}
WER_DEF = ("space_token_v1: (S+D+I)/N trên token tách theo khoảng trắng sau hồ sơ chuẩn hoá "
           "(token ≈ âm tiết tiếng Việt, không phải từ đã tách)")
CER_DEF = "char_v1: Levenshtein ký tự trên chuỗi token nối bằng 1 dấu cách / số ký tự ref"
METRICS = ["wer", "cer", "cs_wer", "n_wer", "term_recall"]


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=R.P.REPO, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def unit_of(r: Dict, rec: Dict, boot_unit: str) -> str:
    if boot_unit == "boot_unit_legacy":
        return rec["boot_unit"]
    if boot_unit == "utt":
        return r["id"]
    return str(r.get(boot_unit) or r["id"])


def score_rows(rows: List[Dict], hyps: Dict[str, str], cat: Catalog, prof: Profile, label_field: str,
               boot_unit: str, keys: List[str]) -> List[Dict]:
    out = []
    mode = prof.catalog_mode
    for r in rows:
        ref_raw = r.get(label_field) or r.get("text_tagged") or r.get("text") or ""
        ref = prof.pre(ref_raw)
        hyp = prof.pre(hyps.get(r["id"], ""))
        rr = dict(r, text_tagged=ref, split=r.get("split") or "all", speaker=r.get("speaker") or r["id"])
        rec = score_utt(rr, hyp, cat, "text_tagged")
        if prof.lenient:  # hồ sơ chẩn đoán: số chính lấy theo bảng phonetic
            rec["err_en"], rec["matched_terms"] = rec["err_en_len"], rec["matched_terms_len"]
        rt = [t.tok for t in normalize_ref(ref, cat, mode)[0]]
        ht = normalize_hyp(hyp, cat, mode)
        if prof.lenient:  # WER theo cùng bảng
            ops = align(rt, ht)
            rec["err"] = sum(o[0] != "C" for o in ops)
            rec["n"] = len(rt)
        a, b = " ".join(rt), " ".join(ht)
        rec.update(cerr=char_edit_distance(a, b), nc=len(a), duration=r.get("duration") or 0.0,
                   unit=unit_of(rr, rec, boot_unit))
        for k in keys:
            rec[k] = r.get(k) if r.get(k) not in (None, "") else "unk"
        out.append(rec)
    return out


def summarize(recs: List[Dict], iters: int, seed: int, with_ci: bool = True) -> Dict:
    s = sum_counts(recs)
    d = {"utt": len(recs), "hours": round(sum(r.get("duration") or 0 for r in recs) / 3600, 4),
         "n_tok": s.get("n", 0), "n_en": s.get("n_en", 0), "units": len({r["unit"] for r in recs})}
    for m in METRICS:
        d[m] = ratio(s, m)
    if any(r.get("spans") for r in recs):
        a = agg(recs)
        d.update(span_acc=a["span_acc"], term_acc_macro=a["term_acc_macro"], span_cer=a["span_cer"],
                 n_span=a["n_span"], term_precision=a["term_precision"])
    if with_ci:
        ms = [m for m in METRICS if d.get(m) is not None]
        for m, ci in cluster_bootstrap(recs, ms, "unit", iters, seed).items():
            d[m + "_ci95"] = list(ci) if ci else None
    return d


def group_summaries(recs, keys, split_groups, iters, seed):
    out = {"overall": summarize(recs, iters, seed)}
    for name, splits in (split_groups or {}).items():
        sub = [r for r in recs if r["split"] in splits]
        if sub:
            out[f"split_group:{name}"] = summarize(sub, iters, seed)
    for k in keys:
        g = defaultdict(list)
        for r in recs:
            g[str(r.get(k, "unk"))].append(r)
        if len(g) < 2 and k != "dataset":
            continue
        for v, sub in sorted(g.items()):
            out[f"{k}:{v}"] = summarize(sub, iters, seed, with_ci=(k == "dataset"))
    return out


def macro_by_dataset(groups: Dict, metric: str) -> Optional[Dict]:
    vals = {k.split(":", 1)[1]: v[metric] for k, v in groups.items() if k.startswith("dataset:") and v.get(metric) is not None}
    if not vals:
        return None
    return {"metric": metric, "value": sum(vals.values()) / len(vals), "n_datasets": len(vals), "components": vals,
            "note": "trung bình đều theo bộ, không có CI; luôn đọc kèm số từng bộ"}


def fmt(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def ci_txt(c):
    return f"[{c[0]:.2f}, {c[1]:.2f}]" if c else ""


def write_report(path: Path, summ: Dict):
    L = [f"# Kết quả: {summ['system']} trên {summ['benchmark']['id']} {summ['benchmark']['version']}", ""]
    b = summ["benchmark"]
    if b.get("evidence_level") != "locked":
        L += [f"> **Benchmark {b.get('evidence_level')}** (chưa khoá). Số liệu dùng để chẩn đoán, không phải kết quả cuối.", ""]
    L += [f"- Chế độ chấm: `{summ['mode']}` · hồ sơ chính: `{summ['profile']['name']}` · inference mode: "
          f"`{summ['inference_mode']}`", f"- WER: {WER_DEF}", f"- CER: {CER_DEF}",
          f"- CI 95%: bootstrap theo `{summ['boot_unit']}`, {summ['iters']} lần, seed {summ['seed']}",
          f"- manifest_sha256 `{summ['manifest_sha256'][:16]}`, hyps_sha256 `{(summ['hyps_sha256'] or '')[:16]}`", ""]
    for w in summ["warnings"]:
        L.append(f"- **Cảnh báo:** {w}")
    if summ["warnings"]:
        L.append("")
    for pname, groups in summ["profiles"].items():
        L += [f"## Hồ sơ `{pname}`", "", "| Nhóm | Utt | Giờ | WER | CI95 | CER | CS-WER | N-WER | Term recall |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for g, v in groups.items():
            if g == "macro_by_dataset":
                continue
            L.append(f"| {g} | {v['utt']} | {v['hours']:.2f} | {fmt(v['wer'])} | {ci_txt(v.get('wer_ci95'))} | "
                     f"{fmt(v['cer'])} | {fmt(v['cs_wer'])} | {fmt(v['n_wer'])} | {fmt(v['term_recall'])} |")
        mac = groups.get("macro_by_dataset")
        if mac:
            L += ["", f"Macro theo bộ ({mac['metric']}): **{mac['value']:.2f}** trên {mac['n_datasets']} bộ ({mac['note']})."]
        L.append("")
    if summ.get("paired"):
        L += ["## So cặp với baseline (cùng id câu)", "", "| Nhóm | Metric | Baseline | Hệ này | Δ | CI95 Δ | Câu chung |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        for g, ps in summ["paired"].items():
            for p in ps:
                L.append(f"| {g} | {p['metric']} | {fmt(p['a'])} | {fmt(p['b'])} | {fmt(p['delta'])} | {ci_txt(p['ci95'])} | {p['n_common']} |")
        L.append("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


def load_hyps(path: str):
    rows = read_jsonl(path)
    return rows, {h["id"]: h.get("hyp", "") or "" for h in rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", help="tên trong configs/bench/benchmarks/ hoặc đường dẫn yaml")
    ap.add_argument("--manifest", help="ghi đè manifest (hoặc dùng không cần --benchmark)")
    ap.add_argument("--catalog", help="ghi đè catalog")
    ap.add_argument("--profile", help="ghi đè hồ sơ chuẩn hoá chính")
    ap.add_argument("--hyps", required=True)
    ap.add_argument("--system", help="tên hệ thống (mặc định: tên thư mục chứa hyps)")
    ap.add_argument("--out")
    ap.add_argument("--mode", choices=["diagnostic", "final"], default="diagnostic")
    ap.add_argument("--allow-draft", action="store_true", help="final mode trên benchmark chưa khoá (kết quả ghi draft)")
    ap.add_argument("--inference-mode", choices=sorted(R.MODES), help="khi hyps không có run_meta.json")
    ap.add_argument("--run-meta", help="run_meta.json (mặc định: cạnh hyps)")
    ap.add_argument("--baseline-hyps", help="hyps của hệ baseline để so cặp")
    ap.add_argument("--label-field", default="text_tagged")
    ap.add_argument("--iters", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args(argv)

    bm = R.load_benchmark(args.benchmark) if args.benchmark else {
        "_name": "adhoc", "benchmark_id": "adhoc", "version": "adhoc", "track": "E0", "evidence_level": "draft",
        "status": "active", "profile": "gipbench_strict_v1", "boot_unit": "speaker", "seed": 0, "iters": 1000,
        "group_by": ["dataset"], "allowed_inference_modes": sorted(R.MODES), "primary_metric": "wer"}
    manifest = args.manifest or R.expand(bm.get("manifest"))
    if not manifest:
        sys.exit(f"benchmark {bm['_name']} chưa có manifest (status {bm.get('status')})")
    catalog = args.catalog or R.expand(bm.get("catalog"))
    prof = get_profile(args.profile or bm["profile"])
    profiles = [prof] + [get_profile(p) for p in (bm.get("also_report_profiles") or []) if p != prof.name]
    iters = args.iters or bm.get("iters", 1000)
    seed = bm.get("seed", 0) if args.seed is None else args.seed
    keys = list(bm.get("group_by") or ["dataset"])
    if "dataset" not in keys:
        keys.insert(0, "dataset")
    warnings, errors = [], []

    rows = read_jsonl(manifest)
    hrows, hyps = load_hyps(args.hyps)
    chk = check_hyps([r["id"] for r in rows], [h["id"] for h in hrows], [h.get("hyp", "") for h in hrows])
    if not chk["ok"]:
        msg = (f"hyps không khớp manifest: thiếu {len(chk['missing'])}, thừa {len(chk['extra'])}, trùng id hyps "
               f"{len(chk['duplicate_hyp_ids'])}, trùng id manifest {len(chk['duplicate_manifest_ids'])}")
        (errors if args.mode == "final" else warnings).append(msg + ("" if args.mode == "final" else " (hyp thiếu tính là rỗng)"))
    frozen = R.check_frozen(bm, rows)
    if frozen:
        errors.append(frozen)
    run_meta_path = Path(args.run_meta) if args.run_meta else Path(args.hyps).parent / "run_meta.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else None
    imode = args.inference_mode or (run_meta or {}).get("inference_mode") or "unknown"
    if ((run_meta or {}).get("context_vocab") or {}).get("provenance") == "test_derived" and imode != "oracle_bias":
        errors.append(f"hotwords / từ vựng ngữ cảnh lấy từ test nhưng inference_mode = {imode} (phải là oracle_bias)")
    if bm.get("catalog_from_bench") and imode == "context_bias":
        warnings.append("catalog của benchmark chứa span lấy từ chính test: biasing từ catalog này = oracle_bias")
    allowed = set(bm.get("allowed_inference_modes") or [])
    if imode == "unknown":
        (errors if args.mode == "final" else warnings).append("inference mode chưa rõ (không có run_meta.json / --inference-mode)")
    elif imode not in allowed:
        (errors if args.mode == "final" else warnings).append(f"inference mode {imode} không thuộc {sorted(allowed)} của benchmark")
    if args.mode == "final":
        if bm.get("evidence_level") != "locked" and not args.allow_draft:
            errors.append(f"benchmark evidence_level={bm.get('evidence_level')}: final mode cần bản locked (hoặc --allow-draft)")
        if not prof.rankable:
            errors.append(f"hồ sơ {prof.name} chỉ để chẩn đoán, không dùng làm hồ sơ chính ở final mode")
        reg = R.load_datasets()
        for e in R.validate_benchmark(bm, reg) if args.benchmark else []:
            errors.append(e)
        dsreg = reg.get("datasets") or {}
        srcs = {s["dataset"] for s in bm.get("sources") or []}
        for d in sorted(srcs):
            lic = str((dsreg.get(d) or {}).get("license") or "unknown")
            if lic.lower() == "unknown":
                errors.append(f"nguồn {d}: license chưa rõ")
    if bm.get("evidence_level") != "locked":
        warnings.append(f"benchmark {bm.get('evidence_level')}: chưa khoá, số liệu không phải kết quả cuối")
    for e in errors:
        print("LỖI", e)
    if errors:
        sys.exit(2)
    for w in warnings:
        print("CẢNH BÁO", w)

    cat = Catalog.load(catalog)
    out = Path(args.out or Path(args.hyps).parent)
    out.mkdir(parents=True, exist_ok=True)
    system = args.system or Path(args.hyps).parent.name
    summ = {"system": system, "scorer": SCORER, "git_commit": git_commit(), "mode": args.mode,
            "benchmark": {"id": bm.get("benchmark_id"), "version": bm.get("version"), "track": bm.get("track"),
                          "evidence_level": bm.get("evidence_level"), "config": bm.get("_path")},
            "manifest": str(manifest), "manifest_sha256": R.manifest_sha256(rows), "hyps": str(args.hyps),
            "hyps_sha256": R.file_sha256(args.hyps), "catalog": catalog, "catalog_sha256": R.file_sha256(catalog),
            "profile": prof.spec() | {"sha256": prof.sha256()}, "wer_definition": WER_DEF, "cer_definition": CER_DEF,
            "boot_unit": bm.get("boot_unit", "speaker"), "iters": iters, "seed": seed, "hyp_check": {
                k: (len(v) if isinstance(v, list) else v) for k, v in chk.items()},
            "inference_mode": imode, "run_meta": run_meta, "warnings": warnings, "profiles": {}}
    main_recs = None
    for p in profiles:
        recs = score_rows(rows, hyps, cat, p, args.label_field, summ["boot_unit"], keys)
        groups = group_summaries(recs, keys, bm.get("split_groups"), iters, seed)
        if bm.get("aggregate") == "macro_by_dataset":
            groups["macro_by_dataset"] = macro_by_dataset(groups, bm.get("primary_metric", "wer"))
        summ["profiles"][p.name] = groups
        if p is prof:
            main_recs = recs
    with open(out / "utt_scores.jsonl", "w", encoding="utf-8") as f:
        for r in main_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if args.baseline_hyps:
        _, bh = load_hyps(args.baseline_hyps)
        base = score_rows(rows, bh, cat, prof, args.label_field, summ["boot_unit"], keys)
        summ["baseline_hyps"] = args.baseline_hyps
        pm = bm.get("primary_metric", "wer")
        ms = [pm] + [m for m in ("wer", "cer") if m != pm]
        paired = {"overall": [paired_bootstrap(base, main_recs, m, "unit", iters, seed) for m in ms]}
        for v in sorted({r["dataset"] for r in main_recs}):
            a = [r for r in base if r["dataset"] == v]
            b = [r for r in main_recs if r["dataset"] == v]
            paired[f"dataset:{v}"] = [paired_bootstrap(a, b, m, "unit", iters, seed) for m in ms]
        for name, splits in (bm.get("split_groups") or {}).items():
            a = [r for r in base if r["split"] in splits]
            b = [r for r in main_recs if r["split"] in splits]
            if a:
                paired[f"split_group:{name}"] = [paired_bootstrap(a, b, m, "unit", iters, seed) for m in ms]
        summ["paired"] = {g: [p for p in ps if p["delta"] is not None] for g, ps in paired.items()}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(out / "report.md", summ)
    ov = summ["profiles"][prof.name]["overall"]
    print(f"[{system}] {bm.get('benchmark_id')} {bm.get('version')} ({prof.name}): {ov['utt']} câu, WER {fmt(ov['wer'])} "
          f"{ci_txt(ov.get('wer_ci95'))} · CER {fmt(ov['cer'])} · CS-WER {fmt(ov['cs_wer'])} -> {out}")
    return summ


if __name__ == "__main__":
    main()
