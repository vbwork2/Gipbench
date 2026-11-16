#!/usr/bin/env python3
"""Giải mã một benchmark bằng một model bất kỳ qua adapter (gipbench.infer.adapters) -> hyps.jsonl + run_meta.json.

  python -m gipbench.bench.transcribe --model configs/bench/models/gip15_mbs4_open.yaml --benchmark general_v1_draft \
      --out results/general_v1_draft/gip15_mbs4_open [--per-dataset 20] [--dry-run]
  python -m gipbench.bench.evaluate --benchmark general_v1_draft --hyps results/general_v1_draft/gip15_mbs4_open/hyps.jsonl

Model yaml: system, engine, model{id, revision, dir}, inference_mode, decode{...}, batch, context_vocab (tuỳ chọn).
Biến trong yaml: ${models_root} (env GIPBENCH_MODELS, mặc định <repo>/models), ${data_root}, ${repo}.
Chạy tiếp được khi bị ngắt (resume theo id). --per-dataset N: lấy N câu đầu mỗi bộ theo thứ tự sha1(id) (smoke test,
cố định, không phụ thuộc thứ tự file); kết quả khi đó KHÔNG phải số của cả benchmark.
Chặn trước khi chạy: inference_mode không thuộc allowed_inference_modes của benchmark; hotwords lấy từ catalog có span
của chính benchmark (catalog_from_bench) mà không khai báo provenance test_derived + oracle_bias.
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from gipbench.bench import registry as R
from gipbench.core.manifest import read_jsonl
from gipbench.infer.adapters import build_adapter
from gipbench.infer.adapters.base import run_manifest


def expand(s):
    if not isinstance(s, str):
        return s
    models = os.environ.get("GIPBENCH_MODELS") or str(R.P.REPO / "models")
    return R.expand(s.replace("${models_root}", models))


def expand_all(x):
    if isinstance(x, dict):
        return {k: expand_all(v) for k, v in x.items()}
    if isinstance(x, list):
        return [expand_all(v) for v in x]
    return expand(x)


def load_model_cfg(path: str) -> dict:
    cfg = expand_all(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    for k in ("system", "engine", "model", "inference_mode"):
        if k not in cfg:
            sys.exit(f"{path}: thiếu trường {k}")
    cfg["_path"] = str(path)
    return cfg


def subsample(rows, n):
    by = defaultdict(list)
    for r in rows:
        by[r.get("dataset", "?")].append(r)
    keep = set()
    for rs in by.values():
        rs = sorted(rs, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())[:n]
        keep |= {r["id"] for r in rs}
    return [r for r in rows if r["id"] in keep]


def guard(cfg: dict, bm: dict):
    errs = []
    mode = cfg["inference_mode"]
    if bm.get("allowed_inference_modes") and mode not in bm["allowed_inference_modes"]:
        errs.append(f"inference_mode {mode} không được phép trên {bm.get('_name')} ({bm['allowed_inference_modes']})")
    cv = cfg.get("context_vocab") or {}
    bcat = R.expand(bm.get("catalog"))
    if cv.get("catalog") and bm.get("catalog_from_bench") and bcat and Path(cv["catalog"]).resolve() == Path(bcat).resolve():
        if cv.get("provenance") != "test_derived" or mode != "oracle_bias":
            errs.append("hotwords lấy từ catalog có span của chính benchmark: phải khai báo provenance test_derived "
                        "và inference_mode oracle_bias")
    return errs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="configs/bench/models/<x>.yaml")
    ap.add_argument("--benchmark", help="tên benchmark trong registry")
    ap.add_argument("--manifest", help="ghi đè manifest")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-dataset", type=int, default=0, help="smoke test: N câu mỗi bộ")
    ap.add_argument("--batch", type=int)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="in cấu hình đã phân giải và số câu, không giải mã")
    args = ap.parse_args(argv)
    cfg = load_model_cfg(args.model)
    bm = R.load_benchmark(args.benchmark) if args.benchmark else {"_name": "adhoc"}
    manifest = args.manifest or R.expand(bm.get("manifest"))
    if not manifest:
        sys.exit("cần --benchmark có manifest hoặc --manifest")
    errs = guard(cfg, bm) if args.benchmark else []
    for e in errs:
        print("LỖI", e)
    if errs:
        sys.exit(2)
    rows = read_jsonl(manifest)
    if args.per_dataset:
        rows = subsample(rows, args.per_dataset)
    hours = sum(r.get("duration") or 0 for r in rows) / 3600
    print(f"{cfg['system']} ({cfg['engine']}, {cfg['inference_mode']}) trên {bm.get('_name')}: {len(rows)} câu, {hours:.2f} h")
    if args.dry_run:
        print(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, ensure_ascii=False, indent=1))
        return
    adapter = build_adapter(cfg, Path(args.out))
    batch = args.batch or int(cfg.get("batch", 8))

    def prog(i, n):
        print(f"\r{i}/{n}", end="", flush=True)
    meta = run_manifest(adapter, rows, args.out, batch=batch, resume=not args.no_resume, progress=prog, extra_meta={
        "model_config": cfg["_path"], "model_config_sha256": R.file_sha256(cfg["_path"]),
        "benchmark": bm.get("_name"), "manifest": str(manifest), "manifest_sha256": R.manifest_sha256(rows),
        "subset": {"per_dataset": args.per_dataset} if args.per_dataset else None})
    rtf = meta["rtf"]
    print(f"\nXong: {meta['n_utts']} câu (giải mã mới {meta['n_decoded_this_run']}), RTF "
          f"{'–' if rtf is None else f'{rtf:.4f}'} trên {meta['device']} -> {args.out}/hyps.jsonl")


if __name__ == "__main__":
    main()
