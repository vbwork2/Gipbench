#!/usr/bin/env python3
"""Chạy bench cho một system: decode (nếu cần) -> chấm. Bench chỉ đọc bench release + system YAML.

  python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/e1_hw1.5.yaml
  python -m gipbench.bench.run --bench bench/v0-draft --system configs/bench/systems/*.yaml --hyps-only   # chỉ chấm lại
  python -m gipbench.bench.run --bench bench/v0-draft --system ... --dry-run                             # in lệnh
  python -m gipbench.bench.leaderboard --results results/v0_draft --baseline e0_baseline --rules configs/bench/rules.yaml \
      --systems e0_baseline e1_hw1.5 e3_v1 e7_v1 --out reports/bench/v0_draft/leaderboard

System YAML (configs/bench/systems/<system>.yaml, đặc tả docs/CONTRACTS.md):
  system: e7_v1
  model: ${exp_root}/artifacts/e3_v1      # thư mục có encoder/decoder/joiner *.onnx (hoặc <dir>/onnx/); hoặc thay bằng
  hyps: path/to/hyps.jsonl                # hyps có sẵn (model ngoài, hậu xử lý E8): bỏ qua decode
  decode: {decoding_method, beam, quantize, threads, batch, blank_penalty, case, hotwords_catalog, hotwords_exclude_b3,
           hotwords_score, hotwords_file, hotwords_distractors, hotwords_distractors_n, hotwords_seed}
Khoá decode = tham số của gipbench.infer.decode (gạch dưới thay gạch ngang). hotwords_catalog nhận release ref
(catalog/v0-draft) hoặc đường dẫn .tsv. Kết quả: <results>/<system>/{hyps.jsonl, decode_summary.json, utt_scores.jsonl,
summary.json, slices.md}; log decode: <results>/<system>_decode.log. <results> mặc định = results/<version bench, '-' -> '_'>.
Đã có hyps.jsonl thì không decode lại (trừ --force). Đổi metric / luật chọn: --hyps-only rồi chạy lại leaderboard.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core import release as R
from gipbench.core.paths import REPO

DECODE_KEYS = {"splits", "quantize", "decoding_method", "beam", "threads", "batch", "blank_penalty", "hotwords_file",
               "hotwords_catalog", "hotwords_exclude_b3", "hotwords_score", "hotwords_distractors",
               "hotwords_distractors_n", "hotwords_seed", "case", "save_logprobs"}


def load_system(path):
    s = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for k in ("model", "hyps"):
        if isinstance(s.get(k), str):
            s[k] = s[k].replace("${exp_root}", str(paths.get("exp_root")))
    if not s.get("system") or not (s.get("model") or s.get("hyps")):
        raise SystemExit(f"{path}: cần system và model (hoặc hyps)")
    bad = set(s.get("decode") or {}) - DECODE_KEYS
    if bad:
        raise SystemExit(f"{path}: khoá decode không hợp lệ {sorted(bad)} (hợp lệ: {sorted(DECODE_KEYS)})")
    return s


def model_dir(m):
    p = Path(m) if Path(m).is_absolute() else REPO / m
    return p / "onnx" if (p / "onnx").is_dir() else p


def decode_args(dec, resolve_refs=True):
    out = []
    for k, v in (dec or {}).items():
        if k == "hotwords_catalog" and isinstance(v, str) and not v.endswith(".tsv") and resolve_refs:
            v = str(R.resolve(v).catalog)
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            out += [flag] if v else []
        elif isinstance(v, list):
            out += [flag, *map(str, v)]
        elif v is not None:
            out += [flag, str(v)]
    return out


def show(cmd):
    print("$ " + " ".join(str(c) for c in cmd))


def run_one(sys_path, bench, results, args):
    s = load_system(sys_path)
    name, out = s["system"], results / s["system"]
    hyps = out / "hyps.jsonl"
    if s.get("hyps"):
        print(f"== {name}: hyps có sẵn {s['hyps']}")
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(s["hyps"], hyps)
    elif args.hyps_only or (hyps.exists() and not args.force):
        print(f"== {name}: dùng hyps đã có ({hyps})" if hyps.exists() else f"== {name}: THIẾU {hyps} (--hyps-only)")
        if not hyps.exists() and not args.dry_run:
            return 1
    else:
        cmd = [sys.executable, "-m", "gipbench.infer.decode", "--bench", bench.manifest, "--model-dir", model_dir(s["model"]),
               "--system", name, "--results", results, *decode_args(s.get("decode"), not args.dry_run)]
        print(f"== {name}: decode")
        show(cmd)
        if not args.dry_run:
            results.mkdir(parents=True, exist_ok=True)
            with open(results / f"{name}_decode.log", "w", encoding="utf-8") as log:
                rc = subprocess.run([str(c) for c in cmd], cwd=REPO, stdout=log, stderr=subprocess.STDOUT).returncode
            if rc:
                print(f"LỖI decode {name} (rc={rc}), xem {results / f'{name}_decode.log'}")
                return rc
    cmd = [sys.executable, "-m", "gipbench.bench.score", "--bench", bench.manifest]
    if bench.catalog:
        cmd += ["--catalog", bench.catalog]
    cmd += ["--hyps", hyps, "--iters", args.iters]
    show(cmd)
    if args.dry_run:
        return 0
    return subprocess.run([str(c) for c in cmd], cwd=REPO).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True, help="bench release, vd bench/v0-draft")
    ap.add_argument("--system", nargs="+", required=True, help="system YAML (một hoặc nhiều)")
    ap.add_argument("--results", help="mặc định results/<version bench>")
    ap.add_argument("--hyps-only", action="store_true", help="không decode, chỉ chấm lại hyps đã có")
    ap.add_argument("--force", action="store_true", help="decode lại kể cả khi đã có hyps")
    ap.add_argument("--iters", default="1000", help="số vòng bootstrap của score")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    bench = R.resolve(args.bench, verify=not args.dry_run)
    if bench.meta.get("flags", {}).get("catalog_from_bench"):
        print(f"CẢNH BÁO: catalog của {args.bench} có span lấy từ chính bench: system dùng hotwords từ catalog này là cận "
              "trên lạc quan.")
    results = Path(args.results) if args.results else paths.path("results_root") / bench.meta["version"].replace("-", "_")
    rc = 0
    for sp in args.system:
        rc = run_one(sp, bench, results, args) or rc
    sys.exit(rc)


if __name__ == "__main__":
    main()
