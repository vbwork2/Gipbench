"""Đọc và kiểm registry đánh giá: configs/bench/datasets.yaml và configs/bench/benchmarks/<tên>.yaml.

  python -m gipbench.bench.registry                           # kiểm mọi file, in bảng tóm tắt; thoát mã 1 nếu sai
  python -m gipbench.bench.registry --hash general_v1_draft   # tính manifest_sha256 của một benchmark

Đường dẫn trong yaml được mở rộng: ${data_root}, ${results_root}, ${repo} (từ gipbench.core.paths).
manifest_sha256 = sha256 của các dòng manifest đã chuẩn hoá (JSON sort_keys, xếp theo id, BỎ trường `audio` vì là
đường dẫn phụ thuộc máy, và các trường bắt đầu bằng "_"). Danh tính audio nằm ở trường `audio_sha256` của từng dòng.
"""
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from gipbench.core import paths as P
from gipbench.core.fileio import sha256_file_or_none as file_sha256  # noqa: F401  (tên cũ: R.file_sha256)
from gipbench.core.manifest import manifest_sha256, read_jsonl  # noqa: F401  (R.manifest_sha256)
from gipbench.core.normprofiles import PROFILES

REG_DIR = P.REPO / "configs" / "bench"
TRACKS = {"G0", "C0", "C1", "Z0", "R0", "E0", "C0+R0"}
EVIDENCE = {"draft", "human_reviewed", "locked"}
STATUS = {"active", "planned", "blocked"}
MODES = {"open_vocab", "lm", "context_bias", "oracle_bias"}
OVERLAP = {"confirmed", "known_possible", "not_listed", "unknown"}
DS_REQUIRED = ["hf_id", "revision", "license", "access", "eval_split", "official_scorer", "speaker_key",
               "pretraining_overlap"]
BM_REQUIRED = ["benchmark_id", "version", "track", "purpose", "evidence_level", "status", "manifest", "profile",
               "primary_metric", "boot_unit", "seed", "allowed_inference_modes", "sources"]


def expand(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    return (str(s).replace("${data_root}", str(P.path("data_root")))
            .replace("${results_root}", str(P.path("results_root"))).replace("${repo}", str(P.REPO)))


def load_datasets(path: Optional[Path] = None) -> Dict:
    return yaml.safe_load(Path(path or REG_DIR / "datasets.yaml").read_text(encoding="utf-8"))


def load_benchmark(name_or_path: str) -> Dict:
    p = Path(name_or_path)
    if not p.suffix:
        p = REG_DIR / "benchmarks" / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    d["_path"] = str(p)
    d["_name"] = p.stem
    return d


def validate_datasets(reg: Dict) -> List[str]:
    errs = []
    models = set(reg.get("models") or {})
    for k, d in (reg.get("datasets") or {}).items():
        for f in DS_REQUIRED:
            if f not in d:
                errs.append(f"datasets.{k}: thiếu trường {f}")
        for m, v in (d.get("pretraining_overlap") or {}).items():
            if m not in models:
                errs.append(f"datasets.{k}.pretraining_overlap: model lạ '{m}'")
            if v not in OVERLAP:
                errs.append(f"datasets.{k}.pretraining_overlap.{m}: '{v}' không thuộc {sorted(OVERLAP)}")
        if not d.get("license") or str(d.get("license")).strip().lower() == "unknown":
            errs.append(f"datasets.{k}: license chưa rõ")
    return errs


def validate_benchmark(bm: Dict, reg: Dict) -> List[str]:
    n = bm.get("_name", "?")
    errs = [f"{n}: thiếu trường {f}" for f in BM_REQUIRED if f not in bm]
    if bm.get("track") not in TRACKS:
        errs.append(f"{n}: track '{bm.get('track')}' không thuộc {sorted(TRACKS)}")
    if bm.get("evidence_level") not in EVIDENCE:
        errs.append(f"{n}: evidence_level '{bm.get('evidence_level')}' không thuộc {sorted(EVIDENCE)}")
    if bm.get("status") not in STATUS:
        errs.append(f"{n}: status '{bm.get('status')}' không thuộc {sorted(STATUS)}")
    for p in [bm.get("profile")] + list(bm.get("also_report_profiles") or []):
        if p not in PROFILES:
            errs.append(f"{n}: hồ sơ chuẩn hoá lạ '{p}'")
    modes = set(bm.get("allowed_inference_modes") or [])
    if modes - MODES:
        errs.append(f"{n}: inference mode lạ {sorted(modes - MODES)}")
    if bm.get("track") in ("G0", "Z0") and "oracle_bias" in modes:
        errs.append(f"{n}: track {bm['track']} không được dùng oracle_bias (hotwords lấy từ test)")
    ds = reg.get("datasets") or {}
    for s in bm.get("sources") or []:
        if s.get("dataset") not in ds:
            errs.append(f"{n}: nguồn '{s.get('dataset')}' không có trong datasets.yaml")
    if bm.get("status") == "active" and not bm.get("manifest"):
        errs.append(f"{n}: status active nhưng không có manifest")
    if bm.get("evidence_level") == "locked" and not bm.get("manifest_sha256"):
        errs.append(f"{n}: locked nhưng chưa ghi manifest_sha256")
    if bm.get("boot_unit") == "per_dataset" and not isinstance(bm.get("boot_units"), dict):
        errs.append(f"{n}: boot_unit per_dataset cần boot_units: {{default: ..., <bộ>: ...}}")
    return errs


def check_frozen(bm: Dict, rows: List[Dict]) -> Optional[str]:
    """Lỗi nếu benchmark đã khoá (có manifest_sha256) mà manifest hiện tại khác; hoặc benchmark `require_audio_sha`
    ở mức locked mà còn câu thiếu audio_sha256 (khoá bằng metadata không đủ: audio sau fetch có thể lệch)."""
    if bm.get("require_audio_sha") and bm.get("evidence_level") == "locked":
        miss = sum(1 for r in rows if not r.get("audio_sha256"))
        if miss:
            return f"benchmark require_audio_sha nhưng {miss} / {len(rows)} câu thiếu audio_sha256: chưa khoá được"
    want = bm.get("manifest_sha256")
    if not want:
        return None
    got = manifest_sha256(rows)
    return None if got == want else f"manifest đã đổi so với bản khoá: {got[:12]} != {want[:12]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hash", metavar="BENCHMARK", help="in manifest_sha256 của benchmark (cần manifest trên máy)")
    args = ap.parse_args()
    reg = load_datasets()
    if args.hash:
        bm = load_benchmark(args.hash)
        rows = read_jsonl(expand(bm["manifest"]))
        print(manifest_sha256(rows), f"({len(rows)} dòng)")
        return
    errs = validate_datasets(reg)
    print(f"datasets.yaml: {len(reg.get('datasets') or {})} nguồn, registry_version {reg.get('registry_version')}")
    for p in sorted((REG_DIR / "benchmarks").glob("*.yaml")):
        bm = load_benchmark(str(p))
        e = validate_benchmark(bm, reg)
        errs += e
        m = expand(bm.get("manifest"))
        have = "có" if m and Path(m).exists() else "chưa có"
        print(f"  {p.stem:24s} {bm.get('track'):6s} {bm.get('evidence_level'):14s} {bm.get('status'):8s} "
              f"manifest {have}" + (f"  [{len(e)} lỗi]" if e else ""))
    for e in errs:
        print("LỖI", e)
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
