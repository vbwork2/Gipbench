#!/usr/bin/env python3
"""Đóng gói bản share của bench: manifest KHÔNG kèm audio + datasheet + config chấm + SHA256SUMS.

  python -m gipbench.dataset.release.package --benchmarks general_v1 code_switch_v1 cs_ggroup_v1 \
      --name gipbench-bench-v1.0 [--out <data_root>/share]
Benchmark có manifest_sha256 (bản khoá): manifest share phải trùng hash đó (và catalog trùng catalog_sha256), nếu không thì dừng
trước khi ghi gì.

Mỗi manifest share = manifest của benchmark, bỏ trường `audio` (đường dẫn máy) và mọi trường bắt đầu bằng "_", thêm
`audio_derivation` (upstream_bytes | soxr_hq_16k_mono_pcm16) để fetch biết cách kiểm. Từ chối đóng gói nếu có dòng
license test_only / nguồn riêng tư, hoặc nguồn không có trong gipbench.dataset.release.fetch.SOURCES.
Người nhận: python -m gipbench.dataset.release.fetch --manifest manifests/<x>.jsonl --out ... (xem README trong gói).
Gói kèm code/ (src gipbench tối thiểu + configs + pyproject, tính bằng AST từ fetch / leak_audit / evaluate / transcribe):
người nhận cài bằng `pip install -e code/`, không cần repo. Đóng gói dừng nếu code đó import module train / dữ liệu train
(CODE_FORBIDDEN).
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.release.build_general import manifest_sha256
from gipbench.dataset.release.fetch import DERIVED, SOURCES, sha256_file

PRIVATE_LICENSES = {"test_only"}
DOCS = ["docs/datasheet_bench_v1.md", "docs/benchmark_protocol_v1.md", "docs/evaluation_metrics_v1.md"]
CONFIGS = ["configs/bench/datasets.yaml", "configs/dataset/labeling/term_decisions.tsv"]  # term_decisions: nguồn tag <en> của C1


def manifest_path(name: str) -> Path:
    import yaml
    bm = yaml.safe_load((P.REPO / "configs" / "bench" / "benchmarks" / f"{name}.yaml").read_text(encoding="utf-8"))
    return Path(bm["manifest"].replace("${data_root}", str(P.path("data_root")))), bm


def share_rows(rows):
    out = []
    for r in rows:
        if str(r.get("license") or "").lower() in PRIVATE_LICENSES:
            raise SystemExit(f"{r['id']}: license {r.get('license')} không được share")
        if r["dataset"] not in SOURCES:
            raise SystemExit(f"{r['id']}: bộ {r['dataset']} không có nguồn tải trong fetch.SOURCES")
        if not r.get("upstream_id") or not (r.get("audio_sha256") or r.get("audio_derivation") == "pinned_source_extract"):
            raise SystemExit(f"{r['id']}: thiếu upstream_id / audio_sha256")
        s = {k: v for k, v in r.items() if k != "audio" and not k.startswith("_")}
        s.setdefault("audio_derivation", "soxr_hq_16k_mono_pcm16" if r["dataset"] in DERIVED else "upstream_bytes")
        out.append(s)
    return sorted(out, key=lambda x: x["id"])


def strip_text(rows):
    """Bộ không cho phân phối lại lời thoại (text_redistributable = false, vd PhoAudiobook, GigaSpeech2): bỏ các trường
    lời thoại khỏi bản share; người nhận chạy fetch (đã đồng ý điều khoản trên HF) để lấy lại đúng giá trị từ nguồn."""
    from gipbench.dataset.ingest.evalsets import TEXT_FIELDS
    return [{k: v for k, v in r.items() if not (r.get("text_redistributable") is False and k in TEXT_FIELDS)}
            for r in rows]


# --- code chấm điểm kèm gói (--with-code) ---
# Gói chạy độc lập: người nhận không cần repo. Chỉ copy các module mà fetch / leak_audit / evaluate / transcribe import
# (tính bằng AST lúc đóng gói, nên không lệch khi code đổi), và DỪNG nếu có module thuộc phần train / dữ liệu train.
CODE_ROOTS = ["gipbench.bench.evaluate", "gipbench.bench.transcribe",
              "gipbench.dataset.release.fetch", "gipbench.dataset.release.leak_audit"]
CODE_FORBIDDEN = ("gipbench.model", "gipbench.dataset.pseudolabel", "gipbench.dataset.label", "gipbench.dataset.derive",
                  "gipbench.dataset.synth", "gipbench.dataset.report")
CODE_CONFIGS = ["configs/bench/datasets.yaml", "configs/dataset/labeling/term_decisions.tsv"] + [
    f"configs/bench/asr_models/{n}.yaml" for n in ("gip15_mbs4_open", "gip15_greedy_open", "viegem_greedy", "viegem_lm")]
# Adapter whisper / qwen3 (chỉ để chẩn đoán) kéo theo infer/asr_hyps.py và cache.py, có docstring về nguồn dữ liệu nội bộ:
# không kèm. Người nhận chấm model khác bằng script tự viết (README mục 4a) hoặc adapter Gipformer / VieGem.
CODE_SKIP = {"gipbench.infer.adapters.hf_multilingual"}
# core/paths.py của repo có đường dẫn máy, khoá của pipeline train (icefall, exp_root) và file chặn bộ test-only:
# bản kèm gói chỉ có các khoá bench dùng.
CODE_PATHS = '''"""Đường dẫn gốc của gói code (REPO = thư mục chứa src/ và configs/) và thư mục dữ liệu / kết quả.

data_root và results_root đọc từ biến môi trường GIPBENCH_DATA / GIPBENCH_RESULTS, mặc định <REPO>/data và <REPO>/results.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_DEFAULTS = {"data_root": "data", "results_root": "results"}
_ENV = {"data_root": "GIPBENCH_DATA", "results_root": "GIPBENCH_RESULTS"}


def get(key):
    return os.environ.get(_ENV[key]) or _DEFAULTS[key]


def path(key):
    """Đường dẫn tuyệt đối: ~ được mở rộng, đường dẫn tương đối tính từ REPO."""
    p = Path(os.path.expanduser(str(get(key))))
    return p if p.is_absolute() else REPO / p
'''
CODE_PYPROJECT = """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "gipbench"
version = "{version}"
description = "Gipbench: code chấm điểm, tải audio và kiểm trùng dữ liệu cho bench ASR tiếng Việt"
requires-python = ">=3.10"
dependencies = ["numpy", "pyyaml"]

[project.optional-dependencies]
audio = ["soundfile", "soxr"]
asr = ["faster-whisper"]

[tool.setuptools.packages.find]
where = ["src"]
"""


def _module_file(src: Path, mod: str):
    p = src / Path(*mod.split("."))
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    return None


def code_closure(src: Path, roots=CODE_ROOTS):
    """Tập module gipbench.* mà `roots` import (kể cả gián tiếp, và các package cha). Trả {tên: file}."""
    import ast
    seen, todo = {}, list(roots)
    while todo:
        m = todo.pop()
        if m in seen or m in CODE_SKIP:
            continue
        f = _module_file(src, m)
        if f is None:
            continue  # tên import không phải module (vd hàm / lớp)
        seen[m] = f
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                todo += [n.module] + [f"{n.module}.{a.name}" for a in n.names]
        parts = m.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
    return {m: f for m, f in seen.items() if m == "gipbench" or m.startswith("gipbench.")}


def copy_code(dst: Path, version: str):
    src = P.REPO / "src"
    mods = code_closure(src)
    bad = sorted(m for m in mods if m.startswith(CODE_FORBIDDEN))
    if bad:
        raise SystemExit(f"code chấm điểm import module không được share: {bad}")
    out = dst / "code"
    for m, f in sorted(mods.items()):
        t = out / "src" / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes((CODE_PATHS.encode("utf-8") if m == "gipbench.core.paths" else f.read_bytes()).replace(b"\r\n", b"\n"))
    for rel in CODE_CONFIGS:
        s = P.REPO / rel
        if not s.exists():
            raise SystemExit(f"thiếu {rel} để kèm trong code/")
        t = out / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(s.read_bytes().replace(b"\r\n", b"\n"))
    (out / "pyproject.toml").write_text(CODE_PYPROJECT.format(version=version), encoding="utf-8", newline="\n")
    return sorted(mods)


def clean_meta(m):
    """run_meta.json cho bản share: bỏ đường dẫn máy (manifest, file model), giữ cấu hình giải mã + sha256."""
    if isinstance(m, dict):
        return {k: clean_meta(v) for k, v in m.items() if k not in ("path", "manifest", "dir", "encoder_dir")}
    if isinstance(m, list):
        return [clean_meta(x) for x in m]
    if isinstance(m, str) and (m.startswith("/") or (len(m) > 2 and m[1] == ":" and m[2] in "\\/")):
        return "<local path>"
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", help="mặc định: <data_root>/share")
    ap.add_argument("--baselines", nargs="*", default=[], metavar="DIR",
                    help="thư mục kết quả <results>/<benchmark>/<hệ> (hyps.jsonl + run_meta.json): chép vào baselines/ để "
                         "người nhận so ghép cặp (evaluate --baseline-hyps); bỏ mọi trường đường dẫn máy")
    args = ap.parse_args(argv)
    dst = Path(args.out or P.path("data_root") / "share") / args.name
    if dst.exists():
        raise SystemExit(f"{dst} đã có: bản share không được ghi đè (đổi --name / version)")
    for b in args.benchmarks:  # kiểm trước khi ghi gì: lỗi giữa chừng không để lại gói dở
        mp, bm = manifest_path(b)
        rows = share_rows(read_jsonl(str(mp)))
        if bm.get("evidence_level") == "locked" or bm.get("manifest_sha256"):
            # người nhận chấm bằng manifest share (+ đường dẫn audio): phải trùng bản khoá
            want, got = bm.get("manifest_sha256"), manifest_sha256(rows)
            if got != want:
                raise SystemExit(f"{b}: manifest share {got[:12]} khác bản khoá {str(want)[:12]}")
        if bm.get("catalog_from_bench"):
            raise SystemExit(f"{b}: catalog chứa span lấy từ chính bench, không share làm catalog chấm")
        if bm.get("catalog_sha256"):
            cat = Path(bm["catalog"].replace("${data_root}", str(P.path("data_root"))))
            if sha256_file(cat) != bm["catalog_sha256"]:
                raise SystemExit(f"{b}: catalog {cat} khác bản khoá")
    (dst / "manifests").mkdir(parents=True)
    summary = {}
    for b in args.benchmarks:
        mp, bm = manifest_path(b)
        rows = share_rows(read_jsonl(str(mp)))
        name = b.replace("_draft", "")
        with open(dst / "manifests" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in strip_text(rows):
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        shutil.copyfile(P.REPO / "configs" / "bench" / "benchmarks" / f"{b}.yaml", dst / "manifests" / f"{name}.benchmark.yaml")
        if bm.get("catalog_from_bench"):
            raise SystemExit(f"{b}: catalog chứa span lấy từ chính bench, không share làm catalog chấm")
        if bm.get("catalog"):
            cat = Path(bm["catalog"].replace("${data_root}", str(P.path("data_root"))))
            shutil.copyfile(cat, dst / "manifests" / f"{name}.catalog.tsv")
        by = {}
        for r in rows:
            d = by.setdefault(r["dataset"], {"rows": 0, "hours": 0.0, "license": r.get("license")})
            d["rows"] += 1
            if r.get("duration"):
                d["hours"] = round(d["hours"] + r["duration"] / 3600, 4)
                d["_timed"] = True
        for d in by.values():  # bộ khoá chỉ bằng metadata chưa có duration: "chưa đo" (null), không phải 0 giờ
            if not d.pop("_timed", False):
                d["hours"] = None
        summary[name] = {"rows": len(rows), "evidence_level": bm.get("evidence_level"), "track": bm.get("track"),
                         "by_dataset": by}
    for bdir in args.baselines:
        bdir = Path(bdir)
        bench, system = bdir.parent.name.replace("_draft", ""), bdir.name
        if bench not in summary:
            raise SystemExit(f"{bdir}: benchmark '{bench}' không có trong --benchmarks")
        ids = {json.loads(l)["id"] for l in open(dst / "manifests" / f"{bench}.jsonl", encoding="utf-8") if l.strip()}
        hyps = [json.loads(l) for l in open(bdir / "hyps.jsonl", encoding="utf-8") if l.strip()]
        if {h["id"] for h in hyps} != ids or len(hyps) != len(ids):
            raise SystemExit(f"{bdir}: hyps không khớp đúng một dòng cho mỗi id của {bench}")
        out = dst / "baselines" / bench / system
        out.mkdir(parents=True)
        with open(out / "hyps.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for h in hyps:
                f.write(json.dumps({"id": h["id"], "hyp": h.get("hyp", "")}, ensure_ascii=False) + "\n")
        meta = json.loads((bdir / "run_meta.json").read_text(encoding="utf-8"))
        (out / "run_meta.json").write_text(json.dumps(clean_meta(meta), ensure_ascii=False, indent=1), encoding="utf-8")
        if (bdir / "report.md").exists():
            shutil.copyfile(bdir / "report.md", out / "report.md")
        summary[bench].setdefault("baselines", []).append(system)
    for rel in DOCS + CONFIGS:
        src = P.REPO / rel
        if src.exists():
            sub = dst / ("docs" if rel.startswith("docs/") else "configs")
            sub.mkdir(exist_ok=True)
            shutil.copyfile(src, sub / Path(rel).name)
    print(f"code/: {len(copy_code(dst, args.name.rsplit('-v', 1)[-1]))} module")  # người nhận không cần repo
    (dst / "README.md").write_text(README.replace("{name}", args.name).replace(
        "{summary}", "```json\n" + json.dumps(summary, ensure_ascii=False, indent=1) + "\n```"),
                                   encoding="utf-8")
    (dst / "release.json").write_text(json.dumps({"name": args.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                  "benchmarks": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = []
    for p in sorted(x for x in dst.rglob("*") if x.is_file()):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dst).as_posix()}")
    (dst / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{dst}: {sum(v['rows'] for v in summary.values())} dòng, {len(lines)} file")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


README = (Path(__file__).with_name("share_readme.md")).read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
