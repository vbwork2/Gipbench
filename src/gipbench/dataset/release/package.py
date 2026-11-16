#!/usr/bin/env python3
"""Đóng gói bản share của bench: manifest KHÔNG kèm audio + datasheet + config chấm + SHA256SUMS.

  python -m gipbench.dataset.release.package --benchmarks general_v1_draft code_switch_v1_draft \
      --name gipbench-bench-v1.0-draft [--out <data_root>/share]

Mỗi manifest share = manifest của benchmark, bỏ trường `audio` (đường dẫn máy) và mọi trường bắt đầu bằng "_", thêm
`audio_derivation` (upstream_bytes | soxr_hq_16k_mono_pcm16) để fetch biết cách kiểm. Từ chối đóng gói nếu có dòng
license test_only / nguồn riêng tư, hoặc nguồn không có trong gipbench.dataset.release.fetch.SOURCES.
Người nhận: python -m gipbench.dataset.release.fetch --manifest manifests/<x>.jsonl --out ... (xem README trong gói).
"""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.release.fetch import DERIVED, SOURCES

PRIVATE_LICENSES = {"test_only"}
DOCS = ["docs/datasheet_bench_v1.md", "docs/benchmark_protocol_v1.md", "docs/evaluation_metrics_v1.md"]
CONFIGS = ["configs/bench/datasets.yaml"]


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
        if not r.get("upstream_id") or not r.get("audio_sha256"):
            raise SystemExit(f"{r['id']}: thiếu upstream_id / audio_sha256")
        s = {k: v for k, v in r.items() if k != "audio" and not k.startswith("_")}
        s["audio_derivation"] = "soxr_hq_16k_mono_pcm16" if r["dataset"] in DERIVED else "upstream_bytes"
        out.append(s)
    return sorted(out, key=lambda x: x["id"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", help="mặc định: <data_root>/share")
    args = ap.parse_args(argv)
    dst = Path(args.out or P.path("data_root") / "share") / args.name
    if dst.exists():
        raise SystemExit(f"{dst} đã có: bản share không được ghi đè (đổi --name / version)")
    (dst / "manifests").mkdir(parents=True)
    summary = {}
    for b in args.benchmarks:
        mp, bm = manifest_path(b)
        rows = share_rows(read_jsonl(str(mp)))
        name = b.replace("_draft", "")
        with open(dst / "manifests" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
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
            d["hours"] = round(d["hours"] + (r.get("duration") or 0) / 3600, 4)
        summary[name] = {"rows": len(rows), "evidence_level": bm.get("evidence_level"), "track": bm.get("track"),
                         "by_dataset": by}
    for rel in DOCS + CONFIGS:
        src = P.REPO / rel
        if src.exists():
            sub = dst / ("docs" if rel.startswith("docs/") else "configs")
            sub.mkdir(exist_ok=True)
            shutil.copyfile(src, sub / Path(rel).name)
    (dst / "README.md").write_text(README.format(name=args.name, summary=json.dumps(summary, ensure_ascii=False, indent=1)),
                                   encoding="utf-8")
    (dst / "release.json").write_text(json.dumps({"name": args.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                  "benchmarks": summary}, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = []
    for p in sorted(x for x in dst.rglob("*") if x.is_file()):
        lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dst).as_posix()}")
    (dst / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{dst}: {sum(v['rows'] for v in summary.values())} dòng, {len(lines)} file")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


README = """# {name}

Bench ASR tiếng Việt và Việt – Anh code-switching của gipbench. Gói này **không chứa audio**: chạy `fetch` để tải từ nguồn gốc
(revision cố định, kiểm sha256). Đọc `docs/datasheet_bench_v1.md` trước khi dùng (license, giới hạn, trùng dữ liệu train).

## Cài và tải audio

```bash
git clone <repo gipbench> && cd gipbench && pip install -e ".[audio]" huggingface_hub pyarrow pyyaml
export HF_HUB_DISABLE_XET=1     # nếu tải bị treo
python -m gipbench.dataset.release.fetch --manifest <gói>/manifests/general_v1.jsonl --out bench_data/general_v1 \\
    --cv-root /path/cv-corpus-27.0-2026-09-11/vi     # Common Voice vi 27.0: tải tay tại commonvoice.mozilla.org
python -m gipbench.dataset.release.fetch --manifest <gói>/manifests/code_switch_v1.jsonl --out bench_data/code_switch_v1
```

## Chấm model của bạn

1. Giải mã mọi câu trong `bench_data/<x>/manifest.jsonl`, ghi `hyps.jsonl`: mỗi dòng `{{"id": ..., "hyp": ...}}`, đúng một
   dòng mỗi id. Khai báo cách giải mã trong `run_meta.json` cạnh hyps (ít nhất `"inference_mode"`: open_vocab | lm |
   context_bias | oracle_bias), hoặc dùng `python -m gipbench.bench.transcribe` với adapter có sẵn.
2. Chấm:
```bash
python -m gipbench.bench.evaluate --manifest bench_data/general_v1/manifest.jsonl --hyps my_system/hyps.jsonl \\
    --profile gipbench_strict_v1 --mode final --allow-draft
```
Code-switching (CS-WER, term recall): thêm `--catalog <gói>/manifests/code_switch_v1.catalog.tsv`.
`--mode final` từ chối hyps thiếu / thừa / trùng id. Báo kết quả theo TỪNG bộ dữ liệu, kèm hồ sơ chuẩn hoá và scorer.

## Nội dung

{summary}

`SHA256SUMS`: sha256 mọi file trong gói.
"""

if __name__ == "__main__":
    main()
