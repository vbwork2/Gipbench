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
        if not r.get("upstream_id") or not r.get("audio_sha256"):
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
            d["hours"] = round(d["hours"] + (r.get("duration") or 0) / 3600, 4)
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
