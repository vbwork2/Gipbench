#!/usr/bin/env python3
"""Dựng bench PHIÊN BẢN MỚI v1.1 từ bản khoá v1.0 (bản khoá giữ nguyên; audit 17/11/2026, IP-07, R04 / R12 / R20).

  python -m gipbench.dataset.release.derive_v1_1 --which code_switch   # -> data/benchmarks/code_switch_v1_1/
  python -m gipbench.dataset.release.derive_v1_1 --which external      # -> data/benchmarks/external_v1_1/

code_switch : cùng 658 câu C0. Thêm `train_video_overlap` (bool): câu đến từ video YouTube có trong ViMedCSS train
              (so bằng core.decontam.source_key, không so chuỗi). Báo cáo nhóm theo trường này: lát
              `train_video_overlap:False` (video chưa thấy khi train) là lát duy nhất đo được tổng quát cho model đã
              train trên ViMedCSS train.
external    : cùng 32 203 câu E0. Thêm cho mọi câu đã có audio tải về (gipbench.dataset.ingest.evalsets):
              `audio` (đường dẫn tương đối), `audio_sha256`, `duration` thật (v1.0: 26 252 câu chỉ khoá bằng metadata).
              Câu đã có sha256 trong v1.0 phải KHỚP sha256 lúc tải (khác thì dừng). ViMedCSS test: `speaker` = `yt_<ID>`,
              `source` = `youtube:<ID>` (cùng định dạng ViMedCSS train / C0; giá trị gốc giữ ở `speaker_upstream`), thêm
              `train_video_overlap`. Câu chưa tải xong audio: giữ nguyên, liệt kê trong build_meta.json (`missing_audio`);
              benchmark khi đó chỉ là draft (khoá cần đủ sha256 mọi câu: `require_audio_sha`).
Không đổi lời thoại, id, thứ tự câu. Ghi manifest.jsonl + build_meta.json (manifest_sha256, sha256 của bản gốc và của
pool train dùng để đánh dấu, số câu mỗi loại).
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.decontam import source_key
from gipbench.core.fileio import sha256_file as file_sha
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.textnorm import plain_text

_YT_ID = re.compile(r"^yt:(.+)$")


def train_videos(pool: Path):
    return {k for r in read_jsonl(str(pool)) for k in [source_key(r)] if k and k.startswith("yt:")}


def ingest_rows(dl: Path, ds: str):
    """Dòng do evalsets ghi cho một bộ: rows.jsonl khi đã tải xong; đang tải thì gộp các .part*.jsonl ({"rows": [...]})."""
    d = dl / ds
    if (d / "rows.jsonl").exists():
        return {r["id"]: r for r in read_jsonl(str(d / "rows.jsonl"))}, True
    out = {}
    for p in sorted(d.glob(".part*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                for r in json.loads(line).get("rows", []):
                    out[r["id"]] = r
    return out, False


def clean(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def derive_code_switch(src: Path, pool: Path):
    rows = read_jsonl(str(src))
    vids = train_videos(pool)
    out = [dict(clean(r), train_video_overlap=bool(source_key(r) in vids), bench_version="code-switch-v1.1") for r in rows]
    unseen = [r for r in out if not r["train_video_overlap"]]
    return out, {"stats": {"rows": len(out), "train_video_overlap": len(out) - len(unseen), "video_unseen_rows": len(unseen),
                           "video_unseen_videos": len({source_key(r) for r in unseen}),
                           "overlap_videos": len({source_key(r) for r in out if r["train_video_overlap"]})}}


def derive_external(src: Path, pool: Path, dl: Path):
    rows = read_jsonl(str(src))
    vids = train_videos(pool)
    cache, complete = {}, {}
    stats = Counter()
    missing = defaultdict(list)
    text_mismatch = []
    out = []
    for r in rows:
        ds = r["dataset"]
        if ds not in cache:
            cache[ds], complete[ds] = ingest_rows(dl, ds)
        g = cache[ds].get(r["id"])
        n = dict(clean(r), bench_version="external-v1.1")
        apath = dl / ds / g["audio"] if g and g.get("audio") else None
        if apath is not None and apath.exists():
            sha = g.get("audio_sha256") or file_sha(apath)
            if r.get("audio_sha256") and r["audio_sha256"] != sha:
                raise SystemExit(f"{r['id']}: audio_sha256 lúc tải ({sha[:12]}) khác bản khoá v1.0 ({r['audio_sha256'][:12]})")
            if plain_text(r.get("text_tagged") or "").strip() != plain_text(g.get("text_tagged") or "").strip():
                text_mismatch.append(r["id"])
            # manifest ở <data>/benchmarks/<tên>/ nên "../.." = <data>; audio ở <data>/dl/evalsets/<bộ>/audio/...
            n["audio"] = str(Path("..") / ".." / apath.relative_to(dl.parent.parent)).replace("\\", "/")
            n["audio_sha256"] = sha
            if not r.get("duration") and g.get("duration"):
                n["duration"] = g["duration"]
            stats["sha_verified" if r.get("audio_sha256") else "sha_added"] += 1
        else:
            missing[ds].append(r["id"])
        if ds == "vimedcss_test":
            k = source_key(r)
            if k and k.startswith("yt:"):
                vid = _YT_ID.match(k).group(1)
                n["speaker_upstream"], n["source_upstream"] = r.get("speaker"), r.get("source")
                n["speaker"], n["source"] = f"yt_{vid}", f"youtube:{vid}"
            n["train_video_overlap"] = bool(k in vids)
            stats["vimedcss_test_train_video_overlap"] += n["train_video_overlap"]
        out.append(n)
    stats["rows"] = len(out)
    stats["missing_audio"] = sum(len(v) for v in missing.values())
    if text_mismatch:
        print(f"CẢNH BÁO {len(text_mismatch)} câu có lời thoại (bỏ tag) khác dòng tải về; xem build_meta.json")
    return out, {"stats": dict(stats), "missing_audio": {k: len(v) for k, v in missing.items()},
                 "missing_audio_examples": {k: v[:3] for k, v in missing.items()}, "ingest_complete": complete,
                 "text_mismatch_vs_ingest": len(text_mismatch), "text_mismatch_examples": text_mismatch[:5]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["code_switch", "external"], required=True)
    ap.add_argument("--train-pool", help="pool train để đánh dấu trùng video (mặc định ViMedCSS train@full)")
    ap.add_argument("--out", help="thư mục ra (mặc định <data_root>/benchmarks/<which>_v1_1)")
    args = ap.parse_args(argv)
    data = P.path("data_root")
    src = data / "benchmarks" / f"{args.which}_v1" / "manifest.jsonl"
    pool = Path(args.train_pool) if args.train_pool else data / "releases" / "pool" / "vimedcss_train@full" / "manifest.jsonl"
    out_dir = Path(args.out) if args.out else data / "benchmarks" / f"{args.which}_v1_1"
    t = time.time()
    if args.which == "code_switch":
        rows, meta = derive_code_switch(src, pool)
    else:
        rows, meta = derive_external(src, pool, data / "dl" / "evalsets")
    write_jsonl(str(out_dir / "manifest.jsonl"), rows)
    meta.update({"source_manifest": str(src), "source_manifest_sha256": manifest_sha256(read_jsonl(str(src))),
                 "train_pool": str(pool), "train_pool_file_sha256": file_sha(pool),
                 "manifest_sha256": manifest_sha256(rows), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "sec": round(time.time() - t, 1)})
    (out_dir / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("stats", "manifest_sha256", "source_manifest_sha256")}
                     | ({"missing_audio": meta["missing_audio"]} if "missing_audio" in meta else {}), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
