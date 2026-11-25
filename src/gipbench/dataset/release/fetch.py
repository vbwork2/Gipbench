#!/usr/bin/env python3
"""Tải audio của bench từ NGUỒN GỐC (revision cố định), kiểm sha256, ghi manifest có đường dẫn audio cục bộ.

Bản share của gipbench KHÔNG kèm audio (license của bên phát hành). Người dùng chạy lệnh này để tự dựng lại:

  pip install -e ".[audio]" huggingface_hub pyarrow
  python -m gipbench.dataset.release.fetch --manifest manifests/general_v1.jsonl --out bench_data/general_v1 \
      --cv-root /path/cv-corpus-27.0-2026-09-11/vi            # Common Voice: tải tay từ commonvoice.mozilla.org
  python -m gipbench.dataset.release.fetch --manifest manifests/code_switch_v1.jsonl --out bench_data/code_switch_v1

Nguồn (mọi file nguồn được kiểm sha256 LFS trước khi dùng):
  vivos              AILAB-VNUHCM/vivos @3cbfb25: data/vivos.tar.gz (1.47 GB), tách test/waves/<spk>/<id>.wav, giữ nguyên byte
  ggroup_*           g-group-ai-lab/vi-asr-*-test: data/test-00000-of-00001.parquet (~10 MB / bộ), bytes mp3 gốc
  vimedcss           tensorxt/ViMedCSS @b6959a1: data/hard-0000{0,1}-of-00002.parquet (757 MB), audio 44.1 kHz ->
                     resample soxr HQ 16 kHz mono PCM16 (giống converter của gipbench)
  commonvoice_vi_27  không có trên HF: --cv-root trỏ tới thư mục vi/ của bản 27.0 đã giải nén; copy clips/<file>.mp3
Kiểm: audio giữ nguyên byte (vivos, ggroup, commonvoice) phải trùng `audio_sha256` của manifest, lệch = LỖI (thoát mã 1).
ViMedCSS là audio phái sinh (resample): sha256 phụ thuộc phiên bản soxr / libsndfile, lệch chỉ CẢNH BÁO nếu độ dài khớp
(±0.011 s); điểm số chỉ phụ thuộc id + text nên không đổi.
Ghi: <out>/audio/<dataset>/..., <out>/manifest.jsonl (manifest gốc + trường audio tương đối), <out>/fetch_report.json.
--local-source DATASET=PATH: dùng file nguồn đã có (vẫn kiểm sha256). Nên đặt HF_HUB_DISABLE_XET=1 nếu tải bị treo.
"""
import argparse
import io
import json
import os
import shutil
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.fileio import sha256_file

GG_REV = {"tech": "0b8c87452173239063fe389662c2f034307bedb6", "edu": "9a15ad66d229187c9acc1a5c5ea9811c2e27ddf7",
          "finance": "ebadeb02804de29423668802f797e5f563143da5", "pubadmin": "40aabb05703f30cda255890aaf11f15be21a970f"}
GG_SHA = {"tech": "d4c73b116e7c0888c3aabbe45a36962b395099c82ef9ac5543c8eaaf674fc701",
          "edu": "ceb6401cd5e4e3e7eab1ae50efe2d6e2c598f10ef6e3b0e97289ead968ddd750",
          "finance": "d45f8d32c981118d16c4f1b4b336d4a6ea08a6d4f8c0cb0a6f15757ef709244f",
          "pubadmin": "e483f27e05d096c790689c41255fd56fa992e98e8840c8bda6ce53a2b2ea18e3"}
SOURCES = {
    "vivos": {"kind": "tar", "repo": "AILAB-VNUHCM/vivos", "revision": "3cbfb2502e5e84776b4b778b020a09759f723f52",
              "files": [("data/vivos.tar.gz", "147477f7a7702cbafc2ee3808d1c142989d0dbc8d9fce8e07d5f329d5119e4ca")]},
    **{f"ggroup_{n}": {"kind": "parquet_bytes", "repo": f"g-group-ai-lab/vi-asr-{n}-test", "revision": GG_REV[n],
                       "files": [("data/test-00000-of-00001.parquet", GG_SHA[n])], "id_col": "id"} for n in GG_REV},
    "vimedcss": {"kind": "parquet_resample", "repo": "tensorxt/ViMedCSS",
                 "revision": "b6959a18a08739464733930a872e7125c03e6558",
                 "files": [("data/hard-00000-of-00002.parquet", "82c41d58472c81794465d7002f6c2dd2459cb04c62b02fce13c5bcdcee33c17f"),
                           ("data/hard-00001-of-00002.parquet", "dcba6a4b102dc3e42c7195cb1c1742a976c30bcaccd7e53437d3793e72805e55")],
                 "id_col": "segment_id", "target_sr": 16000},
    "commonvoice_vi_27": {"kind": "local_cv"},
}
DERIVED = {"vimedcss"}  # audio phái sinh: sha256 phụ thuộc phiên bản thư viện


def _evalset_sources():
    """Các bộ test bổ sung (track E0): nguồn khoá trong gipbench.dataset.ingest.evalsets.SPECS."""
    from gipbench.dataset.ingest.evalsets import SPECS as EV
    return {k: {"kind": "evalset", "repo": v["repo"], "revision": v["revision"], "files": v["files"]} for k, v in EV.items()}


SOURCES.update(_evalset_sources())


def is_derived(r) -> bool:
    return r.get("audio_derivation") == "soxr_hq_16k_mono_pcm16" or r["dataset"] in DERIVED


def download(src, cache):
    """Tải các file nguồn theo revision cố định, kiểm sha256 LFS; trả về danh sách đường dẫn cục bộ."""
    from huggingface_hub import hf_hub_download
    out = []
    for fn, want in src["files"]:
        p = hf_hub_download(src["repo"], fn, repo_type="dataset", revision=src["revision"], cache_dir=cache)
        got = sha256_file(p)
        if got != want:
            raise SystemExit(f"{src['repo']}/{fn}: sha256 {got[:12]} khác bản khoá {want[:12]} (file nguồn đã bị đổi?)")
        out.append(p)
    return out


def fetch_parquet(files, rows, out_dir: Path, id_col: str, resample_to=None):
    """{id dòng manifest: đường dẫn} cho các dòng có upstream_id khớp cột id của parquet."""
    import pyarrow.parquet as pq
    want = {str(r["upstream_id"]): r for r in rows}
    got = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            ids = [str(x) for x in pf.read_row_group(rg, columns=[id_col]).column(0).to_pylist()]
            hit = [i for i, x in enumerate(ids) if x in want and want[x]["id"] not in got]
            if not hit:
                continue
            audio = pf.read_row_group(rg, columns=["audio"]).column(0).to_pylist()
            for i in hit:
                r = want[ids[i]]
                data = audio[i]["bytes"]
                if resample_to:
                    import numpy as np
                    import soundfile as sf
                    import soxr
                    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
                    wav = wav.mean(axis=1)
                    if sr != resample_to:
                        wav = soxr.resample(wav, sr, resample_to, quality="HQ")
                    p = out_dir / f"{r['id']}.wav"
                    sf.write(str(p), np.asarray(wav), resample_to, subtype="PCM_16")
                else:
                    ext = Path(audio[i].get("path") or "x.mp3").suffix or ".mp3"
                    p = out_dir / f"{r['id']}{ext}"
                    p.write_bytes(data)
                got[r["id"]] = p
    return got


def fetch_tar(files, rows, out_dir: Path):
    """VIVOS: tách đúng các file test/waves/<spk>/<upstream_id>.wav cần dùng, giữ nguyên byte."""
    want = {f"test/waves/{r['upstream_id'].split('_')[0]}/{r['upstream_id']}.wav": r for r in rows}
    got = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(files[0], "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            key = next((k for k in want if m.name.endswith(k)), None)
            if key is None:
                continue
            r = want[key]
            p = out_dir / f"{r['id']}.wav"
            with tf.extractfile(m) as src, open(p, "wb") as dst:
                shutil.copyfileobj(src, dst)
            got[r["id"]] = p
            if len(got) == len(want):
                break
    return got


def fetch_evalset(key, files, rows, out_dir: Path):
    """Bộ E0: tách đúng các upstream_id cần bằng cùng code với lúc nhập (giữ byte hoặc resample 16 kHz)."""
    from gipbench.dataset.ingest.evalsets import TEXT_FIELDS, materialize
    by_up = {str(r["upstream_id"]): r for r in rows}
    got = {}
    for row, why in materialize(key, files, out_dir, want=set(by_up)):
        if why or "_path" not in row:
            continue
        r = by_up[row["upstream_id"]]
        # bản share không có lời thoại (cờ text_redistributable: false, hoặc share_policy.text_not_redistributable):
        # lấy lại đúng giá trị từ nguồn. Chỉ dòng đã bị bỏ lời thoại (không còn text_tagged); dòng có lời thoại giữ nguyên.
        if r.get("text_redistributable") is False or "text_tagged" not in r:
            for k in TEXT_FIELDS:
                if k in row and k not in r:
                    r[k] = row[k]
        src = Path(row["_path"])
        dst = out_dir / f"{r['id']}{src.suffix}"
        if src != dst:
            src.replace(dst)
        got[r["id"]] = dst
    return got


def fetch_cv(cv_root, rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    got = {}
    for r in rows:
        s = Path(cv_root) / "clips" / r["upstream_id"]
        if s.exists():
            p = out_dir / f"{r['id']}{s.suffix}"
            shutil.copyfile(s, p)
            got[r["id"]] = p
    return got


def duration(p) -> float:
    import soundfile as sf
    i = sf.info(str(p))
    return i.frames / i.samplerate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="manifest bản share (không có trường audio)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv-root", help="thư mục vi/ của Common Voice 27.0 đã giải nén (có clips/)")
    ap.add_argument("--datasets", nargs="*", help="chỉ tải các bộ này (mặc định: mọi bộ trong manifest)")
    ap.add_argument("--cache", help="cache_dir của huggingface_hub")
    ap.add_argument("--local-source", nargs="*", default=[], metavar="DATASET=PATH",
                    help="dùng file nguồn đã có thay vì tải (vẫn kiểm sha256), vd ggroup_tech=data/dl/ggroup/tech.parquet")
    args = ap.parse_args(argv)
    rows = [json.loads(l) for l in open(args.manifest, encoding="utf-8") if l.strip()]
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    todo = [d for d in by if not args.datasets or d in args.datasets]
    local = defaultdict(list)
    for x in args.local_source:
        k, v = x.split("=", 1)
        local[k].append(v)
    out = Path(args.out)
    got, status = {}, {}
    for d in todo:
        src = SOURCES.get(d)
        if src is None:
            status[d] = "không có nguồn tải (bộ riêng tư hoặc chưa hỗ trợ)"
            continue
        adir = out / "audio" / d
        if src["kind"] == "local_cv":
            if not args.cv_root:
                status[d] = "bỏ qua: cần --cv-root (Common Voice tải tay)"
                continue
            g = fetch_cv(args.cv_root, by[d], adir)
        else:
            if d in local:
                files = local[d]
                if len(files) != len(src["files"]):
                    raise SystemExit(f"{d}: cần {len(src['files'])} file nguồn, nhận {len(files)}")
                for (fn, want), f in zip(src["files"], files):
                    if sha256_file(f) != want:
                        raise SystemExit(f"{f}: sha256 khác bản khoá của {src['repo']}/{fn}")
            else:
                files = download(src, args.cache)
            if src["kind"] == "tar":
                g = fetch_tar(files, by[d], adir)
            elif src["kind"] == "evalset":
                g = fetch_evalset(d, files, by[d], adir)
            else:
                g = fetch_parquet(files, by[d], adir, src["id_col"],
                                  src.get("target_sr") if src["kind"] == "parquet_resample" else None)
        status[d] = "ok"
        got.update(g)
        print(f"{d}: {len(g)} / {len(by[d])} file", flush=True)
    problems, warn, kept, no_sha = Counter(), [], [], 0
    for r in rows:
        if r["dataset"] not in todo or r["dataset"] not in status or status[r["dataset"]] != "ok":
            continue
        p = got.get(r["id"])
        if p is None:
            problems["thiếu file"] += 1
            continue
        if not r.get("audio_sha256"):  # dòng metadata-only: toàn vẹn đã kiểm bằng sha256 file nguồn lúc tải
            no_sha += 1
        elif sha256_file(p) != r["audio_sha256"]:
            if is_derived(r) and abs(duration(p) - float(r.get("duration") or 0)) <= 0.011:
                warn.append(r["id"])
            else:
                problems["sha256 lệch"] += 1
                continue
        kept.append(dict(r, audio=os.path.relpath(p, out).replace(os.sep, "/")))
    out.mkdir(parents=True, exist_ok=True)
    # chạy từng bộ vào cùng --out: giữ dòng của bộ KHÔNG tải lần này nếu file audio của nó vẫn còn
    prev = out / "manifest.jsonl"
    done_ids = {r["id"] for r in kept}
    if prev.exists():
        for line in open(prev, encoding="utf-8"):
            r = json.loads(line) if line.strip() else None
            if r and r["id"] not in done_ids and r["dataset"] not in status and (out / r["audio"]).exists():
                kept.append(r)
    order = {r["id"]: i for i, r in enumerate(rows)}
    kept.sort(key=lambda r: order.get(r["id"], len(order)))
    with open(prev, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"manifest": args.manifest, "datasets": status,
           "rows_requested": sum(len(by[d]) for d in todo if status.get(d) == "ok"), "rows_ok": len(kept),
           "problems": dict(problems), "derived_sha_mismatch_duration_ok": len(warn), "derived_examples": warn[:5],
           "rows_checked_by_source_sha_only": no_sha}
    (out / "fetch_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if warn:
        print(f"CẢNH BÁO: {len(warn)} file audio phái sinh khác sha256 (resample khác phiên bản thư viện) nhưng độ dài khớp")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
