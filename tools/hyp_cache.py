#!/usr/bin/env python3
"""Quản lý cache hyps ASR theo id + hash audio (lib/hyp_cache.py).

  # bổ sung audio_sha256 / audio_rel cho hyps cũ (ghi trước Phase 0d), hash lấy từ file cục bộ hoặc LFS trên HF
  python tools/hyp_cache.py backfill --manifest data/raw/vss_full/vss_cs_pilot.jsonl data/raw/vss_full/vss_cs_20.jsonl \
      --hyps results/asr_hyps/qwen3_asr_0.6b results/asr_hyps/fw_large_v3_int8 \
      --hash-index results/asr_hyps/vss_audio_sha256.tsv --audio-root data/dl/VietSuperSpeech \
      --hf-repo thanhnew2001/VietSuperSpeech

Bản ghi hyps cũ chỉ có id; id được nối với manifest đã dùng lúc giải mã (--manifest) để biết file audio, rồi lấy
hash của file đó. File hyps được ghi đè sau khi sao lưu thành <file>.bak (chỉ khi có thay đổi).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.hyp_cache import HashIndex, audio_rel, file_sha256, iter_hyp_files  # noqa: E402
from lib.manifest import read_jsonl  # noqa: E402


def backfill(args):
    rows = {}
    for m in args.manifest:
        for r in read_jsonl(m):
            rows.setdefault(r["id"], r)
    index = HashIndex(args.hash_index)
    files = [f for p in args.hyps for f in iter_hyp_files(p)]
    need = set()
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            if line.strip():
                h = json.loads(line)
                if not h.get("audio_sha256") and h["id"] in rows:
                    need.add(h["id"])
    rel = {i: audio_rel(rows[i], args.audio_root) for i in need}
    for i in need:
        if index.get(rel[i]) is None and Path(rows[i]["_audio_path"]).exists():
            index.put(rel[i], file_sha256(rows[i]["_audio_path"]))
    if args.hf_repo:
        n_new, no_lfs = index.fill_from_hf(args.hf_repo, (rel[i] for i in need))
        print(f"  chỉ mục hash: +{n_new} mục từ HF, {len(no_lfs)} file không có sha256 LFS")
    index.save()
    tot = {"thêm hash": 0, "đã có hash": 0, "id không có trong manifest": 0, "không lấy được hash": 0}
    for fp in files:
        out, changed = [], False
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            h = json.loads(line)
            if h.get("audio_sha256"):
                tot["đã có hash"] += 1
            elif h["id"] not in rows:
                tot["id không có trong manifest"] += 1
            elif index.get(rel[h["id"]]) is None:
                tot["không lấy được hash"] += 1
            else:
                h["audio_sha256"], h["audio_rel"] = index.get(rel[h["id"]]), rel[h["id"]]
                tot["thêm hash"] += 1
                changed = True
            out.append(json.dumps(h, ensure_ascii=False))
        if changed:
            shutil.copy2(fp, str(fp) + ".bak")
            Path(fp).write_text("\n".join(out) + "\n", encoding="utf-8")
            print(f"  {fp}: đã ghi (bản cũ ở {fp}.bak)")
    print("  " + "; ".join(f"{k}: {v}" for k, v in tot.items()))
    return tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="bổ sung audio_sha256 cho hyps cũ")
    b.add_argument("--manifest", nargs="+", required=True, help="manifest đã dùng lúc giải mã")
    b.add_argument("--hyps", nargs="+", required=True, help="file hoặc thư mục hyps")
    b.add_argument("--hash-index", required=True)
    b.add_argument("--audio-root")
    b.add_argument("--hf-repo", help="lấy sha256 LFS từ HF cho file không có cục bộ")
    args = ap.parse_args()
    if args.cmd == "backfill":
        backfill(args)


if __name__ == "__main__":
    main()
