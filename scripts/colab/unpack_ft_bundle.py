#!/usr/bin/env python3
"""Giải nén gói fine-tune (scripts/colab/pack_ft_bundle.py) vào bản clone repo, rồi kiểm từng file theo bundle_index.json.

  python scripts/colab/unpack_ft_bundle.py --bundle /content/drive/MyDrive/gipbench_ft --dest /content/gipbench [--verify sha]

Shard giải nén theo thứ tự tên (bản mới đè bản cũ). Chỉ nhận đường dẫn tương đối nằm trong --dest. Kiểm mặc định theo
cỡ file (nhanh); --verify sha đọc lại toàn bộ (chậm hơn, dùng khi nghi Drive chép hỏng). Thoát 1 nếu có file sai / thiếu.
"""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_members(tf: tarfile.TarFile, dest: Path):
    root = dest.resolve()
    for m in tf.getmembers():
        if not (m.isfile() or m.isdir()):
            raise SystemExit(f"LỖI shard có mục không phải file / thư mục: {m.name}")
        if not (root / m.name).resolve().is_relative_to(root):
            raise SystemExit(f"LỖI đường dẫn ra ngoài --dest: {m.name}")
        yield m


def unpack(bundle: Path, dest: Path, verify: str = "size") -> int:
    idx = json.loads((bundle / "bundle_index.json").read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    for name in sorted(idx["shards"]):
        with tarfile.open(bundle / name) as tf:
            tf.extractall(dest, members=safe_members(tf, dest))
        print(f"  giải nén {name} ({idx['shards'][name]['n_files']} file)", flush=True)
    bad = []
    for rp, info in idx["files"].items():
        p = dest / rp
        if not p.is_file() or p.stat().st_size != info["size"] or (verify == "sha" and sha256_file(p) != info["sha256"]):
            bad.append(rp)
    print(f"kiểm {len(idx['files'])} file ({verify}): {'ĐẠT' if not bad else f'SAI {len(bad)}, vd {bad[:3]}'}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--dest", required=True, help="gốc bản clone repo")
    ap.add_argument("--verify", choices=["size", "sha"], default="size")
    args = ap.parse_args()
    sys.exit(unpack(Path(args.bundle), Path(args.dest), args.verify))


if __name__ == "__main__":
    main()
