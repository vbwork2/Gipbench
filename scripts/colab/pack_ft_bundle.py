#!/usr/bin/env python3
"""Đóng gói dữ liệu + model gốc cho fine-tune trên Colab (code lấy bằng git clone, KHÔNG nằm trong gói).

Gói giữ nguyên đường dẫn tương đối theo gốc repo (data/releases/..., data/dl/..., data/raw/..., models/gip15/...), nên
giải nén vào bản clone trên Colab thì đường dẫn audio tương đối trong manifest release vẫn đúng. Chia thành các file
tar ~1 GB (Drive / Colab chép file lớn nhanh hơn nhiều so với hàng vạn file wav nhỏ) + bundle_index.json (cỡ, sha256,
shard của từng file).

Chạy lại được: file đã có trong index với cùng sha256 thì bỏ qua; file mới / đổi nội dung (vd release đăng ký lại,
dev_v2, catalog/train-v1) vào shard mới đánh số tiếp. Giải nén theo thứ tự shard nên bản mới đè bản cũ.

  python scripts/colab/pack_ft_bundle.py --out D:/gipbench_ft_bundle                    # mặc định: xem DEFAULT_REFS
  python scripts/colab/pack_ft_bundle.py --out <dir> --refs dev/dev_v2_cs@v1 catalog/train-v1   # thêm sau
  python scripts/colab/pack_ft_bundle.py --out <dir> --dry-run

Trên Colab: python scripts/colab/unpack_ft_bundle.py --bundle /content/drive/MyDrive/gipbench_ft --dest /content/gipbench

Không đóng gói audio của bench (chỉ manifest text: cổng rò rỉ của model.run cần text / nguồn / id của bench khoá).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[2]
INDEX = "bundle_index.json"

# pool train sạch với mọi bench khoá (TODO 2c) + nhãn giả / chưng cất đang dùng; dev cũ chỉ để chẩn đoán (không độc lập)
DEFAULT_REFS = ["pool/vimedcss_train@v2", "pool/vivos_train@v2", "pool/confusable_train@v2", "pool/cv_other@v2",
                "pool/vss_cs@e3_v1", "pool/distill@e3_v1"]
DEFAULT_MODEL_FILES = ["models/gip15/model.pt", "models/gip15/bpe.model", "models/gip15/tokens.txt",
                       "models/gip15/config.json", "models/gip15/README.md",
                       # text bộ test riêng tư (consensus): core.leak chặn khỏi train / dev (paths.yaml leak_block_files)
                       "data/raw/consensus/leak_block.txt"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel_to(root: Path, p: Path) -> str:
    rp = os.path.relpath(os.path.abspath(p), os.path.abspath(root))
    if rp.startswith(".."):
        raise SystemExit(f"LỖI {p} nằm ngoài gốc repo {root}: không giữ được đường dẫn tương đối trong gói")
    return rp.replace(os.sep, "/")


def release_files(ref: str) -> Tuple[List[Path], List[Path]]:
    """(file của thư mục release, file audio của manifest). Bench / catalog: chỉ file release, không audio."""
    sys.path.insert(0, str(REPO / "src"))
    from gipbench.core import release as R
    from gipbench.core.manifest import read_jsonl
    rl = R.resolve(ref)  # kiểm sha256 file release
    files = sorted(p for p in rl.dir.iterdir() if p.is_file())
    audio = []
    if R.parse(ref)[0] in ("pool", "dev") and rl.manifest is not None:
        miss = []
        for r in read_jsonl(str(rl.manifest)):
            p = r.get("_audio_path")
            if p:
                (audio if os.path.exists(p) else miss).append(Path(p))
        if miss:
            raise SystemExit(f"LỖI {ref}: {len(miss)} file audio không tồn tại (vd {miss[:2]})")
    return files, audio


def all_bench_catalog() -> List[str]:
    sys.path.insert(0, str(REPO / "src"))
    from gipbench.core import release as R
    out = []
    for kind in ("bench", "catalog"):
        d = R.root() / kind
        if d.is_dir():
            out += [f"{kind}/{p.name}" for p in sorted(d.iterdir()) if (p / "release.json").is_file()]
    return out


def collect(refs: Iterable[str], model_files: Iterable[str], root: Path = REPO) -> List[Tuple[Path, str]]:
    seen, out = set(), []
    for ref in refs:
        files, audio = release_files(ref)
        for p in files + audio:
            rp = rel_to(root, p)
            if rp not in seen:
                seen.add(rp)
                out.append((Path(p), rp))
    for m in model_files:
        p = root / m
        if not p.is_file():
            raise SystemExit(f"LỖI thiếu {p}")
        if m not in seen:
            seen.add(m)
            out.append((p, m))
    return out


def load_index(out: Path) -> Dict:
    f = out / INDEX
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"files": {}, "shards": {}, "history": []}


def pack(items: List[Tuple[Path, str]], out: Path, shard_bytes: int, prefix: str = "gipbench_ft",
         note: str = "", dry: bool = False) -> Dict:
    """items: (đường dẫn thật, đường dẫn trong gói). Ghi shard mới cho file chưa có / đổi sha256; cập nhật index."""
    out.mkdir(parents=True, exist_ok=True)
    idx = load_index(out)
    todo = []
    for p, rp in items:
        size = p.stat().st_size
        old = idx["files"].get(rp)
        if old and old["size"] == size and old["sha256"] == sha256_file(p):
            continue
        todo.append((p, rp, size))
    # metadata trước (nhỏ, cần sớm nhất), rồi audio / model theo đường dẫn để các shard ổn định giữa các lần chạy
    todo.sort(key=lambda t: (not t[1].startswith("data/releases/"), t[1]))
    n0 = len(idx["shards"])
    shards, cur, cur_size = [], [], 0
    for t in todo:
        if cur and cur_size + t[2] > shard_bytes:
            shards.append(cur)
            cur, cur_size = [], 0
        cur.append(t)
        cur_size += t[2]
    if cur:
        shards.append(cur)
    plan = {"new_files": len(todo), "new_bytes": sum(t[2] for t in todo), "skipped": len(items) - len(todo),
            "shards": [f"{prefix}-{n0 + i:03d}.tar" for i in range(len(shards))]}
    if dry:
        return plan
    for i, group in enumerate(shards):
        name = f"{prefix}-{n0 + i:03d}.tar"
        tmp = out / (name + ".part")
        with tarfile.open(tmp, "w", format=tarfile.PAX_FORMAT) as tf:
            for p, rp, size in group:
                tf.add(str(p), arcname=rp, recursive=False)
                idx["files"][rp] = {"size": size, "sha256": sha256_file(p), "shard": name}
        tmp.replace(out / name)
        idx["shards"][name] = {"size": (out / name).stat().st_size, "sha256": sha256_file(out / name),
                               "n_files": len(group), "created": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        print(f"  {name}: {len(group)} file, {(out / name).stat().st_size / 1e9:.2f} GB", flush=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    idx["history"].append({"at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "git_commit": commit,
                           "note": note, **plan})
    (out / INDEX).write_text(json.dumps(idx, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return plan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="thư mục gói (vd thư mục đồng bộ Google Drive)")
    ap.add_argument("--refs", nargs="*", help=f"release cần gói (mặc định {' '.join(DEFAULT_REFS)})")
    ap.add_argument("--no-bench-catalog", action="store_true", help="không kèm mọi release bench/* catalog/* (manifest text)")
    ap.add_argument("--model-files", nargs="*", default=DEFAULT_MODEL_FILES)
    ap.add_argument("--shard-gb", type=float, default=1.0)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    refs = list(args.refs if args.refs is not None else DEFAULT_REFS)
    if not args.no_bench_catalog:
        refs += [r for r in all_bench_catalog() if r not in refs]
    items = collect(refs, args.model_files)
    plan = pack(items, Path(args.out), int(args.shard_gb * 1e9), note=args.note, dry=args.dry_run)
    print(json.dumps({"refs": refs, **plan}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
