#!/usr/bin/env python3
"""Đăng ký data release từ file đã dựng, theo configs/dataset/releases.yaml (hợp đồng: gipbench.core.release).

  python -m gipbench.dataset.release.register --all            # mọi release trong config chưa có trên đĩa
  python -m gipbench.dataset.release.register bench/v0-draft pool/vss_cs@e3_v1
  python -m gipbench.dataset.release.register --list           # trạng thái từng release
  python -m gipbench.dataset.release.register --verify         # kiểm sha256 mọi release đã có

Không sửa file nguồn. Release đã có thì bỏ qua (bất biến). Kiểm sau khi ghi: số dòng và nội dung mọi trường trừ `audio`
khớp file nguồn, audio trỏ cùng một file.
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

from gipbench.core import release as R
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO

CONFIG = REPO / "configs" / "dataset" / "releases.yaml"


def load_config(path=CONFIG):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["releases"]


def check_same(src_rows, rel):
    got = read_jsonl(str(rel.manifest))
    if len(got) != len(src_rows):
        raise SystemExit(f"{rel.ref}: {len(got)} dòng != nguồn {len(src_rows)}")
    for a, b in zip(src_rows, got):
        ka = {k: v for k, v in a.items() if not k.startswith("_") and k != "audio"}
        kb = {k: v for k, v in b.items() if not k.startswith("_") and k != "audio"}
        if ka != kb or (a.get("_audio_path") and os.path.normpath(a["_audio_path"]) != os.path.normpath(b["_audio_path"])):
            raise SystemExit(f"{rel.ref}: dòng {a.get('id')} khác nguồn sau khi đăng ký")


def register(ref, spec, config_path=CONFIG):
    kw = dict(builder="gipbench.dataset.release.register",
              builder_args={"config": os.path.relpath(config_path, REPO).replace(os.sep, "/"),
                            "built_by": spec.get("built_by", "")},
              catalog_ref=spec.get("catalog") if R.parse(ref)[0] != "catalog" else None,
              flags=spec.get("flags"), notes=spec.get("notes", ""))
    if R.parse(ref)[0] == "catalog":
        src = REPO / spec["catalog"]
        return R.write_release(ref, catalog_src=src, source=src, parents=[src], **kw)
    src = REPO / spec["manifest"]
    rows = read_jsonl(str(src))
    rel = R.write_release(ref, manifest_rows=rows, source=src, parents=[src], **kw)
    check_same(rows, rel)
    return rel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("refs", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.list or args.verify:
        bad = 0
        for ref in cfg:
            try:
                rl = R.resolve(ref, verify=args.verify)
                info = next(iter(rl.meta["files"].values()), {})
                print(f"có      {ref:34s} {info.get('rows', '')} dòng {info.get('hours', '')} h")
            except R.ReleaseError as e:
                missing = "chưa có" in str(e)
                print(f"{'CHƯA CÓ' if missing else 'LỖI    '} {ref:34s} {'' if missing else e}")
                bad += not missing
        sys.exit(1 if bad else 0)
    refs = list(cfg) if args.all else args.refs
    if not refs:
        ap.error("truyền ref hoặc --all")
    refs.sort(key=lambda r: R.parse(r)[0] != "catalog")  # catalog trước: bench / pool tham chiếu catalog
    for ref in refs:
        if ref not in cfg:
            raise SystemExit(f"{ref} không có trong {args.config}")
        if (R.release_dir(ref) / "release.json").exists():
            print(f"bỏ qua {ref}: đã có (bất biến)")
            continue
        spec = cfg[ref]
        src = REPO / (spec.get("manifest") or spec["catalog"])
        if not src.exists():
            print(f"BỎ QUA {ref}: thiếu file nguồn {src}")
            continue
        rl = register(ref, spec, Path(args.config))
        info = next(iter(rl.meta["files"].values()))
        print(f"đã ghi {ref}: {rl.dir} ({info.get('rows')} dòng{', %s h' % info['hours'] if 'hours' in info else ''})")


if __name__ == "__main__":
    main()
