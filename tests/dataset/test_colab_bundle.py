#!/usr/bin/env python3
"""Gói fine-tune cho Colab (scripts/colab/pack_ft_bundle.py + unpack_ft_bundle.py): khứ hồi giữ đường dẫn tương đối,
chia shard theo cỡ, chạy lại chỉ gói file mới / đổi, kiểm phát hiện file hỏng, từ chối file ngoài gốc. Dữ liệu GIẢ.

  python tests/dataset/test_colab_bundle.py
"""
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "colab"))
from pack_ft_bundle import pack, rel_to  # noqa: E402
from unpack_ft_bundle import unpack  # noqa: E402

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-300:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    t = Path(tempfile.mkdtemp())
    root, out, dest = t / "repo", t / "bundle", t / "colab"
    files = {"data/releases/pool/x@v2/manifest.jsonl": b'{"audio": "../../../dl/a.wav"}\n',
             "data/dl/a.wav": b"A" * 600, "data/dl/b.wav": b"B" * 600, "models/m/model.pt": b"M" * 300}
    for rp, b in files.items():
        (root / rp).parent.mkdir(parents=True, exist_ok=True)
        (root / rp).write_bytes(b)
    items = [(root / rp, rel_to(root, root / rp)) for rp in files]
    plan = pack(items, out, shard_bytes=1000)
    check("pack: chia shard theo cỡ (1000 B -> >= 2 shard)", len(plan["shards"]) >= 2, plan)
    idx = json.loads((out / "bundle_index.json").read_text(encoding="utf-8"))
    first = sorted(idx["shards"])[0]
    check("pack: metadata release vào shard đầu",
          idx["files"]["data/releases/pool/x@v2/manifest.jsonl"]["shard"] == first, idx["files"])
    check("unpack: khứ hồi đúng nội dung + đường dẫn tương đối", unpack(out, dest) == 0
          and all((dest / rp).read_bytes() == b for rp, b in files.items()))
    plan2 = pack(items, out, shard_bytes=1000)
    check("pack lần 2 không đổi gì: không shard mới", plan2["shards"] == [] and plan2["skipped"] == len(items), plan2)
    (root / "data/dl/b.wav").write_bytes(b"C" * 600)
    (root / "data/dl/new.wav").write_bytes(b"N" * 10)
    items3 = items + [(root / "data/dl/new.wav", "data/dl/new.wav")]
    plan3 = pack(items3, out, shard_bytes=1000)
    check("pack lần 3: chỉ gói file đổi + file mới, shard đánh số tiếp", plan3["new_files"] == 2
          and plan3["shards"] and plan3["shards"][0] not in idx["shards"], plan3)
    dest2 = t / "colab2"
    check("unpack: shard mới đè bản cũ", unpack(out, dest2) == 0 and (dest2 / "data/dl/b.wav").read_bytes() == b"C" * 600)
    (dest2 / "data/dl/a.wav").write_bytes(b"A" * 599)
    idx3 = json.loads((out / "bundle_index.json").read_text(encoding="utf-8"))
    bad = [rp for rp, i in idx3["files"].items() if (dest2 / rp).stat().st_size != i["size"]]
    check("kiểm cỡ phát hiện file hỏng", bad == ["data/dl/a.wav"], bad)
    try:
        rel_to(root, t / "outside.wav")
        check("rel_to: từ chối file ngoài gốc repo", False)
    except SystemExit:
        check("rel_to: từ chối file ngoài gốc repo", True)
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'ĐẠT tất cả'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
