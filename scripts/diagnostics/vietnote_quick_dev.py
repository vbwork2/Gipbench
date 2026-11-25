"""Freeze a small stratified development manifest before running the baseline."""

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl


def select(rows, per_slice, seed):
    strata = defaultdict(list)
    for row in rows:
        duration = float(row["duration"])
        band = next((bound for bound in (5, 10, 15, 20, 25, 40) if duration < bound), "40+")
        has_english = "<en>" in row["text_tagged"]
        key = (row.get("dataset"), row.get("domain"), band, has_english)
        strata[key].append(row)
    keep = []
    for group in strata.values():
        order = sorted(group, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        speakers = Counter()
        for _ in range(min(per_slice, len(order))):
            i = min(range(len(order)), key=lambda i: speakers[order[i].get("speaker") or order[i]["id"]])
            row = order.pop(i)
            speakers[row.get("speaker") or row["id"]] += 1
            keep.append(row)
    return sorted(keep, key=lambda row: row["id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-slice", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--diagnostic", action="store_true")
    args = parser.parse_args()
    if R.parse(args.dev)[0] != "dev" or args.per_slice < 1:
        parser.error("A dev release and a positive stratum quota are required")
    release = R.resolve(args.dev)
    flags = release.meta.get("flags") or {}
    if (flags.get("interim") or not flags.get("independent")) and not args.diagnostic:
        parser.error("This development release is eligible only for diagnostic subsampling")
    meta = args.out.with_suffix(".lock.json")
    if args.out.exists() or meta.exists():
        parser.error("The quick dev is frozen. Use a new output path.")
    rows = read_jsonl(str(release.manifest))
    keep = select(rows, args.per_slice, args.seed)
    for row in keep:
        row["audio"] = os.path.relpath(row["_audio_path"], args.out.resolve().parent).replace(os.sep, "/")
    write_jsonl(str(args.out), keep)
    meta.write_text(json.dumps({
        "parent": args.dev, "parent_manifest_sha256": manifest_sha256(rows),
        "manifest_sha256": manifest_sha256(keep), "rows": len(keep), "full_rows": len(rows),
        "seed": args.seed, "per_slice": args.per_slice, "diagnostic": args.diagnostic,
        "failure_pattern_validation": "pending",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Frozen {len(keep)}/{len(rows)} rows: {args.out}")


if __name__ == "__main__":
    main()
