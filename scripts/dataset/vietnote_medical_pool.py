"""Derive the research-only mixed-language pool from the audited ViMedCSS release."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.decontam import dup_key, read_lines, row_text
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.release.clean_pool import clean, load_refs, verify, write_pool
from gipbench.dataset.release.holdout_dev import eligible

PARENT = "pool/vimedcss_train@v4"
RESERVED = "configs/dataset/reserved_terms_v1.txt"
DEV = ("dev/dev_v2_vi@1", "dev/dev_interim_cs@1", "dev/dev_interim_vi@1")


def reserved_index(path):
    index = defaultdict(list)
    for term in read_lines(path):
        key = tuple(dup_key(term).split())
        if key:
            index[key[0]].append(key)
    return index


def contains_reserved(text, index):
    tokens = dup_key(text).split()
    return any(tuple(tokens[i:i + len(key)]) == key for i, token in enumerate(tokens)
               for key in index.get(token, ()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/work/vietnote/medical_cs/manifest.jsonl"))
    args = parser.parse_args()
    parent = R.resolve(PARENT)
    rows = read_jsonl(str(parent.manifest))
    refs = load_refs(R.locked_bench_refs(), DEV)
    index = reserved_index(RESERVED)
    if not index:
        parser.error("The reserved term list is empty")
    selected, rejected, hashes = [], [], set()
    for source in rows:
        if not eligible(source, "cs"):
            continue
        row = dict(source, license="CC-BY-4.0", research_only=True, label_source="human:vimedcss",
                   text_orig=source.get("text_tagged_orig") or source["text_tagged"],
                   speech_type="medical_video", language_category="mixed",
                   quality_status="upstream_labeled_and_inherited_audio_qc", parent_release=PARENT)
        digest = row.get("audio_sha256")
        if not digest or not Path(row["_audio_path"]).is_file():
            rejected.append(dict(row, drop_reason="missing_verified_audio"))
        elif digest in hashes:
            rejected.append(dict(row, drop_reason="duplicate_audio_sha256"))
        elif contains_reserved(row_text(row), index):
            rejected.append(dict(row, drop_reason="reserved_term"))
        else:
            selected.append(row)
            hashes.add(digest)
    keep, drop, stats = clean(selected, refs, near_threshold=.6)
    checks = verify(keep, refs, near_threshold=.6)
    if not keep or any(test["critical"] for test in checks.values()):
        parser.error("The derived pool is empty or overlaps a held-out set")
    write_pool(args.out, keep, rejected + drop)
    report = {"parent": PARENT, "parent_sha256": parent.meta["files"]["manifest.jsonl"]["sha256"],
              "reserved_terms_sha256": R.sha256(RESERVED), "parent_rows": len(rows),
              "before_overlap_filter": len(selected), "rows_in_pool": len(keep),
              "stats": stats, "overlap_checks": checks}
    args.out.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{len(keep)} mixed-language rows, {stats['h_keep']} h: {args.out}")


if __name__ == "__main__":
    main()
