"""Audit pinned acquisition candidates before registering a supplementary pool."""

import argparse
import json
import os
import posixpath
from collections import Counter
from pathlib import Path, PureWindowsPath

import yaml

from gipbench.core import release as R
from gipbench.core.audiofp import fingerprint_rows, load_cache, save_cache
from gipbench.core.decontam import clip_key, dup_key, read_lines, row_text
from gipbench.core.fileio import hours
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO, path
from gipbench.dataset.ingest.vietnote import CONFIG, write_review_queue
from gipbench.dataset.release import clean_pool as clean
from gipbench.dataset.release.acoustic_audit import audit_acoustic, DEFAULT_CACHE
from gipbench.dataset.report.vietnote import CORPUS_POOLS

POOLS = {"fleurs_vi": "vietnote_fleurs_vi", "librispeech": "vietnote_librispeech_en"}


def unique_candidates(rows, train):
    seen = {key for row in train for key in audio_keys(row)}
    accepted, dropped = [], []
    for row in rows:
        keys = audio_keys(row)
        if seen.intersection(keys):
            dropped.append(dict(row, drop_reason="duplicate_training_audio"))
        else:
            seen.update(keys)
            accepted.append(row)
    return accepted, dropped


def audio_keys(row):
    return {("hash", digest) for field in ("audio_sha256", "upstream_audio_sha256")
            if (digest := row.get(field))} | {("clip", key) for key in (clip_key(row),) if key}


def local_cache(cache, rows):
    normalized = {posixpath.normpath(key): value for key, value in cache.items()}
    aliases = {}
    for row in rows:
        raw = row.get("_audio_path")
        if not raw or raw in cache:
            continue
        local = Path(os.path.abspath(raw))
        old = local.as_posix()
        if os.name == "nt" and local.drive:
            windows = PureWindowsPath(local)
            old = f"/mnt/{windows.drive[0].lower()}/" + "/".join(windows.parts[1:])
        if old in normalized:
            aliases[raw] = normalized[old]
    return aliases


def audit(source, work, workers):
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["sources"][source]
    manifest = work / "candidates.jsonl"
    rows = read_jsonl(manifest)
    if not rows:
        raise ValueError("The candidate manifest is empty")
    for row in rows:
        if (row.get("upstream_repo") != spec["repo"] or row.get("upstream_revision") != spec["revision"]
                or row.get("upstream_split") != spec["split"] or row.get("license") != spec["license"]
                or R.sha256(row["_audio_path"]) != row.get("audio_sha256")):
            raise ValueError(f"Invalid provenance or audio digest: {row['id']}")

    eval_refs = R.locked_bench_refs()
    dev_refs = sorted(f"dev/{folder.name}" for folder in (R.root() / "dev").iterdir()
                      if (folder / "release.json").is_file())
    refs = clean.load_refs(eval_refs, dev_refs)
    audio_stats = clean.hash_refs(refs)
    pool = POOLS[source]
    train_refs = set(CORPUS_POOLS) | {
        f"pool/{folder.name}" for name in POOLS.values() if name != pool
        for folder in (R.root() / "pool").glob(f"{name}@*") if (folder / "release.json").is_file()
    }
    train = []
    for ref in sorted(train_refs):
        if not ref.startswith(f"pool/{pool}@") and (R.release_dir(ref) / "release.json").exists():
            train.extend(read_jsonl(R.resolve(ref).manifest))
    candidates, drops = unique_candidates(rows, train)
    reserved = {tuple(dup_key(term).split()) for term in read_lines(REPO / "configs/dataset/reserved_terms_v1.txt")}
    lengths = {len(term) for term in reserved if term}
    filtered = []
    for row in candidates:
        text = dup_key(row_text(row)).split()
        if any(tuple(text[i:i + length]) in reserved for length in lengths for i in range(len(text) - length + 1)):
            drops.append(dict(row, drop_reason="reserved_term"))
        else:
            filtered.append(row)

    candidate_cache = work / "candidate-fingerprints.npz"
    cache = load_cache(candidate_cache)
    reference_cache = load_cache(DEFAULT_CACHE)
    ref_rows = [row for _, group in refs for row in group]
    cache.update(local_cache(reference_cache, [*train, *ref_rows, *filtered]))
    cache.update({r["_audio_path"]: reference_cache[r["_audio_path"]] for r in [*train, *ref_rows, *filtered]
                  if r.get("_audio_path") in reference_cache})
    del reference_cache
    fingerprint_rows(filtered, workers=workers, cache=cache, log_every=500)
    drops.extend(dict(row, drop_reason="fingerprint_unavailable")
                 for row in filtered if row["_audio_path"] not in cache)
    filtered = [row for row in filtered if row["_audio_path"] in cache]
    if not filtered:
        raise ValueError("No usable candidate fingerprints")
    pending = work / "candidate-fingerprints.tmp.npz"
    save_cache(pending, {row["_audio_path"]: cache[row["_audio_path"]] for row in filtered})
    pending.replace(candidate_cache)
    fingerprints = audit_acoustic([("candidates", filtered)], refs, workers=workers, cache=cache, log_every=1000)
    training_overlap = audit_acoustic([("candidates", filtered)], [("existing_train", train)],
                                      workers=workers, cache=cache, max_ber=clean.ACOUSTIC_SPREAD_BER, log_every=1000)
    matching_train = set(training_overlap["train_ids"].get("candidates", []))
    acoustic = {}
    for hit in fingerprints["matches"]:
        key = "acoustic" if hit["ber"] <= clean.ACOUSTIC_SPREAD_BER else "acoustic_partial"
        acoustic.setdefault(hit["train_id"], {}).setdefault(key, set()).add(hit["bench"])
    keep, dropped, stats = clean.clean(
        filtered, refs, near_threshold=0.6,
        extra_hits={index: acoustic[row["id"]] for index, row in enumerate(filtered) if row["id"] in acoustic},
    )
    drops.extend(dict(row, drop_reason="duplicate_training_fingerprint") for row in keep if row["id"] in matching_train)
    keep = [row for row in keep if row["id"] not in matching_train]
    checks = clean.verify(keep, refs, near_threshold=0.6)
    if any(result["critical"] for result in checks.values()):
        raise ValueError("Evaluation overlap remains after cleaning")
    out = work / "screened.jsonl"
    clean.write_pool(out, keep, [*drops, *dropped])
    reviewed = write_review_queue(work / "review.csv", read_jsonl(out))
    report = {
        "source": source, "revision": spec["revision"], "license": spec["license"],
        "downloaded_hours": hours(rows), "audited_rows": len(keep), "audited_unique_hours": hours(keep),
        "human_reviewed_hours": reviewed, "v03_eligible_hours": 0,
        "screened_manifest_sha256": R.sha256(out), "candidates_sha256": R.sha256(manifest),
        "drops": dict(Counter(row["drop_reason"] for row in [*drops, *dropped])),
        "training_rows_compared": len(train), "evaluation_checks": checks, "evaluation_hashing": audio_stats,
        "acoustic": fingerprints["stats"], "training_acoustic": training_overlap["stats"], "metadata_audit": stats,
    }
    (work / "screened.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("source", "downloaded_hours", "audited_rows", "audited_unique_hours", "drops")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=POOLS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--root", type=Path, default=path("data_root") / "work/vietnote/pilots")
    args = parser.parse_args()
    audit(args.source, args.root / args.source, args.workers)


if __name__ == "__main__":
    main()
