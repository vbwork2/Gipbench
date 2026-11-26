"""Acquire deterministic train pilots from pinned public snapshots."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import yaml

from gipbench.core.download import download_file, verify_file
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO, path
from gipbench.dataset.ingest.evalsets import read_fleurs, write_audio
from gipbench.dataset.ingest.parquet_range import ByteBudgetExceeded, ParquetRanges

CONFIG = REPO / "configs/dataset/vietnote_sources.yaml"


def eligible(row, spec):
    return all(row.get(key) in values for key, values in spec.get("filters", {}).items())


def make_row(key, spec, row, rg, index, out, max_seconds, seen, disk_reserve=3 * 2**30):
    import numpy as np
    import soundfile as sf

    text = str(row.get(spec["text"]) or "").strip()
    if not text or not eligible(row, spec):
        return None, "metadata_filter"
    audio = row.get("audio") or {}
    data = audio.get("bytes")
    if not data:
        return None, "missing_audio"
    sha = hashlib.sha256(data).hexdigest()
    if sha in seen:
        return None, "duplicate_audio"
    try:
        with sf.SoundFile(io.BytesIO(data)) as stream:
            duration = len(stream) / stream.samplerate
            if not 1 <= duration <= min(45, max_seconds) or stream.samplerate < 8000:
                return None, "duration_or_sampling_rate"
            samples = stream.read(dtype="float32")
    except (OSError, RuntimeError, ValueError):
        return None, "decode_error"
    if not samples.size or not np.isfinite(samples).all() or np.max(np.abs(samples)) < 0.001:
        return None, "silent_or_invalid"
    if float(np.mean(np.abs(samples) >= 0.999)) > 0.01:
        return None, "clipping"

    get = lambda field: row.get(spec.get(field, ""))  # noqa: E731
    audio_name = str(audio.get("path") or "")
    upstream_id = get("id") or Path(audio_name).name or f"{rg}:{index}"
    uid = hashlib.sha256(f"{key}\0{spec['file']}\0{upstream_id}\0{sha}".encode()).hexdigest()[:20]
    audio_out = out / "audio" / f"vn_{key}_{uid}"
    if shutil.disk_usage(out).free < len(data) + disk_reserve:
        raise OSError(f"Insufficient disk space to materialize {key}")
    p, derivation, duration = write_audio(data, Path(audio_name).suffix or spec.get("ext", ".wav"), audio_out)
    seen.add(sha)
    source = get("source")
    speaker = get("speaker")
    category = spec["language_category"]
    return {
        "id": audio_out.name, "split": "train", "train_only": True, "dataset": spec["dataset"],
        "audio": os.path.relpath(p, out).replace(os.sep, "/"), "duration": duration,
        "audio_sha256": sha256_file(p), "upstream_audio_sha256": sha, "audio_derivation": derivation,
        "text_orig": text, "text_tagged": f"<en>{text}</en>" if category == "native_en" else text,
        "source": str(source) if source is not None and source != "" else f"{key}:{upstream_id}",
        "speaker": f"{key}:{speaker}" if speaker is not None and str(speaker) not in ("", "-1") else None,
        "upstream_id": str(upstream_id), "upstream_revision": spec["revision"],
        "upstream_repo": spec["repo"], "upstream_split": spec["split"], "upstream_shard": spec["file"],
        "upstream_row_group": rg, "upstream_row_index": index, "license": spec["license"],
        "label_source": spec["label_source"], "domain": spec["domain"], "speech_type": spec["speech_type"],
        "language_category": category, "quality_status": "technical_qc_passed",
        "synthetic": spec["speech_type"] == "synthetic",
        "provenance": {k: row[k] for k in spec.get("metadata", []) if row.get(k) is not None},
        **spec.get("row_flags", {}),
    }, None


def parquet_rows(ranges, fields, key, seed):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(ranges, pre_buffer=False)
    missing = fields - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"{key}: missing parquet columns: {sorted(missing)}")
    groups = list(range(pf.num_row_groups))
    random.Random(seed).shuffle(groups)
    for rg in groups:
        batch = pf.read_row_group(rg, columns=sorted(fields), use_threads=False)
        index = list(range(batch.num_rows))
        random.Random(f"{seed}:{rg}").shuffle(index)
        for i in index:
            yield rg, i, batch.slice(i, 1).to_pylist()[0]


def fleurs_rows(ranges, tsv):
    for i, (name, meta, data, _) in enumerate(read_fleurs({}, [tsv, ranges])):
        yield None, i, {"audio": {"path": name, "bytes": data}, **meta}


def write_review_queue(out, rows):
    fields = ["id", "audio", "audio_sha256", "duration", "text_tagged", "speaker", "decision", "notes"]
    previous = {}
    if out.exists():
        with out.open(encoding="utf-8-sig", newline="") as handle:
            previous = {(row["id"], row["audio_sha256"], row["text_tagged"]): row for row in csv.DictReader(handle)}
    reviewed = 0.0
    pending = out.with_suffix(".tmp")
    with pending.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            old = previous.get((row["id"], row["audio_sha256"], row["text_tagged"]), {})
            decision = old.get("decision", "")
            if decision in {"accept", "reject", "fix", "ok"}:
                reviewed += float(row["duration"]) / 3600
            writer.writerow({**row, "decision": decision, "notes": old.get("notes", "")})
    pending.replace(out)
    return reviewed


def recover_journal(manifest):
    # A terminated write may leave an incomplete final record.
    if not manifest.exists():
        return
    with manifest.open("r+b") as handle:
        content = handle.read()
        if content and not content.endswith(b"\n"):
            handle.truncate(content.rfind(b"\n") + 1)


def acquire(key, spec, out, args):
    budget = 0 if args.cache_only else args.byte_budget_mib * 1024**2
    base = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}"
    dl = path("data_root") / "dl/vietnote" / key / spec["revision"] / spec["sha256"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    fields = {"audio", spec["text"], *spec.get("filters", {}), *spec.get("metadata", []),
              *(spec[k] for k in ("id", "source", "speaker") if spec.get(k))}
    manifest = out / "candidates.jsonl"
    if manifest.exists() and not args.resume:
        raise ValueError(f"Candidate manifest already exists. Use --resume: {manifest}")
    if args.resume:
        recover_journal(manifest)
    rows = read_jsonl(manifest) if manifest.exists() else []
    for row in rows:
        if (row["upstream_revision"] != spec["revision"] or row["upstream_shard"] != spec["file"]
                or sha256_file(row["_audio_path"]) != row["audio_sha256"]):
            raise ValueError(f"Stale candidate: {row['id']}")
    count = len(rows)
    seen = {row["upstream_audio_sha256"] for row in rows}
    previous = out / "acquisition.json"
    prior = json.loads(previous.read_text(encoding="utf-8")) if previous.exists() else {}
    drops = Counter()
    seconds = sum(row["duration"] for row in rows)
    tsv = None
    metadata_bytes = 0
    reserve = int(args.disk_reserve_gib * 2**30)
    if spec.get("tsv"):
        metadata = spec["tsv"]
        tsv = dl / Path(metadata["path"]).name
        metadata_bytes = 0 if tsv.exists() else metadata["bytes"]
        if args.cache_only:
            verify_file(tsv, metadata)
        else:
            if metadata_bytes >= budget:
                raise ByteBudgetExceeded("The TSV exceeds the download budget")
            download_file(f"{base}/{metadata['path']}", tsv, metadata, reserve)
    with (ParquetRanges(f"{base}/{spec['file']}", spec["bytes"], dl, budget - metadata_bytes, reserve) as ranges,
          manifest.open("a", encoding="utf-8") as journal):
        stop = "shard_exhausted"
        try:
            raw_rows = fleurs_rows(ranges, tsv) if tsv else parquet_rows(ranges, fields, key, args.seed)
            while count < args.max_rows and args.hours * 3600 - seconds >= 1:
                current = next(raw_rows, None)
                if current is None:
                    break
                rg, i, raw = current
                row, reject = make_row(key, spec, raw, rg, i, out, args.hours * 3600 - seconds, seen, reserve)
                if reject:
                    drops[reject] += 1
                else:
                    journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                    journal.flush()
                    count += 1
                    seconds += row["duration"]
            else:
                stop = "pilot_limit"
        except ByteBudgetExceeded:
            stop = "byte_budget"
        transferred = ranges.transferred + metadata_bytes
        cached = [dl / f"{start:014d}.block" for start in range(0, spec["bytes"], ranges.block_size)]
        verified = False
        if all(file.exists() for file in cached):
            digest = hashlib.sha256()
            for file in cached:
                digest.update(file.read_bytes())
            if digest.hexdigest() != spec["sha256"]:
                raise ValueError(f"Complete shard digest differs from the pinned digest: {key}")
            verified = True
    report = dict(source=key, revision=spec["revision"], shard=spec["file"], shard_bytes=spec["bytes"],
                  upstream_shard_sha256=spec["sha256"], shard_hash_verified=verified, seed=args.seed,
                  previous_candidate_rows=len(rows), candidate_rows=count,
                  candidate_hours=round(seconds / 3600, 6), usable_registered_hours=0,
                  new_download_bytes=transferred, status=stop, dropped=dict(drops), license=spec["license"],
                  cache_only=args.cache_only,
                  recorded_download_bytes=transferred + prior.get(
                      "recorded_download_bytes", prior.get("new_download_bytes", prior.get("new_range_bytes", 0))),
                  sampling="archive_prefix" if tsv else "seeded_row_groups")
    if spec["status"] != "pilot_eligible":
        report["human_reviewed_hours"] = write_review_queue(out / "quarantine-review.csv", read_jsonl(manifest))
    pending = previous.with_suffix(".tmp")
    pending.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pending.replace(previous)
    print(json.dumps(report), flush=True)


def discover_shard(spec):
    parent = Path(spec["file"]).parent.as_posix()
    url = f"https://huggingface.co/api/datasets/{spec['repo']}/tree/{spec['revision']}/{parent}"
    with urlopen(url, timeout=30) as response:
        shard = next((item for item in json.load(response) if item.get("path") == spec["file"]), None)
    if not shard or shard["size"] != spec["bytes"] or (shard.get("lfs") or {}).get("oid") != spec["sha256"]:
        raise ValueError("The upstream shard no longer matches the registered size and digest")
    return {"path": shard["path"], "bytes": shard["size"], "sha256": spec["sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--discover", action="store_true", help="Verify the pinned train shard against Hugging Face metadata")
    parser.add_argument("--cache-only", action="store_true", help="Process cached bytes without downloading recordings")
    parser.add_argument("--quarantine-pilot", action="store_true",
                        help="Allow selective pilot acquisition for authenticated quarantine sources")
    parser.add_argument("--disk-reserve-gib", type=float, default=1.5,
                        help="Minimum free disk space in GiB to keep on the target volume")
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--byte-budget-mib", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=path("data_root") / "work/vietnote/pilots")
    args = parser.parse_args()
    if args.hours <= 0 or args.byte_budget_mib < 1 or args.max_rows < 1:
        parser.error("Hours, byte budget and max rows must be positive")
    specs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["sources"]
    if not (args.dry_run or args.discover) and not args.source:
        parser.error("Select at least one source")
    failed = False
    for key in args.source or list(specs):
        if key not in specs:
            parser.error(f"Unknown source: {key}")
        spec = specs[key]
        try:
            approved = spec["status"] == "pilot_eligible"
            quarantine_ok = args.quarantine_pilot and str(spec["status"]).startswith("pilot_quarantine_")
            if args.discover and approved:
                print(json.dumps({"source": key, "revision": spec["revision"], **discover_shard(spec)}))
            elif args.dry_run or args.discover or "file" not in spec or not (approved or quarantine_ok or args.cache_only):
                print(json.dumps({"source": key, "status": spec["status"], "shard": spec.get("file"),
                                  "shard_bytes": spec.get("bytes"), "max_download_mib": args.byte_budget_mib,
                                  "max_hours": args.hours, "license": spec.get("license")}, ensure_ascii=False))
                failed |= not (args.dry_run or args.discover)
            else:
                acquire(key, spec, args.out / key, args)
        except (OSError, ValueError, RuntimeError) as exc:
            failed = True
            print(f"{key}: {exc}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
