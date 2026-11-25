"""Acquire deterministic train pilots from pinned public snapshots."""

import argparse
import hashlib
import io
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import yaml

from gipbench.core.download import download_file
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import write_jsonl
from gipbench.core.paths import REPO, path
from gipbench.dataset.ingest.evalsets import read_fleurs, write_audio
from gipbench.dataset.ingest.parquet_range import ByteBudgetExceeded, ParquetRanges

CONFIG = REPO / "configs/dataset/vietnote_sources.yaml"


def eligible(row, spec):
    return all(row.get(key) in values for key, values in spec.get("filters", {}).items())


def make_row(key, spec, row, rg, index, out, max_seconds, seen):
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
    if shutil.disk_usage(out).free < len(data) + 3 * 2**30:
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


def fleurs_rows(ranges, tsv, seed):
    # Sample within the archive prefix because the original gzip stream is sequential.
    rng = random.Random(seed)
    for i, (name, meta, data, _) in enumerate(read_fleurs({}, [tsv, ranges])):
        if rng.random() < 0.65:
            yield None, i, {"audio": {"path": name, "bytes": data}, **meta}


def acquire(key, spec, out, args):
    budget = args.byte_budget_mib * 1024**2
    base = f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}"
    dl = path("data_root") / "dl/vietnote" / key / spec["revision"] / spec["sha256"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(exist_ok=True)
    fields = {"audio", spec["text"], *spec.get("filters", {}), *spec.get("metadata", []),
              *(spec[k] for k in ("id", "source", "speaker") if spec.get(k))}
    rows, seen, drops = [], set(), Counter()
    seconds = 0.0
    tsv = None
    metadata_bytes = 0
    if spec.get("tsv"):
        metadata = spec["tsv"]
        tsv = dl / Path(metadata["path"]).name
        metadata_bytes = 0 if tsv.exists() else metadata["bytes"]
        if metadata_bytes >= budget:
            raise ByteBudgetExceeded("The TSV exceeds the download budget")
        download_file(f"{base}/{metadata['path']}", tsv, metadata, 3 * 2**30)
    with ParquetRanges(f"{base}/{spec['file']}", spec["bytes"], dl, budget - metadata_bytes, 3 * 2**30) as ranges:
        stop = "shard_exhausted"
        try:
            raw_rows = fleurs_rows(ranges, tsv, args.seed) if tsv else parquet_rows(ranges, fields, key, args.seed)
            for rg, i, raw in raw_rows:
                row, reject = make_row(key, spec, raw, rg, i, out, args.hours * 3600 - seconds, seen)
                if reject:
                    drops[reject] += 1
                else:
                    rows.append(row)
                    seconds += row["duration"]
                if len(rows) >= args.max_rows or args.hours * 3600 - seconds < 1:
                    stop = "pilot_limit"
                    break
        except ByteBudgetExceeded:
            stop = "byte_budget"
        transferred = ranges.transferred + metadata_bytes
    write_jsonl(str(out / "candidates.jsonl"), rows)
    report = dict(source=key, revision=spec["revision"], shard=spec["file"], shard_bytes=spec["bytes"],
                  upstream_shard_sha256=spec["sha256"], shard_hash_verified=False, seed=args.seed,
                  candidate_rows=len(rows), candidate_hours=round(seconds / 3600, 6), usable_registered_hours=0,
                  new_download_bytes=transferred, status=stop, dropped=dict(drops), license=spec["license"],
                  sampling="seeded_archive_prefix" if tsv else "seeded_row_groups")
    (out / "acquisition.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--byte-budget-mib", type=int, default=128)
    parser.add_argument("--max-rows", type=int, default=160)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=path("data_root") / "work/vietnote/pilots")
    args = parser.parse_args()
    if args.hours <= 0 or args.byte_budget_mib < 1 or args.max_rows < 1:
        parser.error("Hours, byte budget and max rows must be positive")
    specs = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["sources"]
    if not args.dry_run and not args.source:
        parser.error("Select at least one source")
    failed = False
    for key in args.source or list(specs):
        if key not in specs:
            parser.error(f"Unknown source: {key}")
        spec = specs[key]
        if args.dry_run or "file" not in spec:
            print(json.dumps({"source": key, "status": spec["status"], "shard": spec.get("file"),
                              "shard_bytes": spec.get("bytes"), "max_download_mib": args.byte_budget_mib,
                              "max_hours": args.hours, "license": spec.get("license")}, ensure_ascii=False))
            continue
        try:
            acquire(key, spec, args.out / key, args)
        except (OSError, ValueError, RuntimeError) as exc:
            failed = True
            print(f"{key}: {exc}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
