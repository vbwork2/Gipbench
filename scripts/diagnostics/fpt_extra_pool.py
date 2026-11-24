"""Score the FPT FOSD rows that the sealed pool never sampled, as a separate development pool.

`01_build_manifests.py` sampled 5,600 of the 25,917 rows with seed 42, then sealed a heldout out
of them. This script scores the remaining rows with the same greedy INT8 decoder, filters and
batch size, and writes them next to, never into, the sealed manifests. The sealed
`full_scored.jsonl` and the heldout are not modified.

Safety rules, fixed before the run:
* the sampling is reproduced and must equal pool + heldout + dropped rows exactly, so the extra
  rows are the exact complement;
* a row whose audio checksum matches the pool, the heldout or an earlier extra row is dropped;
* a row whose normalized reference text equals a heldout reference is dropped. The heldout file
  is read only for ids, checksums and reference text, to build these exclusion lists. Nothing from
  it is decoded or scored;
* the same label and duration filters as `01_build_manifests.py` apply.

Greedy hypotheses depend on batch composition, so rows are decoded in index order in batches of
`decode.batch_size`, the order of the original pool run. Shards are independent and resumable. A
shard is 1,600 rows, a multiple of the batch size.
"""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import soundfile

from gipformer_mech.fpt import audio_indices, parquet_files, read_rows, read_waveform, recognizer, settings
from gipformer_mech.fpt import transcribe, write_rows
from gipformer_mech.metrics import score, summarize
from gipformer_mech.model_audit import checksum
from gipformer_mech.text_norm import tokens

SHARD_SIZE = 1600


def unused_indices(manifests: Path, config: dict, case: str, size: int) -> tuple[list[int], dict]:
    lock = json.loads((manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    pool_path = manifests / "full_scored.jsonl"
    if checksum(pool_path) != lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    target = config["dataset"]["valid_pool_size"]
    sample = set(Random(config["seed"]).sample(range(size), min(size, math.ceil(target * 1.12))))
    pool = read_rows(pool_path)
    held = read_rows(manifests / "heldout_test.jsonl")
    dropped = read_rows(manifests / "label_audit_dropped.jsonl")
    used = {row["audio_idx"] for rows in (pool, held, dropped) for row in rows}
    if used != sample:
        raise ValueError(f"Reproduced sampling differs from the sealed rows by {len(used ^ sample)} indices.")
    exclusion = {
        "sha": {row["audio_sha256"] for rows in (pool, held) for row in rows}
        | {row["audio_sha256"] for row in dropped if row.get("audio_sha256")},
        "held_sha": {row["audio_sha256"] for row in held},
        "held_text": {" ".join(tokens(row["reference"], case)) for row in held},
    }
    return [index for index in range(size) if index not in sample], exclusion


def shard_rows(files, indices: list[int], config: dict, case: str, exclusion: dict, rec) -> tuple[list, list]:
    dataset = config["dataset"]
    cs_terms = config["text"]["cs_terms"]
    seen = set(exclusion["sha"])
    scored, dropped, batch, waves = [], [], [], []

    def flush() -> None:
        for item, hyp in zip(batch, transcribe(rec, waves), strict=True):
            counts = score(item["reference"], hyp, case, cs_terms)
            metrics = summarize([counts])
            scored.append({
                **item,
                "cs_tokens": counts["cs_tokens"],
                "baseline_hyp": hyp,
                "baseline_wer": metrics["wer_global"],
                "baseline_cs_wer": metrics["cs_wer"],
                "score_counts": counts,
            })
        batch.clear()
        waves.clear()

    for index, raw in audio_indices(files, set(indices)):
        audio = raw["audio"]
        item = {
            "utt_id": f"fpt_{index:06d}",
            "audio_idx": index,
            "audio_file": audio["path"],
            "reference": raw["transcription"],
            "dataset_revision": dataset["revision"],
        }
        text = item["reference"]
        if not text or not tokens(text, case) or "�" in text:
            dropped.append({**item, "reason": "empty_or_malformed_reference"})
            continue
        try:
            wav, rate = read_waveform(audio["bytes"])
        except soundfile.LibsndfileError:
            dropped.append({**item, "reason": "unreadable_audio"})
            continue
        duration = len(wav) / rate
        if not dataset["min_duration"] <= duration <= dataset["max_duration"] or not np.isfinite(wav).all():
            dropped.append({**item, "duration_sec": duration, "reason": "invalid_audio_or_duration"})
            continue
        digest = sha256(audio["bytes"]).hexdigest()
        if digest in exclusion["held_sha"]:
            dropped.append({**item, "duration_sec": duration, "audio_sha256": digest,
                            "reason": "heldout_audio_overlap"})
            continue
        if digest in seen:
            dropped.append({**item, "duration_sec": duration, "audio_sha256": digest, "reason": "duplicate_audio"})
            continue
        if " ".join(tokens(text, case)) in exclusion["held_text"]:
            dropped.append({**item, "duration_sec": duration, "audio_sha256": digest,
                            "reason": "heldout_text_overlap"})
            continue
        seen.add(digest)
        batch.append({**item, "duration_sec": duration, "audio_sha256": digest})
        waves.append((wav, rate))
        if len(batch) >= config["decode"]["batch_size"]:
            flush()
    flush()
    return scored, dropped


def merge(out_dir: Path, shards: int) -> dict:
    """Join the shards in index order, dropping later rows whose audio repeats an earlier extra row."""
    scored, dropped, seen = [], [], set()
    for shard in range(shards):
        for row in read_rows(out_dir / f"shard_{shard:03d}.scored.jsonl"):
            if row["audio_sha256"] in seen:
                dropped.append({key: row[key] for key in (
                    "utt_id", "audio_idx", "audio_file", "reference", "duration_sec", "audio_sha256"
                )} | {"reason": "duplicate_audio"})
                continue
            seen.add(row["audio_sha256"])
            scored.append(row)
        dropped.extend(read_rows(out_dir / f"shard_{shard:03d}.dropped.jsonl"))
    write_rows(out_dir / "extra_scored.jsonl", scored)
    write_rows(out_dir / "extra_dropped.jsonl", dropped)
    cs_errors = [row for row in scored if row["score_counts"]["cs_word_errors"]]
    summary = {
        "scored": len(scored),
        "dropped": dict(Counter(row["reason"] for row in dropped)),
        "hours": sum(row["duration_sec"] for row in scored) / 3600,
        "rows_with_cs_tokens": sum(bool(row["cs_tokens"]) for row in scored),
        "rows_with_cs_errors": len(cs_errors),
        "cs_error_spans": sum(len(row["score_counts"]["cs_word_errors"]) for row in cs_errors),
        "greedy": summarize(row["score_counts"] for row in scored),
        "extra_scored_sha256": checksum(out_dir / "extra_scored.jsonl"),
        "extra_dropped_sha256": checksum(out_dir / "extra_dropped.jsonl"),
    }
    (out_dir / "extra_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/manifests_extra")
    parser.add_argument("--max-shards", type=int, help="Run at most this many new shards (timing test).")
    args = parser.parse_args()
    if args.manifests.resolve() == args.out_dir.resolve():
        parser.error("The extra pool must not be written into the sealed manifest directory.")

    config, case = settings()
    if config["decode"]["batch_size"] <= 0 or SHARD_SIZE % config["decode"]["batch_size"]:
        raise ValueError("Shard size must be a multiple of the decode batch size.")
    files = parquet_files(args.data_dir)
    size = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    indices, exclusion = unused_indices(args.manifests, config, case, size)
    shards = math.ceil(len(indices) / SHARD_SIZE)
    print(f"{len(indices)} unsampled rows in {shards} shards; {len(exclusion['sha'])} audio checksums "
          f"and {len(exclusion['held_text'])} heldout texts excluded", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rec = recognizer(args.model_dir, args.out_dir, config, case, "greedy_search")
    ran = 0
    for shard in range(shards):
        scored_path = args.out_dir / f"shard_{shard:03d}.scored.jsonl"
        dropped_path = args.out_dir / f"shard_{shard:03d}.dropped.jsonl"
        if scored_path.exists() and dropped_path.exists():
            continue
        if args.max_shards is not None and ran >= args.max_shards:
            break
        scored, dropped = shard_rows(files, indices[shard * SHARD_SIZE:(shard + 1) * SHARD_SIZE],
                                     config, case, exclusion, rec)
        write_rows(scored_path, scored)
        write_rows(dropped_path, dropped)  # written last: its presence marks the shard as complete
        ran += 1
        print(f"Shard {shard + 1}/{shards}: {len(scored)} scored, {len(dropped)} dropped", flush=True)
    if all((args.out_dir / f"shard_{s:03d}.dropped.jsonl").exists() for s in range(shards)):
        print(json.dumps(merge(args.out_dir, shards), indent=2))


if __name__ == "__main__":
    main()
