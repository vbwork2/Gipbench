"""Score the FPT pool with greedy decoding and prepare inputs for split validation."""

import argparse
from collections import Counter
from hashlib import sha256
import math
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pyarrow.parquet as pq
import soundfile

from gipformer_mech.fpt import audio_indices, parquet_files, read_waveform, recognizer, settings, transcribe, write_rows
from gipformer_mech.metrics import score, summarize
from gipformer_mech.splits import seal
from gipformer_mech.text_norm import tokens


def collect(args: argparse.Namespace) -> None:
    config, case = settings()
    dataset = config["dataset"]
    target = args.pool_size or dataset["valid_pool_size"]
    if target < 1:
        raise ValueError("Pool size must be positive.")
    files = parquet_files(args.data_dir)
    size = sum(pq.ParquetFile(path).metadata.num_rows for path in files)
    if target > size:
        raise ValueError(f"Pool size {target} exceeds snapshot length {size}.")
    indices = set(Random(config["seed"]).sample(range(size), min(size, math.ceil(target * 1.12))))
    rec = recognizer(args.model_dir, args.output_dir, config, case, "greedy_search")
    valid, dropped, batch, waves = [], [], [], []
    seen_audio = set()
    duplicate = Counter()
    cs_terms = config["text"]["cs_terms"]

    def flush() -> None:
        for item, hyp in zip(batch, transcribe(rec, waves), strict=True):
            counts = score(item["reference"], hyp, case, cs_terms)
            metrics = summarize([counts])
            valid.append({
                **item,
                "cs_tokens": counts["cs_tokens"],
                "baseline_hyp": hyp,
                "baseline_wer": metrics["wer_global"],
                "baseline_cs_wer": metrics["cs_wer"],
                "score_counts": counts,
            })
        batch.clear()
        waves.clear()

    for index, raw in audio_indices(files, indices):
        audio = raw["audio"]
        item = {
            "utt_id": f"fpt_{index:06d}",
            "audio_idx": index,
            "audio_file": audio["path"],
            "reference": raw["transcription"],
            "dataset_revision": dataset["revision"],
        }
        text = item["reference"]
        if not text or not tokens(text, case) or "\ufffd" in text:
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
        checksum = sha256(audio["bytes"]).hexdigest()
        if checksum in seen_audio:
            duplicate[checksum] += 1
            dropped.append({**item, "duration_sec": duration, "reason": "duplicate_audio"})
            continue
        seen_audio.add(checksum)
        batch.append({**item, "duration_sec": duration, "audio_sha256": checksum})
        waves.append((wav, rate))
        if len(batch) >= config["decode"]["batch_size"]:
            flush()
            if len(valid) % (config["decode"]["batch_size"] * 32) == 0:
                print(f"Greedy baseline: {len(valid)} utterances scored", flush=True)
    flush()
    if len(valid) < target:
        raise RuntimeError(f"Found {len(valid)} usable records for the requested pool of {target}.")
    write_rows(args.output_dir / "full_scored.jsonl", valid)
    write_rows(args.output_dir / "label_audit_dropped.jsonl", dropped)
    print(f"Scored {len(valid)} utterances; excluded {len(dropped)} ({sum(duplicate.values())} exact audio duplicates).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--pool-size", type=int)
    parser.add_argument("--speaker-map", type=Path, help="TSV assigning verified speaker IDs to utt_id.")
    parser.add_argument("--rejected-labels", type=Path, help="Reviewed JSONL with utt_id and exclusion reason.")
    args = parser.parse_args()
    if (args.output_dir / "heldout_test.jsonl").exists():
        parser.error("The heldout manifest already exists. Use a separate run directory for a new pool.")
    scored = args.output_dir / "full_scored.jsonl"
    if not args.speaker_map or not scored.exists():
        collect(args)
    if args.speaker_map:
        config, _ = settings()
        seal(args.output_dir, args.speaker_map, config, args.rejected_labels)
        print("Wrote the development splits and sealed heldout manifest.")


if __name__ == "__main__":
    main()
