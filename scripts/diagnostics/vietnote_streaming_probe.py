"""Measure first and late emissions on fixed long development utterances."""

import argparse
import json
import time
from pathlib import Path

from gipbench.bench.transcribe import load_model_cfg
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.infer.adapters import build_adapter
from gipbench.infer.adapters.base import load_audio


def probe(adapter, rows):
    records = []
    for row in rows:
        samples, rate = load_audio(row["_audio_path"], target_sr=None)
        start = time.perf_counter()
        updates = []

        def emitted(audio_sec, text):
            if text.strip() and (not updates or updates[-1]["text"] != text):
                updates.append({"audio_sec": audio_sec, "compute_sec": time.perf_counter() - start, "text": text})

        final = adapter.transcribe_stream(samples, rate, emitted)
        audio_sec = len(samples) / rate
        records.append({"id": row["id"], "duration": audio_sec, "hyp": final,
                        "emissions": updates, "rtf": (time.perf_counter() - start) / audio_sec,
                        "changed_after_halfway": any(u["audio_sec"] >= audio_sec * 0.5 for u in updates)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--min-sec", type=float, default=20)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in read_jsonl(args.manifest) if float(r["duration"]) >= args.min_sec][:args.limit]
    if not rows:
        parser.error("No qualifying long utterances")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = {"manifest_sha256": manifest_sha256(rows), "min_seconds": args.min_sec, "systems": {}}
    for cfg_path in args.models:
        cfg = load_model_cfg(cfg_path)
        if not cfg["decode"].get("streaming"):
            parser.error(f"Streaming is required in {cfg_path}")
        adapter = build_adapter(cfg, args.out.parent)
        adapter.load()
        records = probe(adapter, rows)
        report["systems"][cfg["system"]] = {
            "model_files": adapter.info.extra["model_files"], "decode": adapter.dec,
            "n_with_late_emission": sum(r["changed_after_halfway"] for r in records),
            "n_empty": sum(not r["hyp"].strip() for r in records), "rows": records,
        }
        del adapter
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
