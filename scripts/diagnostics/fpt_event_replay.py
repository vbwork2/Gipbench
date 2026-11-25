"""Count the Phase 2 near-miss events that replicate under the Joiner replay, per pool.

`TODO_rea.md` item 1.4. Phase 4 found 65 of 104 bug-mining events replicated and the trigger plan needs
the measured count, not the 62.5% estimate. Rules are the ones fixed in `scripts/08_joiner_decomposition.py`:
* event: a NEAR_MISS piece (greedy-prefix `logit_lens.classify`) below rank 1 whose final-stack leader is
  blank, at the Phase 2 frame of its best final-stack rank;
* replicated: in the greedy replay with INT8 encoder and decoder outputs and the fp32 output layer
  (`joiner_probe.replay`), blank is the argmax at that frame and the piece is within the top ten.
Encoder outputs are computed from the audio with `OnnxTransducer` and cached as `<audio_idx>.npy`.

Groups: bug mining (stored Phase 2 ranks), other development (`fpt_pool_near_miss.py`), extra diagnostic
(`fpt_pool_near_miss.py --extra`, from `extra_diagnostic.jsonl` only; the new heldout is not opened).
Harness check: replayed on the Phase 4 encoder cache (`--phase4-encoder-dir`, written by script 07), bug
mining must give back 104 events and 65 replicated, otherwise the script stops. The counts themselves use
freshly computed encoder outputs for every group, so the three pools are measured alike. The INT8 encoder
is sensitive to tiny feature differences (the NumPy filterbank matches kaldi-native-fbank to about 6e-5,
which already moves encoder outputs by about 0.2), so the fresh bug-mining count can differ from 65 by a
few events; it is reported next to the cache count as the size of that numerical noise.
Counting only; nothing is selected.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from gipbench.core.fileio import sha256_file as file_sha
from gipformer_mech import joiner_probe as jp
from gipformer_mech.fpt import audio_indices, manifest_waveform, parquet_files, read_rows
from gipformer_mech.logit_lens import classify
from gipformer_mech.onnx_greedy import OnnxTransducer
from gipformer_mech.rnnt_loader import resample

PHASE4 = {"events": 104, "replicated": 65}


def events_of(findings: list[dict], blank: int) -> list[dict]:
    events = []
    for row in findings:
        for error in row["errors"]:
            if classify(error["pieces"]) != "NEAR_MISS":
                continue
            for index, piece in enumerate(error["pieces"]):
                last = piece["stacks"][-1]["full"]
                if last["best_rank"] > 1 and last["top10"][0]["token_id"] == blank:
                    events.append({"utt_id": row["utt_id"], "word_index": error["word_index"],
                                   "word": error["reference_word"], "piece": index,
                                   "target": piece["target_id"], "frame": last["frame"]})
    return events


def encode_missing(model, rows: dict[str, dict], encoder_dir: Path, data_dir: Path) -> None:
    encoder_dir.mkdir(parents=True, exist_ok=True)
    wanted = {row["audio_idx"]: row for row in rows.values() if not (encoder_dir / f"{row['audio_idx']}.npy").is_file()}
    print(f"Encoding {len(wanted)} of {len(rows)} recordings", flush=True)
    for done, (index, raw) in enumerate(audio_indices(parquet_files(data_dir), set(wanted)), 1):
        waveform, rate = manifest_waveform(wanted[index], raw)
        waveform = resample(waveform, rate, model.sample_rate)
        np.save(encoder_dir / f"{index}.npy", model.encode(model.features(waveform, model.sample_rate)))
        if done % 100 == 0:
            print(f"Encoded {done} of {len(wanted)}", flush=True)


def replicate(model, events: list[dict], rows: dict[str, dict], encoder_dir: Path, weight, bias) -> list[dict]:
    by_utt = defaultdict(list)
    for event in events:
        by_utt[event["utt_id"]].append(event)
    out = []
    for utt_id, items in sorted(by_utt.items()):
        state = jp.replay(model, np.load(encoder_dir / f"{rows[utt_id]['audio_idx']}.npy"), weight, bias)
        for event in items:
            if event["frame"] >= len(state["hidden"]):
                out.append({**event, "replicated": False, "reason": "frame outside replay"})
                continue
            logits = weight @ state["hidden"][event["frame"]] + bias
            rank = int((logits > logits[event["target"]]).sum()) + 1
            blank_leads = int(state["tokens"][event["frame"]]) == model.blank
            out.append({**event, "replay_rank": rank, "blank_leads": blank_leads,
                        "replicated": blank_leads and rank <= 10})
    return out


def describe(results: list[dict]) -> dict:
    hit = [row for row in results if row["replicated"]]
    return {"events": len(results), "replicated": len(hit),
            "replicated_share": len(hit) / len(results) if results else 0.0,
            "recordings_with_replicated": len({row["utt_id"] for row in hit}),
            "distinct_words_replicated": len({row["word"] for row in hit})}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--extra-dir", type=Path, default=ROOT / "data/manifests_extra")
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--extra", type=Path, default=ROOT / "artifacts/extra_near_miss/findings.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/event_replay")
    parser.add_argument("--phase4-encoder-dir", type=Path, default=ROOT / "artifacts/conditional_blank/encoder")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    blank = model_map["blank_id"]
    lock = json.loads((args.manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    pool_path = args.manifests / "full_scored.jsonl"
    if file_sha(pool_path) != lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    extra_path = args.extra_dir / "extra_diagnostic.jsonl"
    if file_sha(extra_path) != json.loads(args.split_lock.read_text(encoding="utf-8"))["diagnostic"]["sha256"]:
        raise ValueError("extra_diagnostic.jsonl differs from the split lock.")
    rows = {row["utt_id"]: row for path in (pool_path, extra_path) for row in read_rows(path)}

    groups = {
        "bug_mining": events_of(json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"], blank),
        "other_development": events_of(read_rows(args.pool), blank),
        "extra_diagnostic": events_of(read_rows(args.extra), blank),
    }
    model = OnnxTransducer(args.model_dir, threads=args.threads)
    weight, bias = jp.load_output_layer(args.model_dir, model_map)
    encoder_dir = args.out_dir / "encoder"
    needed = {event["utt_id"]: rows[event["utt_id"]] for events in groups.values() for event in events}
    encode_missing(model, needed, encoder_dir, args.data_dir)

    check = describe(replicate(model, groups["bug_mining"], rows, args.phase4_encoder_dir, weight, bias))
    if {key: check[key] for key in PHASE4} != PHASE4:
        raise ValueError(f"On the Phase 4 encoder cache bug mining gives {check}, not {PHASE4}.")
    results = {name: replicate(model, events, rows, encoder_dir, weight, bias) for name, events in groups.items()}
    summary = {name: describe(items) for name, items in results.items()}
    summary["development"] = describe(results["bug_mining"] + results["other_development"])
    summary["bug_mining_on_phase4_cache"] = check
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "events.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for name, items in results.items():
            for item in items:
                stream.write(json.dumps({"group": name, **item}, ensure_ascii=False) + "\n")
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
