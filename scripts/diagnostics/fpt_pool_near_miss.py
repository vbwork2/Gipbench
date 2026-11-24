"""Count Phase 2 near-miss spans over the whole sealed FPT development pool.

Phase 2 diagnosed only the 100 bug-mining recordings, which were chosen for their most
repeated confusions. This script applies the same logit-lens diagnosis to every other
development recording that has a CS substitution or deletion, then pools the result with
the stored bug-mining findings. It never reads the heldout: the sealed pool no longer
contains those rows, and the pool hash is checked against the lock.

Counting rules, fixed before the run:
* span = one CS substitution or deletion from `metrics.score` on the stored greedy hypothesis (`baseline_hyp` is greedy ONNX INT8);
* category = `logit_lens.classify` with the emitted greedy prefix (the primary Phase 2 context);
* blank-led near miss = `logit_lens.blank_led_near_miss`;
* the unbiased estimate is the new-recording group alone, because bug-mining was selected
  for repeated confusions. The pooled counts are for sizing a trigger-learning set only.

Findings are appended one recording per line, so an interrupted run resumes where it stopped.

`--extra` (TODO_rea item 1.3) runs the same diagnosis, with the same counting rules, on the
diagnostic part of the extra pool (`manifests_extra/extra_diagnostic.jsonl`). Its sha256 must equal
the `diagnostic` entry of `reports/phase1_extra_split_lock.json`. The new heldout is not opened. The
extra findings are summarized on their own and never pooled with bug mining or the sealed pool.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

from gipformer_mech.fpt import audio_indices, manifest_waveform, parquet_files, read_rows, settings
from gipformer_mech.logit_lens import INTERPRETATION, classify, investigate, summarize_findings
from gipformer_mech.metrics import score
from gipformer_mech.model_audit import checksum
from gipformer_mech.rnnt_loader import load_model, resample
from gipformer_mech.splits import development_rows
from gipformer_mech.text_norm import tokens


def reclassify(findings: list[dict]) -> list[dict]:
    """Apply the current categories to stored ranks (bug-mining was stored before NEAR_MISS existed)."""
    return [
        {**row, "errors": [
            {**error, "category": classify(error["pieces"]),
             "reference_prefix_category": classify(error["pieces"], "reference_full")}
            for error in row["errors"]
        ]}
        for row in findings
    ]


def targets(manifests: Path, config: dict, case: str) -> list[tuple[dict, list[int]]]:
    lock = json.loads((manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    pool_path = manifests / "full_scored.jsonl"
    if checksum(pool_path) != lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    seen = {row["utt_id"] for row in development_rows(manifests, "bug_mining")}
    held = {row["utt_id"] for row in read_rows(manifests / "heldout_test.jsonl")}
    selected = []
    for row in sorted(read_rows(pool_path), key=lambda item: item["utt_id"]):
        if row["utt_id"] in seen:
            continue
        if row["utt_id"] in held:
            raise ValueError("The development pool contains a heldout row.")
        counts = score(row["reference"], row["baseline_hyp"], case, config["text"]["cs_terms"])
        positions = [error["ref_index"] for error in counts["cs_word_errors"]]
        if positions:
            selected.append((row, positions))
    return selected


def extra_targets(extra_dir: Path, lock_path: Path, config: dict, case: str) -> list[tuple[dict, list[int]]]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    path = extra_dir / "extra_diagnostic.jsonl"
    if checksum(path) != lock["diagnostic"]["sha256"]:
        raise ValueError("extra_diagnostic.jsonl differs from the split lock.")
    selected = []
    for row in sorted(read_rows(path), key=lambda item: item["utt_id"]):
        counts = score(row["reference"], row["baseline_hyp"], case, config["text"]["cs_terms"])
        positions = [error["ref_index"] for error in counts["cs_word_errors"]]
        if positions:
            selected.append((row, positions))
    spans = sum(len(positions) for _, positions in selected)
    if (len(selected), spans) != (lock["diagnostic"]["rows_with_cs_errors"], lock["diagnostic"]["cs_error_spans"]):
        raise ValueError(f"Rescored CS errors ({len(selected)} rows, {spans} spans) differ from the split lock.")
    return selected


def diagnose(args, selected, model, sp, model_map, case) -> None:
    out = args.out_dir / "findings.jsonl"
    done = {json.loads(line)["utt_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()} \
        if out.exists() else set()
    todo = {row["audio_idx"]: (row, positions) for row, positions in selected if row["utt_id"] not in done}
    print(f"{len(selected)} recordings with CS errors, {len(done)} already done, {len(todo)} to run", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as stream:
        for count, (index, raw) in enumerate(audio_indices(parquet_files(args.data_dir), set(todo)), 1):
            row, positions = todo[index]
            waveform, rate = manifest_waveform(row, raw)
            waveform = resample(waveform, rate, model_map["sample_rate"])
            errors = investigate(model, sp, model_map, waveform, tokens(row["reference"], case), positions, args.device)
            stream.write(json.dumps({"utt_id": row["utt_id"], "errors": errors}, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            stream.flush()
            if count % 10 == 0:
                print(f"Diagnosed {count} of {len(todo)}", flush=True)


def summarize(args, selected, blank: int) -> dict:
    new = [json.loads(line) for line in (args.out_dir / "findings.jsonl").read_text(encoding="utf-8").splitlines()
           if line.strip()]
    if {item["utt_id"] for item in new} != {row["utt_id"] for row, _ in selected}:
        raise ValueError("Findings do not cover every selected recording.")
    if args.extra:
        return {"interpretation": INTERPRETATION, "recordings": {"extra_diagnostic": len(new)},
                "extra_diagnostic": summarize_findings(new, blank)}
    stored = json.loads(args.bug_report.read_text(encoding="utf-8"))["findings"]
    groups = {"bug_mining": reclassify(stored), "other_development": new}
    groups["pooled"] = groups["bug_mining"] + groups["other_development"]
    return {
        "interpretation": INTERPRETATION,
        "recordings": {name: len(items) for name, items in groups.items()},
        **{name: summarize_findings(items, blank) for name, items in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--bug-report", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--out-dir", type=Path, help="Default artifacts/pool_near_miss, or artifacts/extra_near_miss with --extra.")
    parser.add_argument("--extra", action="store_true", help="Diagnose the diagnostic part of the extra pool instead.")
    parser.add_argument("--extra-dir", type=Path, default=ROOT / "data/manifests_extra")
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, help="Diagnose at most this many recordings (smoke test).")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = ROOT / ("artifacts/extra_near_miss" if args.extra else "artifacts/pool_near_miss")

    config, case = settings()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["decode"]["threads"])
    selected = extra_targets(args.extra_dir, args.split_lock, config, case) if args.extra \
        else targets(args.manifests, config, case)
    print(f"Spans to diagnose: {sum(len(positions) for _, positions in selected)}", flush=True)
    model, sp = load_model(args.model_dir, args.icefall, model_map, args.device)
    diagnose(args, selected[:args.limit] if args.limit else selected, model, sp, model_map, case)
    if args.limit:
        return
    summary = summarize(args, selected, model_map["blank_id"])
    path = args.out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "interpretation"}, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
