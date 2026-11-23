"""Check decoder parity and diagnose errors from the locked development split."""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import soundfile
import torch

from gipformer_mech.fpt import audio_indices, manifest_waveform, parquet_files, recognizer, settings, transcribe
from gipformer_mech.logit_lens import investigate
from gipformer_mech.metrics import score
from gipformer_mech.rnnt_loader import encode, greedy, load_model, resample
from gipformer_mech.splits import development_rows
from gipformer_mech.text_norm import tokens


def write_report(path: Path, report: dict, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    format_options = {"separators": (",", ":")} if compact else {"indent": 2}
    path.write_text(json.dumps(report, ensure_ascii=False, **format_options) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


@torch.inference_mode()
def parity(args, model, sp, model_map, config, case) -> dict:
    files = sorted(args.calibration_dir.glob("*.wav"))
    if len(files) < 10:
        raise ValueError("Decoder parity requires at least ten independent calibration WAV files.")
    sample = Random(config["seed"]).sample(files, 10)
    comparison = {**config, "decode": {**config["decode"], "quantize": args.parity_quantize}}
    rec = recognizer(args.model_dir, args.parity_report.parent, comparison, case, "greedy_search")
    matches = []
    for path in sample:
        waveform, rate = soundfile.read(path, dtype="float32", always_2d=True)
        waveform = resample(waveform.mean(axis=1), rate, model_map["sample_rate"])
        output, lengths = encode(model, waveform, model_map, args.device)
        torch_hyp = greedy(model, output, lengths, sp)
        onnx_hyp, = transcribe(rec, [(waveform, model_map["sample_rate"])])
        matches.append({
            "file": path.name,
            "pytorch": torch_hyp,
            "onnx": onnx_hyp,
            "exact_match": torch_hyp.strip() == onnx_hyp.strip(),
        })
    rate = sum(item["exact_match"] for item in matches) / len(matches)
    return {"passed": rate > 0.99, "exact_match_rate": rate, "quantization": args.parity_quantize, "samples": matches}


def diagnose(args, model, sp, model_map, config, case) -> dict:
    rows = development_rows(args.manifests, "bug_mining")
    by_index = {row["audio_idx"]: row for row in rows}
    findings = []
    for index, raw in audio_indices(parquet_files(args.data_dir), set(by_index)):
        row = by_index[index]
        waveform, rate = manifest_waveform(row, raw)
        waveform = resample(waveform, rate, model_map["sample_rate"])
        words = tokens(row["reference"], case)
        counts = score(row["reference"], row["baseline_hyp"], case, config["text"]["cs_terms"])
        positions = [error["ref_index"] for error in counts["cs_word_errors"]]
        errors = investigate(model, sp, model_map, waveform, words, positions, args.device) if positions else []
        findings.append({"utt_id": row["utt_id"], "errors": errors})
    if len(findings) != len(rows):
        raise ValueError("Missing audio for at least one bug-mining record.")
    categories = Counter(error["category"] for row in findings for error in row["errors"])
    total = sum(categories.values())
    if not total:
        raise ValueError("No aligned CS substitution or deletion errors in bug_mining.")
    share = {key: count / total for key, count in categories.items()}
    return {
        "error_span_count": total,
        "category_counts": dict(categories),
        "category_share": share,
        "reference_prefix_category_counts": dict(Counter(
            error["reference_prefix_category"] for row in findings for error in row["errors"]
        )),
        "predictor_context": "Tokens actually emitted by PyTorch greedy search before each probed frame.",
        "decision": {
            "prioritize_hotwords_and_joiner": share.get("PREDICTOR_OVERRIDE_BUG", 0) >= 0.30,
            "allow_encoder_sweep": share.get("ENCODER_BUG", 0) >= 0.25,
            "prepare_finetune": share.get("TRUE_GAP", 0) > 0.70,
        },
        "interpretation": (
            "TRUE_GAP records a tokenizer unknown or absence from the measured top ten. "
            "It is an operational label, not proof that the acoustic information is absent."
        ),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--calibration-dir", type=Path, required=True, help="WAVs outside the FPT evaluation pool.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parity-quantize", choices=("fp32", "int8"), default="fp32")
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--parity-report", type=Path, default=ROOT / "artifacts/phase2_parity.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    args = parser.parse_args()

    config, case = settings()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["decode"]["threads"])
    model, sp = load_model(args.model_dir, args.icefall, model_map, args.device)
    check = parity(args, model, sp, model_map, config, case)
    write_report(args.parity_report, check)
    if not check["passed"]:
        parser.exit(1, f"Parity gate failed at {check['exact_match_rate']:.0%} exact transcript match.\n")
    if not args.parity_only:
        write_report(args.report, diagnose(args, model, sp, model_map, config, case), compact=True)


if __name__ == "__main__":
    main()
