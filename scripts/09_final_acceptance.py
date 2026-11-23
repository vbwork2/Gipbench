"""Evaluate frozen ONNX configurations once on the sealed FPT holdout."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gipbench.infer.decode import pick
from gipformer_mech.acceptance import gates
from gipformer_mech.fpt import decode_rows, read_rows, recognizer, settings, write_rows
from gipformer_mech.metrics import score, summarize
from gipformer_mech.model_audit import checksum
from gipformer_mech.splits import development_rows


def freeze_system(directory: Path, system: dict, default_quantization: str) -> dict:
    frozen = dict(system)
    model_dir = (directory / frozen["model_dir"]).resolve()
    quantization = frozen.get("quantize", default_quantization)
    files = [pick(model_dir, key, quantization) for key in ("encoder", "decoder", "joiner")]
    files.append(model_dir / "tokens.txt")
    frozen["model_dir"] = str(model_dir)
    frozen["quantize"] = quantization
    if frozen.get("hotwords"):
        if frozen["method"] != "modified_beam_search":
            raise ValueError("Hotwords require modified beam search.")
        hotwords = (directory / frozen["hotwords"]).resolve()
        frozen["hotwords"] = str(hotwords)
        files.extend([hotwords, model_dir / "bpe.model"])
        if (model_dir / "bpe.vocab").exists():
            files.append(model_dir / "bpe.vocab")
    frozen["sha256"] = {str(path): checksum(path) for path in files}
    return frozen


def evaluate_system(name: str, system: dict, cases: list[dict], clean: list[dict],
                    data_dir: Path, run_dir: Path, config: dict, case: str) -> tuple[list[dict], list[dict]]:
    rows = cases + clean
    cached = run_dir / f"{name}.jsonl"
    if cached.is_file():
        entries = read_rows(cached)
        if [row["utt_id"] for row in entries] != [row["utt_id"] for row in rows]:
            raise ValueError(f"Incomplete cached acceptance scores for {name}.")
        scores = [row["score_counts"] for row in entries]
    else:
        decode_config = {**config, "decode": {**config["decode"], "quantize": system["quantize"]}}
        hotwords = Path(system["hotwords"]) if system.get("hotwords") else None
        rec = recognizer(Path(system["model_dir"]), run_dir, decode_config, case,
                         system["method"], hotwords, system.get("hotwords_score"))
        predictions = decode_rows(rows, data_dir, rec, config["decode"]["batch_size"])
        terms = config["text"]["cs_terms"]
        scores = [score(row["reference"], predictions[row["audio_idx"]], case, terms) for row in rows]
        temp = cached.with_suffix(".part")
        write_rows(temp, ({"utt_id": row["utt_id"], "score_counts": counts}
                          for row, counts in zip(rows, scores, strict=True)))
        temp.replace(cached)
    return scores[:len(cases)], scores[len(cases):]


def run(args: argparse.Namespace) -> None:
    config, case = settings()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    systems = {
        name: freeze_system(args.plan.parent, system, config["decode"]["quantize"])
        for name, system in plan["systems"].items()
    }
    baseline, primary = plan["baseline"], plan["primary"]
    if baseline == primary or baseline not in systems or primary not in systems:
        raise ValueError("The plan must select different named baseline and primary systems.")
    if systems[baseline]["method"] != systems[primary]["method"]:
        raise ValueError("The baseline and primary systems must use the same decoding method.")
    if systems[baseline].get("hotwords"):
        raise ValueError("The acceptance baseline must be unbiased.")
    evaluation_kind = plan.get("evaluation_kind", "weight_candidate")
    if evaluation_kind not in {"weight_candidate", "decoder_only"}:
        raise ValueError("Unknown acceptance evaluation kind.")
    if evaluation_kind == "decoder_only":
        first, second = systems[baseline], systems[primary]
        if (
            first["model_dir"] != second["model_dir"]
            or first["quantize"] != second["quantize"]
            or not second.get("hotwords")
            or any(second["sha256"].get(path) != digest for path, digest in first["sha256"].items())
        ):
            raise ValueError("Decoder-only acceptance requires identical model files and a hotword candidate.")
    if any(not name.isidentifier() for name in systems):
        raise ValueError("System names must be valid identifiers.")
    lock = json.loads((args.manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    heldout = args.manifests / "heldout_test.jsonl"
    clean = development_rows(args.manifests, "clean_control")
    receipt = args.run_dir / "frozen_run.json"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    frozen = {
        "baseline": baseline,
        "primary": primary,
        "evaluation_kind": evaluation_kind,
        "systems": systems,
        "heldout_sha256": lock["heldout_sha256"],
        "pipeline_sha256": checksum(ROOT / "configs/pipeline.yaml"),
        "candidate_evidence": plan.get("candidate_evidence", {}),
    }
    if args.report.exists():
        raise FileExistsError("An acceptance report already exists.")
    if receipt.exists():
        if json.loads(receipt.read_text(encoding="utf-8")) != frozen:
            raise ValueError("An interrupted run can only resume with its frozen plan.")
    else:
        with receipt.open("x", encoding="utf-8") as output:
            json.dump(frozen, output, ensure_ascii=False, indent=2)
    if checksum(heldout) != lock["heldout_sha256"]:
        raise ValueError("Heldout differs from the sealed checksum.")
    cases = read_rows(heldout)
    if len(cases) != lock["heldout_size"] or lock["dataset_revision"] != config["dataset"]["revision"]:
        raise ValueError("Heldout size or dataset revision differs from the sealed metadata.")
    if {row["utt_id"] for row in cases} & {row["utt_id"] for row in clean}:
        raise ValueError("Heldout and clean-control utterances overlap.")
    with ProcessPoolExecutor(max_workers=min(args.jobs, len(systems))) as workers:
        evaluations = {
            name: workers.submit(evaluate_system, name, system, cases, clean, args.data_dir, args.run_dir, config, case)
            for name, system in systems.items()
        }
        results = {name: evaluation.result() for name, evaluation in evaluations.items()}

    base_held, base_clean = results[baseline]
    candidate_held, candidate_clean = results[primary]
    decisions = gates(base_held, candidate_held, base_clean, candidate_clean, frozen["candidate_evidence"], config["seed"])
    if evaluation_kind == "decoder_only":
        decisions["G5"] = {"pass": None, "scope": "Decoding-only comparison with identical weights"}
    status = "PASS" if all(value["pass"] is not False for value in decisions.values()) else "FAIL"
    lines = [
        "# Final acceptance",
        "",
        f"Primary candidate {primary} compared with {baseline}. Overall result **{status}**.",
        "",
        f"Heldout has {len(cases):,} utterances, including {lock['heldout_cs_count']:,} with CS reference tokens. "
        f"The split unit is {lock.get('split_unit', 'speaker')}.",
        *(
            ["This is a decoding-only evaluation on identical model weights."]
            if evaluation_kind == "decoder_only" else []
        ),
        "",
        "| System | Heldout WER | Heldout N-WER | Heldout CS-WER | Heldout insertion rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, (scores, _) in results.items():
        metric = summarize(scores)
        lines.append(
            f"| {name} | {metric['wer_global']:.2%} | {metric['n_wer']:.2%} | "
            f"{metric['cs_wer']:.2%} | {metric['ins_rate']:.2%} |"
        )
    lines.extend(["", "| Gate | Status | Evidence |", "|---|---|---|"])
    gain, clean_gate, insertion, vietnamese, causal = (decisions[f"G{index}"] for index in range(1, 6))
    lower, upper = gain["interval"]
    evidence_text = {
        "G1": (f"CS-WER gain {gain['gain'] * 100:+.2f} pp, p={gain['p_value']:.4f}, "
               f"95% CI [{lower * 100:+.2f}, {upper * 100:+.2f}] pp"),
        "G2": f"{clean_gate['utterances_degraded']}/{len(clean)} newly degraded clean utterances",
        "G3": f"Insertion rate delta {insertion['insertion_delta'] * 100:+.3f} pp",
        "G4": f"N-WER delta {vietnamese['n_wer_delta'] * 100:+.3f} pp",
        "G5": causal["scope"] if causal["pass"] is None else (
            f"{causal['random_control_count']} random controls, p95={causal['random_p95']}, "
            f"candidate fix rate={causal['candidate_fix_rate']}"
        ),
    }
    for gate, value in decisions.items():
        gate_status = "N/A" if value["pass"] is None else "PASS" if value["pass"] else "FAIL"
        lines.append(f"| {gate} | {gate_status} | {evidence_text[gate]} |")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print(f"Wrote {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Frozen candidate and baseline configuration JSON.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--run-dir", type=Path, default=ROOT / "artifacts/acceptance")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/final_acceptance.md")
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
