"""Evaluate frozen ONNX configurations once on the sealed FPT holdout."""

import argparse
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
    terms = config["text"]["cs_terms"]
    results = {}
    for name, system in systems.items():
        cached = args.run_dir / f"{name}.jsonl"
        if cached.is_file():
            entries = read_rows(cached)
            if [row["utt_id"] for row in entries] != [row["utt_id"] for row in cases + clean]:
                raise ValueError(f"Incomplete cached acceptance scores for {name}.")
            scores = [row["score_counts"] for row in entries]
        else:
            decode_config = {**config, "decode": {**config["decode"], "quantize": system["quantize"]}}
            hotwords = Path(system["hotwords"]) if system.get("hotwords") else None
            rec = recognizer(Path(system["model_dir"]), args.run_dir, decode_config, case,
                             system["method"], hotwords, system.get("hotwords_score"))
            predictions = decode_rows(cases + clean, args.data_dir, rec, config["decode"]["batch_size"])
            scores = [score(row["reference"], predictions[row["audio_idx"]], case, terms) for row in cases + clean]
            temp = cached.with_suffix(".part")
            write_rows(temp, ({"utt_id": row["utt_id"], "score_counts": counts}
                              for row, counts in zip(cases + clean, scores, strict=True)))
            temp.replace(cached)
        results[name] = scores[:len(cases)], scores[len(cases):]

    base_held, base_clean = results[baseline]
    candidate_held, candidate_clean = results[primary]
    decisions = gates(base_held, candidate_held, base_clean, candidate_clean, frozen["candidate_evidence"], config["seed"])
    status = "PASS" if all(value["pass"] for value in decisions.values()) else "FAIL"
    lines = [
        "# Final acceptance",
        "",
        f"Primary candidate {primary} compared with {baseline}. Overall result **{status}**.",
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
    for gate, value in decisions.items():
        evidence = ", ".join(f"{key}={number}" for key, number in value.items() if key != "pass")
        lines.append(f"| {gate} | {'PASS' if value['pass'] else 'FAIL'} | {evidence} |")
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
    run(parser.parse_args())


if __name__ == "__main__":
    main()
