"""Run the registered G0 acceptance check for one X0 seed."""

import argparse
import json
from pathlib import Path

import yaml

from gipbench.bench import evaluate, registry, transcribe
from gipbench.core.paths import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--baseline-hyps", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    card_path = artifact / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    if card["system"] != "e4_x0" or card["select"]["rule"] != "fixed_last" or not card["leak_gate"]["passed"]:
        parser.error("Expected an X0 artifact selected with fixed_last and a passing leakage report")
    for name, expected in card["files"].items():
        if registry.file_sha256(artifact / "onnx" / name) != expected:
            parser.error(f"Artifact hash mismatch: {name}")
    baseline_hash = registry.file_sha256(args.baseline_hyps)
    if baseline_hash is None:
        parser.error("The registered G0 Gipformer baseline hypotheses are missing")
    plan = {"artifact_sha256": registry.file_sha256(card_path), "baseline_sha256": baseline_hash,
            "seed": card["seed"], "benchmark": "general_v1", "max_delta": 0.3, "max_ci95_upper": 0.5}
    args.out.mkdir(parents=True, exist_ok=True)
    registration = args.out / "plan.json"
    if registration.exists() and json.loads(registration.read_text(encoding="utf-8")) != plan:
        parser.error("This output directory belongs to a different artifact or baseline")
    registration.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    result_file = args.out / "acceptance.json"
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding="utf-8"))
    else:
        model = transcribe.load_model_cfg(str(REPO / "configs/bench/asr_models/gip15_mbs4_open.yaml"))
        del model["_path"]
        model["system"] = f"e4_x0_seed{card['seed']}"
        model["model"] = {"id": model["system"], "dir": str(artifact / "onnx")}
        config = args.out / "model.yaml"
        config.write_text(yaml.safe_dump(model), encoding="utf-8")
        if not (args.out / "hyps.jsonl").is_file():
            transcribe.main(["--model", str(config), "--benchmark", plan["benchmark"], "--out", str(args.out)])
        summary = evaluate.main(["--benchmark", plan["benchmark"], "--hyps", str(args.out / "hyps.jsonl"),
                                 "--mode", "final", "--baseline-hyps", str(args.baseline_hyps)])
        wer = next(metric for metric in summary["paired"]["overall"] if metric["metric"] == "wer")
        result = {**plan, "wer_delta": wer["delta"], "ci95": wer["ci95"],
                  "passed": bool(wer["ci95"] and wer["delta"] <= plan["max_delta"] + 1e-9
                                 and wer["ci95"][1] <= plan["max_ci95_upper"] + 1e-9)}
        result_file.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
