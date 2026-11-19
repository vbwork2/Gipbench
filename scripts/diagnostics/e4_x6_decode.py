"""Score a registered decoding grid with a fixed, previously selected E4 artifact."""

import argparse
import json
from pathlib import Path

import yaml

from gipbench.bench import evaluate, registry, transcribe
from gipbench.core.paths import REPO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, nargs="+")
    parser.add_argument("--lm", required=True, type=Path, help="ONNX LM trained on audited non-benchmark text")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    card = json.loads((artifact / "model_card.json").read_text(encoding="utf-8"))
    if not card["system"].startswith("e4_") or not card["leak_gate"]["passed"] or card.get("rankable") is False:
        parser.error("Select an E4 artifact with a passing leakage report")
    for name in ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"):
        if registry.file_sha256(artifact / "onnx" / name) != card["files"][name]:
            parser.error(f"Artifact hash mismatch: {name}")
    lm_hash = registry.file_sha256(args.lm)
    if lm_hash is None:
        parser.error(f"LM file is missing: {args.lm}")
    for name in args.benchmark:
        benchmark = registry.load_benchmark(name)
        if benchmark.get("evidence_level") != "locked" or not {"open_vocab", "lm"} <= set(benchmark["allowed_inference_modes"]):
            parser.error(f"{name} must be a locked benchmark that accepts open-vocabulary and LM decoding")

    cfg = yaml.safe_load((REPO / "configs/model/experiments/e4_x6.yaml").read_text(encoding="utf-8"))
    plan = {"artifact_sha256": registry.file_sha256(artifact / "model_card.json"), "lm_sha256": lm_hash,
            "benchmarks": args.benchmark, "threads": args.threads, "config": cfg}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=True)
    registration = args.out / "plan.json"
    if registration.exists() and json.loads(registration.read_text(encoding="utf-8")) != plan:
        parser.error(f"Use a fresh output directory for a different artifact, LM, or decode grid: {args.out}")
    registration.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    for benchmark in args.benchmark:
        paths = {}
        for variant, overrides in cfg["variants"].items():
            mode = "lm" if "lm_scale" in overrides else "open_vocab"
            decode = {**cfg["decode"], **overrides, "threads": args.threads}
            if mode == "lm":
                decode["lm"] = str(args.lm.resolve())
            model = {"system": f"{card['system']}_seed{card['seed']}_{variant}", "engine": "sherpa_transducer",
                     "model": {"id": card["system"], "dir": str(artifact / "onnx")}, "inference_mode": mode,
                     "decode": decode, "batch": cfg["batch"]}
            path = args.out / f"{variant}.yaml"
            path.write_text(yaml.safe_dump(model), encoding="utf-8")
            dst = args.out / Path(benchmark).stem / variant
            paths[variant] = dst
            if not (dst / "hyps.jsonl").is_file():
                transcribe.main(["--model", str(path), "--benchmark", benchmark, "--out", str(dst)])
        for variant, dst in paths.items():
            baseline = [] if variant == "beam" else ["--baseline-hyps", str(paths["beam"] / "hyps.jsonl")]
            evaluate.main(["--benchmark", benchmark, "--hyps", str(dst / "hyps.jsonl"), "--mode", "final", *baseline])


if __name__ == "__main__":
    main()
