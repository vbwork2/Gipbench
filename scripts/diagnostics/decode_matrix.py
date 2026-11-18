"""Compare decode settings on the same locked utterances with unchanged model weights."""

import argparse
import json
from pathlib import Path

import yaml

from gipbench.bench import evaluate, registry, transcribe

BENCHES = (("general_v1", "wer"), ("code_switch_v1", "cs_wer"), ("cs_ggroup_v1", "cs_wer"))


def write_matrix(out, existing):
    lines = [
        "# Decode comparison",
        "",
        "| System | G0 WER | C0 CS-WER | C1 CS-WER |",
        "| --- | ---: | ---: | ---: |",
    ]
    lm_grid = sorted({p.name for p in out.glob("*/gip15_mbs4_lm*") if p.is_dir()}) or ["gip15_mbs4_lm"]
    for system in ("gip15_greedy_open", "gip15_mbs4_open", *lm_grid, "viegem_greedy",
                   "viegem_beam", "viegem_lm", "whisper_large_v3", "qwen3_asr_0.6b"):
        cells = []
        for bench, metric in BENCHES:
            for root in (out, existing):
                path = root / bench / system / "summary.json"
                if path.is_file():
                    result = json.loads(path.read_text(encoding="utf-8"))
                    value = result["profiles"]["gipbench_strict_v1"]["overall"][metric]
                    cells.append(f"{value:.2f}" if value is not None else "N/A")
                    break
            else:
                cells.append("Pending")
        lines.append("| " + " | ".join([system, *cells]) + " |")
    (out / "matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", choices=("gipformer", "viegem"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--gipformer-lm-config", type=Path, nargs="+",
                        help="một hoặc nhiều YAML system gip15_mbs4_lm* (mỗi file một lm_scale; báo cáo cả lưới)")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    results = registry.P.path("results_root")
    existing = results / "general_asr"
    out = results / "phase4" / "decode"
    out.mkdir(parents=True, exist_ok=True)
    if args.report_only:
        write_matrix(out, existing)
        return
    names = ("gip15_greedy_open",) if args.family == "gipformer" else ("viegem_beam", "viegem_lm")
    systems = {name: Path("configs/bench/asr_models") / f"{name}.yaml" for name in names}
    if args.gipformer_lm_config:
        if args.family != "gipformer":
            parser.error("--gipformer-lm-config requires the gipformer family")
        beam = transcribe.load_model_cfg("configs/bench/asr_models/gip15_mbs4_open.yaml")
        for lm_config in args.gipformer_lm_config:
            cfg = transcribe.load_model_cfg(str(lm_config))
            if (not cfg["system"].startswith("gip15_mbs4_lm") or cfg["inference_mode"] != "lm"
                    or not cfg["decode"].get("lm")):
                parser.error("Expected system gip15_mbs4_lm* with decode.lm and inference_mode lm")
            for key in ("engine", "model", "batch"):
                if cfg[key] != beam[key]:
                    parser.error(f"Keep {key} identical to gip15_mbs4_open")
            settings = {key: value for key, value in cfg["decode"].items() if key not in ("lm", "lm_scale")}
            if settings != beam["decode"]:
                parser.error("Keep all non-LM decoder settings identical to gip15_mbs4_open")
            if cfg["system"] in systems:
                parser.error(f"Duplicate system {cfg['system']}")
            systems[cfg["system"]] = lm_config
    for bench in ("cs_ggroup_v1", "code_switch_v1", "general_v1"):
        for system, src_cfg in systems.items():
            config = src_cfg
            if args.family == "gipformer":
                model = transcribe.load_model_cfg(str(src_cfg))
                del model["_path"]
                model["decode"]["threads"] = args.threads
                config = out / f"{system}.yaml"
                config.write_text(yaml.safe_dump(model), encoding="utf-8")
            dst = out / bench / system
            if not (dst / "hyps.jsonl").is_file():
                transcribe.main(["--model", str(config), "--benchmark", bench, "--out", str(dst)])
            if system == "viegem_lm":
                base = out / bench / "viegem_beam"
            else:
                base = existing / bench / ("gip15_mbs4_open" if args.family == "gipformer" else "viegem_greedy")
            if not (dst / "summary.json").is_file():
                result = evaluate.main(["--benchmark", bench, "--hyps", str(dst / "hyps.jsonl"),
                                        "--baseline-hyps", str(base / "hyps.jsonl")])
                if not result["hyp_check"]["ok"] or not result["baseline_hyp_check"]["ok"]:
                    raise SystemExit(f"Incomplete hypothesis comparison for {bench}/{system}")
            write_matrix(out, existing)


if __name__ == "__main__":
    main()
