"""Decode the historical E3 candidates on C1 and the remaining G0 G-Group utterances."""

import argparse
import json
from pathlib import Path

import yaml

from gipbench.bench import evaluate, registry, transcribe
from gipbench.core.manifest import read_jsonl, write_jsonl


def prepare_vi_slice(out, c1_rows):
    g0 = registry.load_benchmark("general_v1")
    path = registry.expand(g0["manifest"])
    rows = read_jsonl(path)
    if error := registry.check_frozen(g0, rows):
        raise SystemExit(error)
    c1_sources = {row["source"] for row in c1_rows}
    selected = [dict(row, audio=row["_audio_path"]) for row in rows
                if row["dataset"].startswith("ggroup_") and row["source"] not in c1_sources]
    if len(selected) != 787:
        raise SystemExit(f"Expected 787 G-Group rows outside C1, found {len(selected)}")
    manifest = out / "g0_vi.jsonl"
    write_jsonl(manifest, selected)
    return manifest


def write_comparison(out, history, candidates):
    old = {}
    for kind in ("ckpt", "wise"):
        old.update(json.loads((history / f"select_{kind}.json").read_text(encoding="utf-8"))["results"])
    lines = [
        "# E3 v1 checkpoint comparison",
        "",
        "| Candidate | Old dev CS-WER | Old dev VI WER | C1 CS-WER | G0 complement WER |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for candidate in candidates:
        alias = out / f"{candidate}.alias.json"
        name = json.loads(alias.read_text(encoding="utf-8"))["same_model_as"] if alias.exists() else candidate
        values = [f"{old[candidate][key]:.2f}" for key in ("cs_wer", "wer_vi")]
        for bench, metric in (("c1", "cs_wer"), ("g0_vi", "wer")):
            summary = out / name / bench / "summary.json"
            if summary.exists():
                result = json.loads(summary.read_text(encoding="utf-8"))
                value = result["profiles"]["gipbench_strict_v1"]["overall"][metric]
                values.append(f"{value:.2f}")
            else:
                values.append("Pending")
        lines.append("| " + " | ".join([candidate, *values]) + " |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/phase4/checkpoints"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    history = registry.P.path("results_root") / "e3_v1_dev"
    candidates = sorted(p.name for p in history.iterdir() if (p / "dev_cs/hyps.jsonl").is_file())
    # Prepare the baseline, then test the historically selected checkpoint.
    candidates = ["base", "ck4132", *[name for name in candidates if name not in ("base", "ck4132")]]
    if args.report_only:
        write_comparison(out, history, candidates)
        return
    c1 = registry.load_benchmark("cs_ggroup_v1")
    c1_rows = read_jsonl(registry.expand(c1["manifest"]))
    if error := registry.check_frozen(c1, c1_rows):
        raise SystemExit(error)
    vi = prepare_vi_slice(out, c1_rows)

    slice_ids = {"c1": {row["id"] for row in c1_rows}, "g0_vi": {row["id"] for row in read_jsonl(vi)}}
    base = transcribe.load_model_cfg("configs/bench/asr_models/gip15_mbs4_open.yaml")
    base.pop("_path")
    decoded = {}
    for candidate in candidates:
        model_dir = (Path(base["model"]["dir"]) if candidate == "base" else
                     registry.P.path("exp_root") / "e3_v1" / "cand" / candidate / "onnx")
        model_hashes = tuple(registry.file_sha256(model_dir / file) for file in
                             ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt"))
        if None in model_hashes:
            raise SystemExit(f"Missing ONNX or tokens.txt in {model_dir}")
        previous = decoded.get(model_hashes)
        if previous:
            (out / f"{candidate}.alias.json").write_text(
                json.dumps({"same_model_as": previous, "sha256": model_hashes}, indent=2) + "\n",
                encoding="utf-8")
            write_comparison(out, history, candidates)
            continue
        decoded[model_hashes] = candidate
        config = out / f"{candidate}.yaml"
        cfg = dict(base, system=f"e3_v1_{candidate}", model={**base["model"], "dir": str(model_dir)},
                   decode={**base["decode"], "threads": args.threads})
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        for name, flags in (("c1", ["--benchmark", "cs_ggroup_v1"]), ("g0_vi", ["--manifest", str(vi)])):
            dst = out / candidate / name
            hyps = dst / "hyps.jsonl"
            if candidate == "base":
                # Reuse the locked-baseline decode with the same INT8 beam and batch size.
                benchmark = "cs_ggroup_v1" if name == "c1" else "general_v1"
                original = (registry.P.path("results_root") / "general_asr" / benchmark /
                            "gip15_mbs4_open" / "hyps.jsonl")
                meta = json.loads(original.with_name("run_meta.json").read_text(encoding="utf-8"))
                hashes = tuple(meta["extra"]["model_files"][part]["sha256"]
                               for part in ("encoder", "decoder", "joiner"))
                if hashes != model_hashes[:3] or meta["batch_size"] != base["batch"] or any(
                        meta["decoding"][key] != base["decode"][key]
                        for key in ("quantize", "decoding_method", "beam", "blank_penalty")):
                    raise SystemExit(f"Unexpected baseline decoder settings in {original.parent}")
                ids = slice_ids[name]
                cached = [row for row in read_jsonl(original) if row["id"] in ids]
                if len(cached) != len(ids) or {row["id"] for row in cached} != ids:
                    raise SystemExit(f"Incomplete baseline hypotheses in {original}")
                write_jsonl(hyps, cached)
            elif not hyps.exists():
                transcribe.main(["--model", str(config), *flags, "--out", str(dst)])
            if not (dst / "summary.json").exists():
                paired = [] if candidate == "base" else [
                    "--baseline-hyps", str(out / "base" / name / "hyps.jsonl")]
                result = evaluate.main([*flags, "--hyps", str(hyps), "--system", cfg["system"],
                                        "--inference-mode", "open_vocab", *paired])
                if not result["hyp_check"]["ok"] or (paired and not result["baseline_hyp_check"]["ok"]):
                    raise SystemExit(f"Incomplete hypothesis comparison for {candidate}/{name}")
            write_comparison(out, history, candidates)


if __name__ == "__main__":
    main()
