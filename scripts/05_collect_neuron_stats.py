"""Collect bounded FFN activation summaries after the encoder diagnosis gate."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from random import Random
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from gipformer_mech.fpt import audio_indices, parquet_files, read_rows, read_waveform, settings
from gipformer_mech.model_audit import checksum
from gipformer_mech.rnnt_loader import encode, load_model, resample


def run(args) -> None:
    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    if not diagnosis["decision"]["allow_encoder_sweep"]:
        raise ValueError("Phase 2 did not meet the encoder sweep threshold.")
    lock = json.loads((args.manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    pool = args.manifests / "full_scored.jsonl"
    if checksum(pool) != lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    config, _ = settings()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["decode"]["threads"])
    random = Random(config["seed"])
    eligible = [row for row in read_rows(pool) if row["baseline_wer"] == 0 and not row["cs_tokens"]]
    selected = {row["audio_idx"]: row for row in random.sample(eligible, args.clean_count)}
    model, _ = load_model(args.model_dir, args.icefall, model_map, args.device)

    residuals = defaultdict(list)
    activations = defaultdict(list)
    handles = []

    def record_residual(_module, _inputs, output, stack):
        residuals[stack].append(output[:, 0].float().norm(dim=-1).cpu().numpy())

    def record_activation(_module, inputs, key):
        frames = inputs[0][:, 0]
        indices = random.sample(range(len(frames)), min(4, len(frames)))
        activations[key].append(frames[indices].float().cpu().numpy())

    column_norms = {}
    for stack in model_map["encoder_stacks"][-2:]:
        module = model.get_submodule(stack["module"])
        handles.append(module.register_forward_hook(
            lambda module, inputs, output, key=stack["module"]: record_residual(module, inputs, output, key)
        ))
        for name in stack["feed_forward"]:
            projection = model.get_submodule(f"{name}.out_proj")
            column_norms[f"{name}.out_column_l2"] = projection.weight.detach().float().norm(dim=0).cpu().numpy()
            handles.append(projection.register_forward_pre_hook(
                lambda module, inputs, key=name: record_activation(module, inputs, key)
            ))
    try:
        seen = 0
        with torch.inference_mode():
            for index, raw in audio_indices(parquet_files(args.data_dir), set(selected)):
                if raw["audio"]["path"] != selected[index]["audio_file"]:
                    raise ValueError(f"Audio filename mismatch for FPT dataset row {index}.")
                waveform, rate = read_waveform(raw["audio"]["bytes"])
                waveform = resample(waveform, rate, model_map["sample_rate"])
                encode(model, waveform, model_map, args.device)
                seen += 1
        if seen != args.clean_count:
            raise ValueError("Some activation-analysis utterances are missing from the FPT snapshot.")
    finally:
        for handle in handles:
            handle.remove()

    result = {**column_norms, "utterances": np.array(seen), "seed": np.array(config["seed"])}
    for name, values in residuals.items():
        result[f"{name}.median_residual_norm"] = np.median(np.concatenate(values))
    for name, frames in activations.items():
        values = np.concatenate(frames)
        for sign, mask in (("positive", values > 0), ("negative", values < 0)):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                percentiles = np.nanpercentile(np.where(mask, values, np.nan), [1, 50, 99], axis=0)
            result[f"{name}.{sign}_percentiles"] = percentiles.astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"Wrote {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--clean-count", type=int, default=750, choices=range(500, 1001), metavar="500-1000")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/stats/encoder_stats.npz")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
