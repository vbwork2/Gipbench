"""Evaluate exported FP32 and per-channel INT8 VE01 checkpoints against pretrained VietNote on Windows Sherpa-ONNX."""

import argparse
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from gipbench.bench.evaluate import score_rows, scorer_code_sha256
from gipbench.core import release as R
from gipbench.core.evalstats import paired_bootstrap
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.normprofiles import get_profile
from gipbench.core.paths import REPO
from gipbench.core.textnorm import Catalog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietnote_diag_ft_eval import score_records  # noqa: E402
from vietnote_parity import decode  # noqa: E402


def build_stratified_vi_dev(out_path: Path, per_band: int = 100, seed: int = 0):
    rel = R.resolve("dev/dev_v2_vi@1")
    rows = read_jsonl(rel.manifest)
    bands = [
        [r for r in rows if float(r["duration"]) < 10.0],
        [r for r in rows if 10.0 <= float(r["duration"]) <= 20.0],
        [r for r in rows if float(r["duration"]) > 20.0],
    ]
    picked = []
    for band in bands:
        ordered = sorted(band, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        picked.extend(ordered[:per_band])
    picked.sort(key=lambda r: r["id"])
    write_jsonl(out_path, picked)
    return picked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ve01-export", required=True, type=Path)
    parser.add_argument("--authorized-dev", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    catalog_path = R.resolve("catalog/train-v1").catalog
    catalog = Catalog.load(str(catalog_path))
    profile = get_profile("gipbench_strict_v1")

    strat_path = args.out / "dev_v2_vi_strat300.jsonl"
    strat_rows = build_stratified_vi_dev(strat_path, per_band=100, seed=0)

    datasets = {
        "authorized_dev": (read_jsonl(args.authorized_dev), ("offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8")),
        "vi_quick": (read_jsonl(REPO / "data/work/vietnote/dev/vi_quick.jsonl"), ("offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8")),
        "interim_cs_quick": (read_jsonl(REPO / "data/work/vietnote/dev/interim_cs_quick.jsonl"), ("offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8")),
        "dev_interim_cs": (read_jsonl(R.resolve("dev/dev_interim_cs@1").manifest), ("offline_fp32", "offline_int8")),
        "dev_v2_vi_strat300": (strat_rows, ("offline_fp32", "offline_int8")),
    }

    base = REPO / "models/vietnote78"
    models = {
        "pretrained": {
            "offline_fp32": base / "offline/fp32",
            "offline_int8": base / "offline/int8",
            "streaming_fp32": base / "streaming/fp32",
            "streaming_int8": base / "streaming/int8",
        },
        "ve01": {
            "offline_fp32": args.ve01_export / "offline",
            "offline_int8": args.ve01_export / "offline",
            "streaming_fp32": args.ve01_export / "streaming",
            "streaming_int8": args.ve01_export / "streaming",
        },
    }

    results = {}
    scored_recs = {}
    for model_name, dirs in models.items():
        results[model_name] = {}
        scored_recs[model_name] = {}
        model_out = args.out / model_name
        model_out.mkdir(parents=True, exist_ok=True)
        for ds_name, (rows, idents) in datasets.items():
            results[model_name][ds_name] = {}
            scored_recs[model_name][ds_name] = {}
            for ident in idents:
                mode, precision = ident.split("_")
                hyp_file = model_out / f"{ds_name}_{ident}.jsonl"
                records = decode(mode, precision, dirs[ident], rows, hyp_file)
                results[model_name][ds_name][ident] = score_records(rows, records, catalog, profile)
                hyps = {r["id"]: r["hyp"] for r in records}
                scored_recs[model_name][ds_name][ident] = score_rows(
                    rows, hyps, catalog, profile, "text_tagged", "speaker", []
                )
                s = results[model_name][ds_name][ident]
                print(
                    f"[{model_name}] {ds_name} {ident}: wer={s['wer']:.2f}% "
                    f"cs_wer={s.get('cs_wer')} recall={s.get('term_recall')} "
                    f"S/D/I={s['S']}/{s['D']}/{s['I']}",
                    flush=True,
                )

    paired = {}
    for ds_name, metrics in (
        ("dev_interim_cs", ("wer", "cs_wer", "term_recall", "n_wer", "en_hallucination_per100")),
        ("dev_v2_vi_strat300", ("wer", "cer", "en_hallucination_per100")),
    ):
        paired[ds_name] = {}
        for ident in ("offline_fp32", "offline_int8"):
            paired[ds_name][ident] = {
                m: paired_bootstrap(
                    scored_recs["pretrained"][ds_name][ident],
                    scored_recs["ve01"][ds_name][ident],
                    m, "unit", iters=1000, seed=0,
                )
                for m in metrics
            }

    export_meta = json.loads((args.ve01_export / "export.json").read_text(encoding="utf-8"))
    summary = {
        "scope": "exploratory_ve01_not_v03_no_locked_test",
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sherpa-onnx": version("sherpa-onnx"),
        },
        "ve01_export_sha256": sha256_file(args.ve01_export / "export.json"),
        "ve01_checkpoint_sha256": export_meta["checkpoint_sha256"],
        "dataset_sha256": {name: manifest_sha256(rows) for name, (rows, _) in datasets.items()},
        "dataset_rows": {name: len(rows) for name, (rows, _) in datasets.items()},
        "scorer_code_sha256": scorer_code_sha256(),
        "catalog_sha256": R.sha256(catalog_path),
        "models": results,
        "paired_vs_pretrained": paired,
    }
    eval_file = args.out / "evaluation.json"
    eval_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {eval_file}", flush=True)


if __name__ == "__main__":
    main()
