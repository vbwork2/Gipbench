"""Evaluate exported FP32 and per-channel INT8 checkpoints from diag-ft on Windows Sherpa-ONNX."""

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

from gipbench.bench.evaluate import score_rows, scorer_code_sha256, summarize
from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.normprofiles import get_profile
from gipbench.core.paths import REPO
from gipbench.core.textnorm import Catalog

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vietnote_parity import decode  # noqa: E402


def _band(seconds):
    if seconds < 10.0:
        return "<10s"
    if seconds <= 20.0:
        return "10-20s"
    return ">20s"


def score_records(rows, records, catalog, profile):
    hyps = {r["id"]: r["hyp"] for r in records}
    recs = score_rows(rows, hyps, catalog, profile, "text_tagged", "speaker", [])
    overall = summarize(recs, 0, 0, with_ci=False)
    overall.update({k: sum(r[k] for r in recs) for k in ("S", "D", "I")})
    row_by_id = {r["id"]: r for r in rows}
    bands = {}
    for band in ("<10s", "10-20s", ">20s"):
        sub = [r for r in recs if _band(float(row_by_id[r["id"]]["duration"])) == band]
        if sub:
            s = summarize(sub, 0, 0, with_ci=False)
            s.update({k: sum(r[k] for r in sub) for k in ("S", "D", "I")})
            bands[band] = s
    overall["by_duration_band"] = bands
    if records and "emissions" in records[0]:
        long = [r for r in records if float(r["duration"]) >= 20.0]
        overall["streaming_long"] = {
            "long_rows": len(long),
            "with_emissions": sum(1 for r in long if r["emissions"]),
            "changed_after_halfway": sum(1 for r in long if r["changed_after_halfway"]),
            "lost_late_emissions": sum(1 for r in long if not r["changed_after_halfway"]),
            "max_first_emission_sec": max((r["emissions"][0]["audio_sec"] for r in long if r["emissions"]), default=None),
        }
    return overall


def evaluate_model(label, model_dirs, datasets, out_dir, catalog, profile):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for ds_name, rows in datasets.items():
        result[ds_name] = {}
        for mode in ("offline", "streaming"):
            for precision in ("fp32", "int8"):
                ident = f"{mode}_{precision}"
                model_dir = model_dirs[ident]
                hyp_file = out_dir / f"{ds_name}_{ident}.jsonl"
                records = decode(mode, precision, model_dir, rows, hyp_file)
                result[ds_name][ident] = score_records(rows, records, catalog, profile)
                print(f"[{label}] {ds_name} {ident}: wer={result[ds_name][ident]['wer']:.2f}% "
                      f"cs_wer={result[ds_name][ident].get('cs_wer')} "
                      f"long={result[ds_name][ident].get('streaming_long')}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-export", required=True, type=Path)
    parser.add_argument("--extended-export", required=True, type=Path)
    parser.add_argument("--diag-dev", required=True, type=Path, help="Source-disjoint authorized dev manifest")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    catalog_path = R.resolve("catalog/train-v1").catalog
    catalog = Catalog.load(str(catalog_path))
    profile = get_profile("gipbench_strict_v1")

    datasets = {
        "authorized_dev": read_jsonl(args.diag_dev),
        "vi_quick": read_jsonl(REPO / "data/work/vietnote/dev/vi_quick.jsonl"),
        "interim_cs_quick": read_jsonl(REPO / "data/work/vietnote/dev/interim_cs_quick.jsonl"),
    }

    base = REPO / "models/vietnote78"
    models = {
        "pretrained": {
            "offline_fp32": base / "offline/fp32",
            "offline_int8": base / "offline/int8",
            "streaming_fp32": base / "streaming/fp32",
            "streaming_int8": base / "streaming/int8",
        },
        "control_20s": {
            "offline_fp32": args.control_export / "offline",
            "offline_int8": args.control_export / "offline",
            "streaming_fp32": args.control_export / "streaming",
            "streaming_int8": args.control_export / "streaming",
        },
        "extended_45s": {
            "offline_fp32": args.extended_export / "offline",
            "offline_int8": args.extended_export / "offline",
            "streaming_fp32": args.extended_export / "streaming",
            "streaming_int8": args.extended_export / "streaming",
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "authorized_diagnostic_not_v03_no_locked_test",
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "sherpa-onnx": version("sherpa-onnx"),
        },
        "dataset_sha256": {name: manifest_sha256(rows) for name, rows in datasets.items()},
        "dataset_rows": {name: len(rows) for name, rows in datasets.items()},
        "scorer_code_sha256": scorer_code_sha256(),
        "catalog_sha256": R.sha256(catalog_path),
        "models": {},
    }
    for name, dirs in models.items():
        summary["models"][name] = evaluate_model(name, dirs, datasets, args.out / name, catalog, profile)

    (args.out / "evaluation.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out / 'evaluation.json'}", flush=True)


if __name__ == "__main__":
    main()
