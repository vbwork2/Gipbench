"""Isolate the INT8 parity difference on the fixed V01 subset."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

from gipbench.core import release as R
from gipbench.bench.evaluate import scorer_code_sha256
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.textnorm import Catalog
from vietnote_parity import PROTOCOL, ROOT, compare, decode

PARTS = ("encoder", "decoder", "joiner")


def compare_same_runtime(args, base, rows, cfg, catalog, catalog_sha256, evidence, exported):
    prior_path = args.out / "result.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    subset = manifest_sha256(rows)
    parity_sha = R.sha256(args.parity / "result.json")
    if (subset != evidence["dev_manifest_sha256"] or subset != prior["manifest_sha256"]
            or parity_sha != prior["parent_parity_sha256"] or len(rows) != 113):
        raise ValueError("The candidates and comparison must use the locked V01 subset")
    output = args.out / "same_runtime"
    output.mkdir(exist_ok=False)
    results, runtime_check = {}, {}
    for mode in ("offline", "streaming"):
        dest = output / mode
        dest.mkdir()
        names = [*(f"{part}.int8.onnx" for part in PARTS), "tokens.txt"]
        official_hashes = {name: evidence["official_files"][f"{mode}/int8/{name}"] for name in names}
        official = base / mode / "int8"
        if any(R.sha256(official / name) != digest for name, digest in official_hashes.items()):
            raise ValueError(f"Stale official INT8 artifact: {mode}")
        baseline = decode(mode, "int8", official, rows, dest / "official.jsonl")
        old = read_jsonl(args.parity / f"{mode}_int8_official.jsonl")
        runtime_check[mode] = compare(old, baseline, cfg, "int8", catalog, rows)
        candidates = {"requantized_official_fp32": "official_fp32_requantized",
                      "reexport_encoder": "reexport_encoder", "reexport_decoder_joiner": "reexport_decoder_joiner"}
        for name, folder in (*candidates.items(), ("full_reexport", None)):
            if folder is None:
                model = args.export / mode
                hashes = {file: exported["files"][f"{mode}/{file}"] for file in names}
            else:
                model = args.out / mode / folder
                hashes = {f"{part}.int8.onnx": digest for part, digest in
                          prior["results"][f"{mode}/{name}"]["components_sha256"].items()}
                hashes["tokens.txt"] = official_hashes["tokens.txt"]
            if any(R.sha256(model / file) != digest for file, digest in hashes.items()):
                raise ValueError(f"Stale INT8 comparison artifact: {mode}/{name}")
            hyps = decode(mode, "int8", model, rows, dest / f"{name}.jsonl")
            score = compare(baseline, hyps, cfg, "int8", catalog, rows)
            results[f"{mode}/{name}"] = dict(score, artifact_sha256=hashes)
            print(f"{mode}/{name}: {score['exact']}/{len(rows)} exact", flush=True)
    report = {"manifest_sha256": subset, "rows": len(rows), "parent_parity_sha256": parity_sha,
              "parent_isolation_sha256": R.sha256(prior_path), "catalog_sha256": catalog_sha256,
              "scorer_code_sha256": scorer_code_sha256(), "config_sha256": R.sha256(PROTOCOL),
              "runtime": {"platform": platform.platform(), "python": platform.python_version(),
                          "sherpa-onnx": version("sherpa-onnx"), "quantizer_onnxruntime": prior["onnxruntime_version"]},
              "platform_reference_comparison": runtime_check, "results": results,
              "outputs": {file.relative_to(output).as_posix(): R.sha256(file) for file in sorted(output.glob("*/*.jsonl"))}}
    (args.out / "matched_runtime.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-runtime", action="store_true",
                        help="Decode the official and candidate artifacts together on this runtime")
    args = parser.parse_args()
    base = ROOT / "models/vietnote78"
    quantizer = base / "upstream/scripts/quantize_int8.py"
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))["parity"]
    evidence = json.loads((args.parity / "result.json").read_text(encoding="utf-8"))
    if R.sha256(args.export / "export.json") != evidence["export_metadata_sha256"]:
        raise ValueError("The requested export changed")
    if evidence["config_sha256"] != R.sha256(PROTOCOL) or evidence["scorer_code_sha256"] != scorer_code_sha256():
        raise ValueError("The frozen V01 configuration or scorer changed")
    export = json.loads((args.export / "export.json").read_text(encoding="utf-8"))
    if R.sha256(quantizer) != export["quantizer_sha256"]:
        raise ValueError("The upstream quantizer changed")
    for name, digest in evidence["outputs"].items():
        if R.sha256(args.parity / name) != digest:
            raise ValueError(f"Stale parity artifact: {name}")
    rows = read_jsonl(args.parity / "dev.jsonl")
    for row in rows:
        if R.sha256(row["_audio_path"]) != row["audio_sha256"]:
            raise ValueError(f"Audio digest mismatch: {row['id']}")
    catalog_path = R.resolve("catalog/train-v1").catalog
    catalog_sha = R.sha256(catalog_path)
    if catalog_sha != evidence["catalog_sha256"]:
        raise ValueError("The frozen V01 catalog changed")
    catalog = Catalog.load(str(catalog_path))
    if args.verify_runtime:
        compare_same_runtime(args, base, rows, cfg, catalog, catalog_sha, evidence, export)
        return
    import onnxruntime

    if onnxruntime.__version__ != "1.32.0":
        raise ValueError("Use ONNX Runtime 1.32.0 to match the checkpoint export")
    args.out.mkdir(parents=True, exist_ok=False)
    results = {}
    for mode in ("offline", "streaming"):
        output = args.out / mode
        official = base / mode / "int8"
        requant = output / "official_fp32_requantized"
        reference = read_jsonl(args.parity / f"{mode}_int8_official.jsonl")
        output.mkdir()
        requant.mkdir()
        for part in (*PARTS, "tokens"):
            filename = f"{part}.onnx" if part != "tokens" else "tokens.txt"
            source = base / mode / "fp32" / filename
            if R.sha256(source) != evidence["official_files"][f"{mode}/fp32/{filename}"]:
                raise ValueError(f"Stale official FP32 artifact: {filename}")
            shutil.copyfile(source, requant / filename)
        with (requant / "quantize.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(quantizer), str(requant)], check=True,
                           stdout=log, stderr=subprocess.STDOUT)
        candidates = {"requantized_official_fp32": requant}
        for part_set, name in ((("encoder",), "reexport_encoder"),
                               (("decoder", "joiner"), "reexport_decoder_joiner")):
            hybrid = output / name
            hybrid.mkdir()
            for part in PARTS:
                filename = f"{part}.int8.onnx"
                folder = args.export / mode if part in part_set else official
                expected = (export["files"][f"{mode}/{filename}"] if part in part_set else
                            evidence["official_files"][f"{mode}/int8/{filename}"])
                if R.sha256(folder / filename) != expected:
                    raise ValueError(f"Stale INT8 artifact: {folder / filename}")
                os.link(folder / filename, hybrid / filename)
            os.link(requant / "tokens.txt", hybrid / "tokens.txt")
            candidates[name] = hybrid
        for name, folder in candidates.items():
            h = decode(mode, "int8", folder, rows, output / f"{name}.jsonl")
            result = compare(reference, h, cfg, "int8", catalog, rows)
            result["components_sha256"] = {part: R.sha256(folder / f"{part}.int8.onnx") for part in PARTS}
            results[f"{mode}/{name}"] = result
            print(f"{mode}/{name}: {result['exact']}/{result['rows']} exact", flush=True)
        report = {"manifest_sha256": manifest_sha256(rows), "onnxruntime_version": onnxruntime.__version__,
                  "parent_parity_sha256": R.sha256(args.parity / "result.json"), "results": results}
        (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
