"""Measure separately restored FP32 encoder constants on the locked V01 INT8 subset."""

import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path

import onnx
import yaml

from gipbench.bench.evaluate import scorer_code_sha256
from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.textnorm import Catalog
from vietnote_parity import PROTOCOL, ROOT, compare, decode

MODES = ("offline", "streaming")
ENCODER = "encoder.int8.onnx"
DEFAULT_WORK = ROOT / "data/work/vietnote/v01"


def checked(path, digest):
    if R.sha256(path) != digest:
        raise ValueError(f"Artifact digest changed: {path}")
    return path


def tensor_groups(published, reexport, diff):
    def tensors(model):
        initializers = {tensor.name: tensor for tensor in model.graph.initializer}
        constants = {node.name: next(attr.t for attr in node.attribute if attr.name == "value")
                     for node in model.graph.node if node.name in diff["constant_nodes_changed"]}
        return {"downsampling_multiplier": {name: initializers[name] for name in diff["initializer_values_changed"]},
                "positional_encoding": constants}

    original, candidate = tensors(published), tensors(reexport)
    for group, entries in original.items():
        for name, source in entries.items():
            target = candidate[group][name]
            if (source.data_type != onnx.TensorProto.FLOAT or target.data_type != source.data_type
                    or list(source.dims) != list(target.dims) or source.raw_data == target.raw_data):
                raise ValueError(f"Unexpected FP32 tensor difference: {name}")
    return original


def evaluate_mode(mode, rows, spec, catalog, parity, parent, diff, export, isolation, out):
    published_dir = ROOT / "models/vietnote78" / mode / "int8"
    published = {name: checked(published_dir / name, parity["official_files"][f"{mode}/int8/{name}"])
                 for name in (ENCODER, "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")}
    reexported = checked(export / mode / ENCODER,
                         parent["results"][f"{mode}/reexport_encoder"]["artifact_sha256"][ENCODER])
    if (R.sha256(published[ENCODER]) != diff["published_encoder_sha256"]
            or R.sha256(reexported) != diff["reexport_encoder_sha256"]):
        raise ValueError(f"The graph diff belongs to different encoders: {mode}")
    prior = isolation / "same_runtime" / mode
    published_rows, reexport_rows = (
        read_jsonl(checked(prior / f"{case}.jsonl", parent["outputs"][f"{mode}/{case}.jsonl"]))
        for case in ("official", "reexport_encoder")
    )
    known = {R.sha256(published[ENCODER]): (published_rows, "published"),
             R.sha256(reexported): (reexport_rows, "reexport")}
    source_model = onnx.load(published[ENCODER])
    original = tensor_groups(source_model, onnx.load(reexported), diff)
    dest = out / mode
    dest.mkdir()
    results = {}
    for case, selected in (*((key, (key,)) for key in original), ("both", tuple(original))):
        model = onnx.load(reexported)
        for name in original["downsampling_multiplier"] if "downsampling_multiplier" in selected else ():
            target = next(tensor for tensor in model.graph.initializer if tensor.name == name)
            target.raw_data = original["downsampling_multiplier"][name].raw_data
        if "positional_encoding" in selected:
            for node in model.graph.node:
                if node.name in original["positional_encoding"]:
                    target = next(attr.t for attr in node.attribute if attr.name == "value")
                    target.raw_data = original["positional_encoding"][node.name].raw_data
        variant = dest / case
        variant.mkdir()
        encoder = variant / ENCODER
        onnx.save(model, encoder)
        digest = R.sha256(encoder)
        output = dest / f"{case}.jsonl"
        if digest in known:
            hyps, reused = known[digest]
        else:
            for name, path in published.items():
                if name != ENCODER:
                    shutil.copyfile(path, variant / name)
            onnx.checker.check_model(encoder)
            hyps = decode(mode, "int8", variant, rows, output)
            reused = None
        result = {"restored_from_published": {key: list(original[key]) for key in selected},
                  "encoder_sha256": digest, "identical_encoder_control": reused,
                  "vs_published": compare(published_rows, hyps, spec, "int8", catalog, rows),
                  "vs_reexport": compare(reexport_rows, hyps, spec, "int8", catalog, rows)}
        if reused is None:
            result["hypotheses_sha256"] = R.sha256(output)
        results[case] = result
        print(f"{mode}/{case}: {result['vs_published']['exact']}/{len(rows)} vs published, "
              f"{result['vs_reexport']['exact']}/{len(rows)} vs re-export", flush=True)
    return results


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--parity", type=Path, default=DEFAULT_WORK / "parity")
    cli.add_argument("--isolation", type=Path, default=DEFAULT_WORK / "int8_isolation")
    cli.add_argument("--export", type=Path, default=DEFAULT_WORK / "checkpoint_export")
    cli.add_argument("--out", required=True, type=Path)
    args = cli.parse_args()
    parity_path = args.parity / "result.json"
    parent_path = args.isolation / "matched_runtime.json"
    diff_path = args.isolation / "encoder_graph_diff.json"
    parity, parent, diff = (json.loads(path.read_text(encoding="utf-8"))
                            for path in (parity_path, parent_path, diff_path))
    runtime = {"platform": platform.platform(), "python": platform.python_version(),
               "sherpa-onnx": version("sherpa-onnx")}
    if any(parent["runtime"][name] != value for name, value in runtime.items()):
        raise ValueError("Run with the Windows runtime used for the pinned V01 controls")
    if (R.sha256(parity_path) != parent["parent_parity_sha256"]
            or R.sha256(PROTOCOL) != parent["config_sha256"]
            or scorer_code_sha256() != parent["scorer_code_sha256"]):
        raise ValueError("The V01 controls, protocol or scorer changed")
    catalog = Catalog.load(str(checked(R.resolve("catalog/train-v1").catalog, parent["catalog_sha256"])))
    rows = read_jsonl(checked(args.parity / "dev.jsonl", parity["outputs"]["dev.jsonl"]))
    subset = manifest_sha256(rows)
    if len(rows) != 113 or subset != parent["manifest_sha256"]:
        raise ValueError("Expected the 113 locked V01 utterances")
    for row in rows:
        checked(row["_audio_path"], row["audio_sha256"])
    spec = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))["parity"]
    args.out.mkdir(parents=True, exist_ok=False)
    results = {mode: evaluate_mode(mode, rows, spec, catalog, parity, parent, diff[mode],
                                   args.export, args.isolation, args.out)
               for mode in MODES}
    report = {"manifest_sha256": subset, "rows": len(rows), "runtime": runtime,
              "matched_runtime_sha256": R.sha256(parent_path), "graph_diff_sha256": R.sha256(diff_path),
              "results": results}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
