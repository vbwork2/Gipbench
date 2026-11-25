"""Re-export the pinned VietNote checkpoint with the upstream Icefall exporters."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import platform
from importlib.metadata import version
from pathlib import Path

import onnx

from gipbench.core.download import verify_file
from gipbench.core.release import sha256
from gipbench.model.train.vietnote_icefall import icefall_source, model_args
from gipbench.model.vietnote import LOCK, load_checkpoint


def average_checkpoints(manifest, card, output):
    import torch

    plan = json.loads(manifest.read_text(encoding="utf-8"))
    base = next(item["sha256"] for item in card["model_files"] if item["path"] == "pytorch/pretrained.pt")
    members = plan["checkpoints"]
    if (plan.get("model_revision") != card["revision"] or plan.get("base_checkpoint_sha256") != base
            or len(members) < 2 or len({item["sha256"] for item in members}) != len(members)):
        raise ValueError("Checkpoint averaging requires unique derivatives of the pinned VietNote checkpoint")
    state, dtypes = {}, {}
    with torch.no_grad():
        for index, item in enumerate(members):
            checkpoint = manifest.parent / item["path"]
            if sha256(checkpoint) != item["sha256"]:
                raise ValueError(f"Checkpoint digest mismatch: {checkpoint}")
            model = load_checkpoint(checkpoint)["model"]
            if not model or index and model.keys() != state.keys():
                raise ValueError("Checkpoint model keys differ")
            for key, tensor in model.items():
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"Checkpoint state entry is not a tensor: {key}")
                if index and (tensor.shape != state[key].shape or tensor.dtype != dtypes[key]):
                    raise ValueError(f"Incompatible tensor shape or dtype: {key}")
                if tensor.is_floating_point():
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"Non-finite checkpoint tensor: {key}")
                    if index:
                        state[key].add_(tensor)
                    else:
                        state[key] = tensor.to(dtype=torch.float64, copy=True)
                elif index:
                    if not torch.equal(tensor, state[key]):
                        raise ValueError(f"Non-floating checkpoint tensor differs: {key}")
                else:
                    state[key] = tensor.clone()
                dtypes[key] = tensor.dtype
        for key, tensor in state.items():
            if tensor.is_floating_point():
                state[key] = tensor.div_(len(members)).to(dtypes[key])
    pending = output.with_suffix(".tmp")
    torch.save({"model": state}, pending)
    pending.replace(output)
    return {"manifest_sha256": sha256(manifest), "checkpoints": [item["sha256"] for item in members],
            "rule": "uniform_model_state_mean"}


def set_downsample_weights(path, mode, raw):
    graph = onnx.load(path)
    consumer = "/encoder/3/downsample/Mul_2" if mode == "offline" else "/downsample_2/Mul"
    node, = (n for n in graph.graph.node if n.name == consumer and n.op_type == "Mul")
    tensor, = (t for t in graph.graph.initializer if t.name in node.input)
    if tensor.data_type != onnx.TensorProto.FLOAT or list(tensor.dims) != [8, 1, 1]:
        raise ValueError(f"Unexpected downsampling weights in {path}")
    tensor.raw_data = raw
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    return tensor.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--softmax-python", default=sys.executable,
                        help="Python with the validated CPU FP32 softmax kernel for the locked checkpoint")
    parser.add_argument("--model", type=Path, default=Path("models/vietnote78"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", type=Path, help="VietNote checkpoint containing a model state dictionary")
    source.add_argument("--average-manifest", type=Path, help="JSON manifest of compatible VietNote derivatives")
    args = parser.parse_args()
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    base = args.model.resolve()
    icefall = icefall_source(card, args.icefall.resolve())
    recipe = icefall / "egs/librispeech/ASR/zipformer"
    out = args.out.resolve()
    specs = {item["path"]: item for item in card["model_files"]}
    averaging = None
    out.mkdir(parents=True, exist_ok=False)
    if args.average_manifest:
        checkpoint = out / "averaged.pt"
        averaging = average_checkpoints(args.average_manifest.resolve(), card, checkpoint)
    else:
        checkpoint = args.checkpoint.resolve() if args.checkpoint else base / "pytorch/pretrained.pt"
    digest = sha256(checkpoint)
    pretrained = digest == specs["pytorch/pretrained.pt"]["sha256"]
    tokens = base / "tokens.txt"
    quantizer = base / "upstream/scripts/quantize_int8.py"
    if args.checkpoint is None and averaging is None:
        verify_file(checkpoint, specs["pytorch/pretrained.pt"])
    verify_file(tokens, specs["tokens.txt"])
    verify_file(quantizer, next(item for item in card["recipe_files"] if item["path"] == "scripts/quantize_int8.py"))
    helper = Path(__file__).with_name("vietnote_checkpoint_softmax.py")
    export_weights = checkpoint if pretrained else out / "model.pt"
    # Relative paths also work when WSL invokes a Windows Python interpreter.
    softmax = json.loads(subprocess.check_output([args.softmax_python, os.path.relpath(helper),
                                                  os.path.relpath(checkpoint),
                                                  *([] if pretrained else [os.path.relpath(export_weights)])], text=True))
    raw = bytes.fromhex(softmax["fp32_le_hex"])
    if pretrained and hashlib.sha256(raw).hexdigest() != "b5508121a041cd7b60e1e8997df5c5df3dbde5bd58257302e939edfeb5241ff2":
        raise ValueError("Checkpoint softmax rounding differs. Use --softmax-python with the validated Windows CPU runtime.")
    softmax["initializers"] = {}
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", PYTHONPATH=str(icefall))
    # The pinned exporters use tracing and predate the change to PyTorch's default exporter.
    launcher = (
        "import functools, runpy, sys, torch; "
        "torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False); "
        "sys.path.insert(0, '.'); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    commands = {}
    for mode in ("offline", "streaming"):
        dest = out / mode
        dest.mkdir()
        (dest / "epoch-1.pt").symlink_to(export_weights)
        script = "export-onnx-streaming.py" if mode == "streaming" else "export-onnx.py"
        flags = model_args(card, export=True)
        if mode == "offline":
            flags += ["--chunk-size=-1", "--left-context-frames=-1"]
        command = [sys.executable, "-c", launcher, script, "--exp-dir", str(dest),
                   "--tokens", str(tokens), "--epoch", "1", "--avg", "1",
                   "--use-averaged-model", "0", "--enable-int8-quantization", "0",
                   "--context-size", str(card["context_size"]), *flags]
        commands[mode] = command
        with (dest / "export.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=recipe, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        (dest / "epoch-1.pt").unlink()
        suffix = "epoch-1-avg-1"
        if mode == "streaming":
            arch = card["architecture"]
            suffix += f"-chunk-{arch['streaming_export_chunk_size']}-left-{arch['streaming_export_left_context_frames']}"
        for part in ("encoder", "decoder", "joiner"):
            (dest / f"{part}-{suffix}.onnx").rename(dest / f"{part}.onnx")
        shutil.copyfile(tokens, dest / "tokens.txt")
        softmax["initializers"][mode] = set_downsample_weights(dest / "encoder.onnx", mode, raw)
        with (dest / "quantize.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(quantizer), str(dest)], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    files = {file.relative_to(out).as_posix(): sha256(file)
             for mode in commands for file in sorted((out / mode).iterdir())
             if file.suffix in {".onnx", ".txt"}}
    if sha256(checkpoint) != digest:
        raise ValueError("Checkpoint changed during export")
    evidence = {"model_revision": card["revision"], "checkpoint_sha256": digest,
                "architecture": card["architecture"], "averaging": averaging,
                "exporter_sha256": sha256(__file__), "softmax_helper_sha256": sha256(helper),
                "model_state_file_sha256": sha256(export_weights),
                "runtime": {"python": sys.version, "platform": platform.platform(),
                            **{name: version(name) for name in ("torch", "onnx", "onnxruntime")}},
                "quantization": {"method": "dynamic", "per_channel": True, "ops": ["MatMul"], "dtype": "QInt8"},
                "icefall_commit": card["icefall_commit"], "quantizer_sha256": sha256(quantizer),
                "commands": commands, "files": files, "checkpoint_softmax": softmax}
    (out / "export.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
