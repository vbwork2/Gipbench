"""Re-export the pinned VietNote checkpoint with the upstream Icefall exporters."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import onnx

from gipbench.core.download import verify_file
from gipbench.core.release import sha256
from gipbench.model.train.vietnote_icefall import icefall_source, model_args
from gipbench.model.vietnote import LOCK


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
    args = parser.parse_args()
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    base = args.model.resolve()
    icefall = icefall_source(card, args.icefall.resolve())
    recipe = icefall / "egs/librispeech/ASR/zipformer"
    out = args.out.resolve()
    specs = {item["path"]: item for item in card["model_files"]}
    checkpoint = base / "pytorch/pretrained.pt"
    tokens = base / "tokens.txt"
    quantizer = base / "upstream/scripts/quantize_int8.py"
    verify_file(checkpoint, specs["pytorch/pretrained.pt"])
    verify_file(tokens, specs["tokens.txt"])
    verify_file(quantizer, next(item for item in card["recipe_files"] if item["path"] == "scripts/quantize_int8.py"))
    helper = Path(__file__).with_name("vietnote_checkpoint_softmax.py")
    # Relative paths also work when WSL invokes a Windows Python interpreter.
    softmax = json.loads(subprocess.check_output([args.softmax_python, os.path.relpath(helper),
                                                  os.path.relpath(checkpoint)], text=True))
    raw = bytes.fromhex(softmax["fp32_le_hex"])
    if hashlib.sha256(raw).hexdigest() != "b5508121a041cd7b60e1e8997df5c5df3dbde5bd58257302e939edfeb5241ff2":
        raise ValueError("Checkpoint softmax rounding differs. Use --softmax-python with the validated Windows CPU runtime.")
    softmax["initializers"] = {}
    out.mkdir(parents=True, exist_ok=False)
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
        (dest / "epoch-1.pt").symlink_to(checkpoint)
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
    evidence = {"model_revision": card["revision"], "checkpoint_sha256": sha256(checkpoint),
                "icefall_commit": card["icefall_commit"], "quantizer_sha256": sha256(quantizer),
                "commands": commands, "files": files, "checkpoint_softmax": softmax}
    (out / "export.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
