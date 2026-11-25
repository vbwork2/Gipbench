"""Check checkpoint-to-ONNX tensors, streaming caches, and INT8 weight layout."""

import argparse
import json
import logging
import runpy
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.download import verify_file
from gipbench.core.release import sha256
from gipbench.infer.adapters.base import load_audio
from gipbench.model.train.vietnote_icefall import model_args
from gipbench.model.vietnote import LOCK
from gipformer_mech.fbank import fbank

ATOL = 0.002
RTOL = 0.001


def quantization(path):
    graph = onnx.load(path).graph
    tensors = {tensor.name: tensor for tensor in graph.initializer}
    ops = [node for node in graph.node if node.op_type == "MatMulInteger"]
    per_channel = 0
    for node in ops:
        weight = tensors[node.input[1]]
        scale = tensors.get(weight.name.removesuffix("_quantized") + "_scale")
        per_channel += (weight.data_type == onnx.TensorProto.INT8 and scale is not None
                        and list(scale.dims) == [weight.dims[-1]])
    return {"quantized_matmuls": len(ops), "per_channel_qint8_matmuls": per_channel,
            "passed": bool(ops) and per_channel == len(ops), "sha256": sha256(path)}


def ort_sessions(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    return [ort.InferenceSession(str(model / f"{part}.onnx"), sess_options=options,
                                 providers=["CPUExecutionProvider"]) for part in ("encoder", "decoder", "joiner")]


def infer(session, inputs):
    return session.run(None, {item.name: tensor.detach().numpy().astype(
        np.int32 if item.type == "tensor(int32)" else tensor.detach().numpy().dtype, copy=False)
        for item, tensor in zip(session.get_inputs(), inputs, strict=True)})


def compare_mode(base, icefall, mode, rows, card, export, checkpoint=None):
    streaming = mode == "streaming"
    script = "export-onnx-streaming.py" if streaming else "export-onnx.py"
    wrappers = runpy.run_path(str(icefall / "egs/librispeech/ASR/zipformer" / script))
    torch.set_num_threads(4)
    flags = model_args(card, export=True)
    if not streaming:
        flags += ["--chunk-size=-1", "--left-context-frames=-1"]
    flags += ["--tokens", str(base / "tokens.txt"), "--context-size", str(card["context_size"])]
    params = wrappers["get_params"]()
    params.update(vars(wrappers["get_parser"]().parse_args(flags)))
    params.update(vocab_size=card["vocab_size"], blank_id=0)
    model = wrappers["get_model"](params)
    if checkpoint is None:
        checkpoint = base / "pytorch/pretrained.pt"
        verify_file(checkpoint, next(item for item in card["model_files"] if item["path"] == "pytorch/pretrained.pt"))
    model.load_state_dict(torch.load(checkpoint, weights_only=True, mmap=True, map_location="cpu")["model"], strict=True)
    model.eval()
    convert_options = {} if streaming else {"is_onnx": True}
    wrappers["convert_scaled_to_non_scaled"](model, inplace=True, **convert_options)
    encoder = wrappers["OnnxEncoder"](model.encoder, model.encoder_embed, model.joiner.encoder_proj)
    decoder = wrappers["OnnxDecoder"](model.decoder, model.joiner.decoder_proj)
    joiner = wrappers["OnnxJoiner"](model.joiner.output_linear)
    sessions = ort_sessions(export / mode)
    sizes = {key: {"comparisons": 0, "max_abs": 0.0, "failures": 0}
             for key in ("encoder", "cache_or_lengths", "decoder", "joiner")}

    def compare(key, expected, actual):
        tensor = expected.detach().numpy()
        stat = sizes[key]
        stat["comparisons"] += 1
        stat["max_abs"] = max(stat["max_abs"], float(np.max(np.abs(tensor - actual))))
        stat["failures"] += not np.allclose(tensor, actual, atol=ATOL, rtol=RTOL)

    context = card["context_size"]
    blank = params.blank_id
    with torch.inference_mode():
        for row in rows:
            features = fbank(load_audio(row["_audio_path"], target_sr=16000)[0])
            if streaming:
                arch = card["architecture"]
                window, stride = arch["streaming_onnx_feature_window"], arch["streaming_onnx_feature_shift"]
                padded = np.pad(features, ((0, 66), (0, 0)), constant_values=np.log(np.finfo(np.float32).eps))
                chunks = (padded[i:i + window] for i in range(0, len(padded) - window + 1, stride))
                state = encoder.get_init_states()
                ort_state = state
            else:
                chunks = [features]
            history = [blank] * context
            for chunk in chunks:
                x = torch.from_numpy(chunk).unsqueeze(0)
                if streaming:
                    expected, state = encoder(x, state)
                    observed, *cache = infer(sessions[0], [x, *ort_state])
                    ort_state = [torch.from_numpy(value) for value in cache]
                    for t, a in zip(state, cache, strict=True):
                        compare("cache_or_lengths", t, a)
                else:
                    lengths = torch.tensor([len(chunk)])
                    expected, expected_lengths = encoder(x, lengths)
                    observed, actual_lengths = infer(sessions[0], [x, lengths])
                    compare("cache_or_lengths", expected_lengths, actual_lengths)
                compare("encoder", expected, observed)
                labels = torch.tensor([history[-context:]], dtype=torch.int64)
                predicted = decoder(labels)
                (actual_predicted,) = infer(sessions[1], [labels])
                compare("decoder", predicted, actual_predicted)
                x_join = expected[:, -1]
                logits = joiner(x_join, predicted)
                (actual_logits,) = infer(sessions[2], [torch.from_numpy(observed[:, -1]), torch.from_numpy(actual_predicted)])
                compare("joiner", logits, actual_logits)
                history.append(int(logits.argmax()))
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--icefall", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-model", type=Path, help="Model-only file from a selected pilot checkpoint export")
    parser.add_argument("--pairs", type=int, choices=(1, 12), default=1, help="Pairs for a selected fine-tuned model")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    logging.disable(logging.INFO)
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    base = Path("models/vietnote78").resolve()
    rows = read_jsonl(args.manifest)
    long = sorted([row for row in rows if float(row["duration"]) >= 20], key=lambda row: row["id"])
    short = sorted([row for row in rows if float(row["duration"]) < 20], key=lambda row: row["id"])
    count = args.pairs if args.pilot_model else 12
    selected = [*long[:count], *short[:count]]
    if len(selected) != 2 * count:
        raise ValueError(f"The tensor check requires {count} short and {count} long utterances")
    for row in selected:
        if sha256(row["_audio_path"]) != row["audio_sha256"]:
            raise ValueError(f"Audio digest mismatch: {row['id']}")
    measurements = {mode: compare_mode(base, args.icefall, mode, selected, card, args.export, args.pilot_model)
                    for mode in ("offline", "streaming")}
    quantized = {f"{kind}/{mode}/{part}": quantization(directory / f"{part}.int8.onnx")
                 for mode in measurements
                 for kind, directory in (("official", base / mode / "int8"), ("reexport", args.export / mode))
                 for part in ("encoder", "decoder", "joiner")}
    passed = all(stat["comparisons"] > 0 and not stat["failures"] for stats in measurements.values() for stat in stats.values())
    evidence = {"passed": passed and all(x["passed"] for x in quantized.values()), "measurements": measurements,
                "quantization": quantized, "atol": ATOL, "rtol": RTOL, "rows": len(selected), "long_rows": count,
                "manifest_sha256": manifest_sha256(rows), "selected_ids": [row["id"] for row in selected],
                "model_revision": card["revision"], "checkpoint_input_sha256": sha256(args.pilot_model or base / "pytorch/pretrained.pt"),
                "export_metadata_sha256": sha256(args.export / "export.json"),
                "onnxruntime_version": ort.__version__, "torch_version": torch.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: evidence[key] for key in ("passed", "measurements")}), flush=True)


if __name__ == "__main__":
    main()
