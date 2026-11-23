"""Audit the released Gipformer checkpoint and its paired decoding artifacts."""

from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path
import json
import re


STACK = re.compile(r"encoder\.encoders\.(\d+)\.")
FFN = re.compile(
    r"(encoder\.encoders\.\d+\.(?:encoder\.)?layers\.\d+\.feed_forward\d+)"
    r"\.(in_proj|out_proj)\.(weight|bias)"
)
PROBES = (
    "xin", "chào", "một", "Tanaka", "Tokyo", "Sheraton", "Alfredo",
    "email", "TV", "CD", "K+", "cent", "benchmark", "streaming",
)


def checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect(model_dir: Path) -> tuple[dict, dict]:
    import onnx
    from onnx import numpy_helper
    import sentencepiece
    import torch

    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    checkpoint = model_dir / config["pytorch"]["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    tensors = {name: list(tensor.shape) for name, tensor in state.items()}

    token_path = model_dir / config["tokens"]
    bpe_path = model_dir / config["bpe_model"]
    token_rows = [line.rsplit(maxsplit=1) for line in token_path.read_text(encoding="utf-8").splitlines() if line]
    tokens = {int(index): token for token, index in token_rows}
    sp = sentencepiece.SentencePieceProcessor(model_file=str(bpe_path))
    if len(tokens) != len(token_rows) or len(tokens) != config["vocab_size"]:
        raise ValueError("Token table size or ID uniqueness disagrees with config.json.")
    if sp.vocab_size() != len(tokens) or any(tokens.get(i) != sp.id_to_piece(i) for i in range(len(tokens))):
        raise ValueError("SentencePiece IDs do not match tokens.txt.")
    blank_id, = [index for index, token in tokens.items() if token == "<blk>"]
    case_counts = Counter(
        "upper" if char.isupper() else "lower"
        for token in tokens.values() if not token.startswith(("<", "#"))
        for char in token if char.isupper() or char.islower()
    )
    if not case_counts:
        raise ValueError("No cased characters found in the vocabulary.")
    case = case_counts.most_common(1)[0][0]

    onnx_files = {}
    graphs = {}
    for part in ("encoder", "decoder", "joiner"):
        for label in (part, f"{part}_int8"):
            if label in config["onnx"] and (model_dir / config["onnx"][label]).is_file():
                onnx_files[label] = model_dir / config["onnx"][label]
        preferred = f"{part}_int8" if f"{part}_int8" in onnx_files else part
        graphs[part] = onnx.load(onnx_files[preferred], load_external_data=False)
    constants = {
        node.output[0]: numpy_helper.to_array(node.attribute[0].t)
        for node in graphs["encoder"].graph.node
        if node.op_type == "Constant" and node.attribute[0].type == onnx.AttributeProto.TENSOR
    }
    divisors = {
        node.name.removesuffix("/Div"): int(constants[node.input[1]].item())
        for node in graphs["encoder"].graph.node
        if node.op_type == "Div"
        and re.fullmatch(r"/encoder/(?:\d+/downsample|downsample_output)/Div", node.name)
    }
    if sum(node.op_type == "Tanh" for node in graphs["joiner"].graph.node) != 1:
        raise ValueError("Expected one Tanh activation in the released Joiner graph.")

    ffn_by_stack = defaultdict(lambda: defaultdict(dict))
    for name, shape in tensors.items():
        match = FFN.fullmatch(name)
        if match:
            module, projection, kind = match.groups()
            index = int(STACK.match(module)[1])
            ffn_by_stack[index][module][f"{projection}.{kind}"] = {"name": name, "shape": shape}
    stack_indices = sorted({int(match[1]) for name in tensors if (match := STACK.match(name))})
    stacks = []
    for index in stack_indices:
        module = f"encoder.encoders.{index}"
        bias = tensors.get(f"{module}.downsample.bias")
        factor = bias[0] if bias else 1
        if factor != divisors.get(f"/encoder/{index}/downsample", 1):
            raise ValueError(f"Checkpoint and ONNX downsampling disagree for stack {index}.")
        ffn = dict(ffn_by_stack[index])
        input_dimensions = {entries["in_proj.weight"]["shape"][1] for entries in ffn.values()}
        if len(input_dimensions) != 1 or any(
            set(entries) != {"in_proj.weight", "in_proj.bias", "out_proj.weight", "out_proj.bias"}
            for entries in ffn.values()
        ):
            raise ValueError(f"Incomplete or inconsistent FFN inventory for {module}.")
        stacks.append({
            "index": index,
            "module": module,
            "downsampling_factor": factor,
            "dimension": input_dimensions.pop(),
            "block_count": len({name.rsplit(".", 1)[0] for name in ffn}),
            "feed_forward": ffn,
        })
    output_factor = tensors["encoder.downsample_output.bias"][0]
    if not stacks or output_factor != divisors["/encoder/downsample_output"]:
        raise ValueError("Checkpoint and ONNX output downsampling disagree.")

    joiner = {
        name: tensors[name] for name in (
            "joiner.encoder_proj.weight", "joiner.encoder_proj.bias",
            "joiner.decoder_proj.weight", "joiner.decoder_proj.bias",
            "joiner.output_linear.weight", "joiner.output_linear.bias",
        )
    }
    predictor = {name: shape for name, shape in tensors.items() if name.startswith("decoder.")}
    context_size = predictor["decoder.conv.weight"][-1]
    decoder_meta = {item.key: item.value for item in graphs["decoder"].metadata_props}
    if int(decoder_meta["context_size"]) != context_size or int(decoder_meta["vocab_size"]) != len(tokens):
        raise ValueError("Checkpoint and ONNX Predictor context or vocabulary disagree.")

    hashes = {path.name: checksum(path) for path in [
        checkpoint, token_path, bpe_path, model_dir / "config.json", *onnx_files.values(),
    ]}
    model_map = {
        "schema_version": 1,
        "model_type": config["model_type"],
        "checkpoint_sha256": hashes[checkpoint.name],
        "sample_rate": config["sample_rate"],
        "feature_dim": graphs["encoder"].graph.input[0].type.tensor_type.shape.dim[2].dim_value,
        "blank_id": blank_id,
        "vocabulary_size": len(tokens),
        "vocabulary_case": case,
        "encoder_stacks": stacks,
        "encoder_output_downsampling_factor": output_factor,
        "joiner": {"activation": "tanh", "tensors": joiner},
        "predictor": {"context_size": context_size, "tensors": predictor},
        "has_ctc_head": any(name.startswith("ctc_output.") for name in tensors),
    }
    probes = {}
    for word in PROBES:
        probes[word] = {}
        for form, text in (("lower", word.lower()), ("upper", word.upper())):
            ids = sp.encode(text, out_type=int)
            probes[word][form] = {
                "ids": ids,
                "pieces": [tokens[index] for index in ids],
                "piece_count": len(ids),
                "unknown_count": ids.count(sp.unk_id()),
            }

    state_values = sum(tensor.numel() for tensor in state.values())
    ctc_values = sum(tensor.numel() for name, tensor in state.items() if name.startswith("ctc_output."))
    report = {
        "status": "passed",
        "sha256": hashes,
        "state_dict_entries": len(tensors),
        "state_dict_values": state_values,
        "ctc_head_values": ctc_values,
        "rnnt_state_dict_values": state_values - ctc_values,
        "advertised_parameters": config["num_parameters"],
        "vocabulary_case_counts": dict(case_counts),
        "tokenizer_probes": probes,
        "onnx_metadata": {
            part: {item.key: item.value for item in graph.metadata_props} for part, graph in graphs.items()
        },
        "encoder_joiner_input_dim": joiner["joiner.encoder_proj.weight"][1],
        "stacks_incompatible_with_direct_joiner": [
            stack["index"] for stack in stacks if stack["dimension"] != joiner["joiner.encoder_proj.weight"][1]
        ],
    }
    return model_map, report
