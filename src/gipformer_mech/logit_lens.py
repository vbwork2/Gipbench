"""Read out intermediate Zipformer activations on the final RNN-T frame grid."""

from bisect import bisect_left
from collections import defaultdict

import numpy as np
import torch
from torch.nn.functional import pad

from gipformer_mech.rnnt_loader import encode


def align_reference(model, output, decoder_projected, ids: list[int], blank: int) -> list[int]:
    """Find a monotone blank/token path using the final Joiner and reference prefixes."""
    encoder_projected = model.joiner.encoder_proj(output[0])
    frame_count = len(encoder_projected)
    token_count = len(ids)
    blank_scores = np.empty((frame_count, token_count + 1), dtype=np.float32)
    label_scores = np.empty((frame_count, token_count), dtype=np.float32)
    for step in range(0, token_count + 1, 8):
        decoder = decoder_projected[step:step + 8]
        scores = model.joiner.output_linear(torch.tanh(encoder_projected[:, None] + decoder[None])).log_softmax(-1)
        blank_scores[:, step:step + len(decoder)] = scores[..., blank].cpu().numpy()
        count = min(len(decoder), token_count - step)
        if count:
            positions = torch.arange(count, device=output.device)
            target_ids = torch.as_tensor(ids[step:step + count], device=output.device)
            label_scores[:, step:step + count] = scores[:, positions, target_ids].cpu().numpy()

    best = np.full((frame_count + 1, token_count + 1), -np.inf)
    emitted = np.zeros(best.shape, dtype=bool)
    best[0, 0] = 0
    for frame in range(frame_count):
        for index in range(token_count):
            candidate = best[frame, index] + label_scores[frame, index]
            if candidate > best[frame, index + 1]:
                best[frame, index + 1] = candidate
                emitted[frame, index + 1] = True
        best[frame + 1] = best[frame] + blank_scores[frame]
    if not np.isfinite(best[-1, -1]):
        raise ValueError("No finite RNN-T alignment path for the reference.")
    frames = [0] * token_count
    frame, index = frame_count, token_count
    while index:
        if emitted[frame, index]:
            index -= 1
            frames[index] = frame
        else:
            frame -= 1
    return frames


def cumulative_outputs(model, outputs: list[torch.Tensor], final: torch.Tensor) -> list[dict]:
    """Use Zipformer's latest available channel at each depth and its actual output pooling."""
    dimension = final.shape[-1]
    result = []
    for index, current in enumerate(outputs):
        used = current.shape[-1]
        pieces = [current]
        for earlier in reversed(outputs[:index]):
            if earlier.shape[-1] > used:
                pieces.append(earlier[..., used:])
                used = earlier.shape[-1]
        combined = pad(torch.cat(pieces, dim=-1), (0, dimension - used))
        output = model.encoder.downsample_output(combined).permute(1, 0, 2)
        result.append({"output": output, "available_channels": used})
    if not torch.allclose(result[-1]["output"], final, rtol=1e-5, atol=1e-6):
        raise ValueError("Reconstructed final probe differs from the actual encoder output.")
    return result


def top_at_frames(logits, target: int, frames: list[int], sp) -> dict:
    logprobs = logits.log_softmax(-1)
    ranks = (logprobs > logprobs[:, target:target + 1]).sum(-1) + 1
    best = min(range(len(frames)), key=lambda i: (int(ranks[i]), -float(logprobs[i, target])))
    values, ids = logprobs[best].topk(10)
    return {
        "best_rank": int(ranks[best]),
        "frame": frames[best],
        "target_logprob": float(logprobs[best, target]),
        "top10": [
            {"token_id": int(i), "piece": sp.id_to_piece(int(i)), "logprob": float(p)}
            for i, p in zip(ids, values, strict=True)
        ],
    }


def classify(pieces: list[dict], context: str = "full") -> str:
    if any(item["unknown"] for item in pieces):
        return "TRUE_GAP"
    if all(item["stacks"][-1]["acoustic"]["best_rank"] <= 10 for item in pieces) and any(
        item["stacks"][-1][context]["best_rank"] > 10 for item in pieces
    ):
        return "PREDICTOR_OVERRIDE_BUG"
    if all(any(stack[context]["best_rank"] <= 10 for stack in item["stacks"][1:4]) for item in pieces) and any(
        item["stacks"][-1][context]["best_rank"] > 10 for item in pieces
    ):
        return "ENCODER_BUG"
    if all(
        stack[mode]["best_rank"] > 10
        for item in pieces for stack in item["stacks"] for mode in (context, "acoustic")
    ):
        return "TRUE_GAP"
    return "INCONCLUSIVE"


def greedy_contexts(model, output, lengths, initial: list[int]) -> torch.Tensor:
    """Replay the emitted prefix available before each greedy RNN-T frame."""
    from beam_search import greedy_search_batch

    trace = greedy_search_batch(model, output, lengths, return_timestamps=True)
    history = initial + trace.hyps[0]
    frames = [bisect_left(trace.timestamps[0], frame) for frame in range(output.shape[1])]
    prefixes = torch.tensor(
        [history[count:count + len(initial)] for count in frames], device=output.device
    )
    return model.joiner.decoder_proj(model.decoder(prefixes, need_pad=False)[:, 0])


@torch.inference_mode()
def investigate(model, sp, model_map: dict, waveform: np.ndarray, words: list[str],
                error_positions: list[int], device: str) -> list[dict]:
    captured = {}
    handles = []
    for stack in model_map["encoder_stacks"]:
        index = stack["index"]

        def store(_module, _args, output, index=index):
            captured[index] = output

        handles.append(model.get_submodule(stack["module"]).register_forward_hook(store))
    try:
        final, lengths = encode(model, waveform, model_map, device)
    finally:
        for handle in handles:
            handle.remove()
    probes = cumulative_outputs(model, [captured[index] for index in sorted(captured)], final)
    ids = []
    positions = defaultdict(list)
    for index, word in enumerate(words):
        pieces = sp.encode(word, out_type=int)
        positions[index] = list(range(len(ids), len(ids) + len(pieces)))
        ids.extend(pieces)
    if ids != sp.encode(" ".join(words), out_type=int):
        raise ValueError("Word-by-word BPE does not match utterance tokenization.")
    context = model_map["predictor"]["context_size"]
    blank = model_map["blank_id"]
    initial = [-1] * (context - 1) + [blank]
    prompt = initial + ids
    contexts = torch.tensor([prompt[i:i + context] for i in range(len(ids) + 1)], device=device)
    reference_predictor = model.joiner.decoder_proj(model.decoder(contexts, need_pad=False)[:, 0])
    anchors = align_reference(model, final, reference_predictor, ids, blank)
    predictor = greedy_contexts(model, final, lengths, initial)
    findings = []
    for index in error_positions:
        pieces = []
        for position in positions[index]:
            target = ids[position]
            center = anchors[position]
            frames = list(range(max(0, center - 1), min(final.shape[1], center + 2)))
            stacks = []
            for stack_index, probe in enumerate(probes):
                acoustic = model.joiner.encoder_proj(probe["output"][0, frames])
                full = model.joiner.output_linear(torch.tanh(acoustic + predictor[frames]))
                disabled = model.joiner.output_linear(torch.tanh(acoustic))
                teacher_forced = model.joiner.output_linear(torch.tanh(acoustic + reference_predictor[position]))
                stacks.append({
                    "stack": stack_index,
                    "available_channels": probe["available_channels"],
                    "full": top_at_frames(full, target, frames, sp),
                    "acoustic": top_at_frames(disabled, target, frames, sp),
                    "reference_full": {
                        "best_rank": int(((teacher_forced > teacher_forced[:, target:target + 1]).sum(-1) + 1).min())
                    },
                })
            pieces.append({
                "target_id": target,
                "target_piece": sp.id_to_piece(target),
                "unknown": target == sp.unk_id(),
                "frame": center,
                "stacks": stacks,
            })
        findings.append({
            "word_index": index,
            "reference_word": words[index],
            "frame_span": [min(piece["frame"] for piece in pieces), max(piece["frame"] for piece in pieces) + 1],
            "category": classify(pieces),
            "reference_prefix_category": classify(pieces, "reference_full"),
            "tokenizer_unknown": any(piece["unknown"] for piece in pieces),
            "pieces": pieces,
        })
    return findings
