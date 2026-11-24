"""Exact per-unit decomposition of the Gipformer Joiner, and the statistics the unit-level analysis needs.

The deployed Joiner is logit = W @ tanh(enc + dec) + b with enc and dec already projected to 512 values
(the ONNX encoder and decoder contain the projections). For a frame with hidden vector h, the log-odds
margin of blank over a piece t splits exactly over the 512 hidden units:

    margin = sum_j (W[blank, j] - W[t, j]) * h[j] + (b[blank] - b[t])

so each unit has a signed contribution, and zeroing h[j] removes exactly that unit's term from every logit.
"""

from hashlib import sha256
from pathlib import Path

import numpy as np

from gipformer_mech.model_audit import checksum


def load_output_layer(model_dir: Path, model_map: dict) -> tuple[np.ndarray, np.ndarray]:
    """fp32 joiner.output_linear (weight [vocab, 512], bias [vocab]) from the audited checkpoint."""
    import torch

    path = model_dir / "model.pt"
    if checksum(path) != model_map["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the Phase 0 audit.")
    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("model", state)
    return (state["joiner.output_linear.weight"].float().numpy().copy(),
            state["joiner.output_linear.bias"].float().numpy().copy())


def fold_of(utt_id: str, folds: int = 2) -> int:
    """Deterministic fold by recording, so selection and evaluation never share a recording."""
    return int(sha256(utt_id.encode("utf-8")).hexdigest()[:8], 16) % folds


def hidden(enc: np.ndarray, dec: np.ndarray) -> np.ndarray:
    return np.tanh(enc + dec)


def contributions(h: np.ndarray, weight: np.ndarray, blank: int, targets: np.ndarray) -> np.ndarray:
    """c[n, j] = (W[blank, j] - W[targets[n], j]) * h[n, j]."""
    return (weight[blank][None, :] - weight[targets]) * h


def margins(c: np.ndarray, bias: np.ndarray, blank: int, targets: np.ndarray) -> np.ndarray:
    return c.sum(axis=1) + bias[blank] - bias[targets]


def best_nonblank(logits: np.ndarray, blank: int) -> np.ndarray:
    masked = logits.copy()
    masked[:, blank] = -np.inf
    return masked.argmax(axis=1)


def mean_contribution(h: np.ndarray, weight: np.ndarray, bias: np.ndarray, blank: int, chunk: int = 4096) -> np.ndarray:
    """Mean over frames of the blank-vs-best-competitor contribution of every unit."""
    total = np.zeros(h.shape[1], dtype=np.float64)
    for start in range(0, len(h), chunk):
        part = h[start:start + chunk]
        competitor = best_nonblank(part @ weight.T + bias, blank)
        total += contributions(part, weight, blank, competitor).sum(axis=0)
    return total / len(h)


def select_units(event_mean: np.ndarray, ordinary_mean: np.ndarray, k: int) -> np.ndarray:
    """Units whose blank-over-piece contribution is largest at events in excess of ordinary frames."""
    if not 0 < k <= len(event_mean) or event_mean.shape != ordinary_mean.shape:
        raise ValueError("Invalid unit count or mismatched contribution vectors.")
    return np.argsort(-(event_mean - ordinary_mean), kind="stable")[:k]


def share_explained(c: np.ndarray, margin: np.ndarray, units) -> np.ndarray:
    """Fraction of each event's margin that the given units contribute."""
    return c[:, units].sum(axis=1) / margin


def random_unit_shares(c: np.ndarray, margin: np.ndarray, k: int, draws: int, seed: int) -> np.ndarray:
    """Median event share for `draws` random unit sets of size k."""
    rng = np.random.default_rng(seed)
    picks = np.zeros((c.shape[1], draws), dtype=c.dtype)
    for draw in range(draws):
        picks[rng.choice(c.shape[1], size=k, replace=False), draw] = 1
    return np.median((c @ picks) / margin[:, None], axis=0)


def ablated_logits(h: np.ndarray, weight: np.ndarray, bias: np.ndarray, units) -> np.ndarray:
    """Logits with the given hidden units zeroed."""
    keep = np.ones(h.shape[1], dtype=h.dtype)
    keep[units] = 0
    return (h * keep) @ weight.T + bias


def argmax_change(h: np.ndarray, weight: np.ndarray, bias: np.ndarray, units, blank: int, chunk: int = 4096) -> dict:
    """How often zeroing the units changes the per-frame argmax, split by baseline blank / piece frames."""
    changed = np.zeros(2, dtype=np.int64)
    counts = np.zeros(2, dtype=np.int64)
    for start in range(0, len(h), chunk):
        part = h[start:start + chunk]
        base = (part @ weight.T + bias).argmax(axis=1)
        after = ablated_logits(part, weight, bias, units).argmax(axis=1)
        for index, mask in enumerate((base == blank, base != blank)):
            counts[index] += mask.sum()
            changed[index] += (after[mask] != base[mask]).sum()
    return {
        "blank_frames": int(counts[0]), "blank_changed": int(changed[0]),
        "piece_frames": int(counts[1]), "piece_changed": int(changed[1]),
        "change_rate": float(changed.sum() / max(1, counts.sum())),
    }


def concentration_curve(mean_excess: np.ndarray, sizes: tuple[int, ...]) -> dict[int, float]:
    """Share of the summed positive excess contribution carried by the top-k units."""
    positive = np.sort(np.clip(mean_excess, 0, None))[::-1]
    total = positive.sum()
    return {k: float(positive[:k].sum() / total) if total else 0.0 for k in sizes}


def participation_ratio(values: np.ndarray) -> float:
    """Effective number of units: (sum |v|)^2 / sum v^2. 1 means one unit, len(values) means uniform."""
    squared = float((values ** 2).sum())
    return float(np.abs(values).sum() ** 2 / squared) if squared else 0.0


def replay(model, enc: np.ndarray, weight: np.ndarray, bias: np.ndarray, keep_decoder: bool = False) -> dict:
    """Greedy decode with the fp32 output layer, one symbol per frame, recording hidden vectors and tokens."""
    blank = model.blank
    context = (-1,) * (model.context_size - 1) + (blank,)
    dec = model.decoder_out(context)[0]
    states = np.empty((len(enc), enc.shape[1]), dtype=np.float32)
    tokens = np.empty(len(enc), dtype=np.int64)
    decoders = np.empty_like(states) if keep_decoder else None
    for frame in range(len(enc)):
        h = hidden(enc[frame], dec)
        states[frame] = h
        if keep_decoder:
            decoders[frame] = dec
        token = int((weight @ h + bias).argmax())
        tokens[frame] = token
        if token != blank and token != model.unk:
            context = context[1:] + (token,)
            dec = model.decoder_out(context)[0]
    result = {"hidden": states, "tokens": tokens}
    if keep_decoder:
        result["decoder"] = decoders
    return result
