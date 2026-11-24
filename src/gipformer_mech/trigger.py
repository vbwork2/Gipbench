"""Gated blank penalty on the Joiner hidden state: a learned trigger and its matched-rate controls.

A gate scores each frame and fires when the score is above a threshold tau. It only acts on frames where
blank is the argmax, because a blank penalty changes nothing elsewhere. When it fires, the blank logit is
lowered by `penalty` before the argmax. Kinds:
  probe   s = w . (h - mean) / std + c, a logistic probe trained on event frames against ordinary ones
  random  s = w . (h - mean) / std with a random Gaussian w (random-direction control)
  margin  s = best non-blank logit - blank logit (the margin control; by TODO_rea 2.1 it equals a
          uniform penalty, so a probe that does not beat it has only learned "the margin is small")
tau is set from the negatives' scores so that every kind fires at the same rate there.
"""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from gipformer_mech.joiner_probe import hidden


@dataclass(frozen=True)
class Gate:
    kind: str
    tau: float
    w: np.ndarray | None = None
    c: float = 0.0
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def scores(self, h: np.ndarray, logits: np.ndarray, blank: int) -> np.ndarray:
        """Scores for frames h [n, 512] with logits [n, vocab]."""
        if self.kind == "margin":
            others = logits.copy()
            others[:, blank] = -np.inf
            return others.max(axis=1) - logits[:, blank]
        if self.kind in ("probe", "random"):
            return ((h - self.mean) / self.std) @ self.w + self.c
        raise ValueError(f"Unknown gate kind {self.kind}.")


def threshold_for_rate(scores: np.ndarray, rate: float) -> float:
    """tau such that k = max(1, round(rate * n)) of the n scores are strictly above it (fewer only on ties)."""
    if not 0 < rate < 1 or len(scores) < 2:
        raise ValueError("Need a rate in (0, 1) and at least two scores.")
    k = min(len(scores) - 1, max(1, round(rate * len(scores))))
    return float(np.sort(np.asarray(scores))[::-1][k])


def fit_probe(positives: np.ndarray, negatives: np.ndarray, l2: float = 1.0) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Class-balanced L2 logistic regression on standardized features. Returns (w, c, mean, std)."""
    from scipy.optimize import minimize

    x = np.concatenate([positives, negatives]).astype(np.float64)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    mean, std = x.mean(axis=0), x.std(axis=0) + 1e-6
    z = (x - mean) / std
    weight = np.where(y == 1, 0.5 / len(positives), 0.5 / len(negatives))

    def loss(theta):
        w, c = theta[:-1], theta[-1]
        logit = z @ w + c
        p = 1.0 / (1.0 + np.exp(-logit))
        value = np.sum(weight * (np.logaddexp(0, logit) - y * logit)) + 0.5 * l2 * (w @ w) / len(y)
        grad = weight * (p - y)
        return value, np.concatenate([z.T @ grad + l2 * w / len(y), [grad.sum()]])

    result = minimize(loss, np.zeros(z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return result.x[:-1], float(result.x[-1]), mean, std


def random_direction(dim: int, seed: int) -> np.ndarray:
    w = np.random.default_rng(seed).standard_normal(dim)
    return w / np.linalg.norm(w)


def gated_replay(model, enc: np.ndarray, weight: np.ndarray, bias: np.ndarray, gate: Gate | None,
                 penalty: float) -> dict:
    """`joiner_probe.replay` with the gate applied on blank-argmax frames. gate None or penalty 0 = baseline."""
    blank = model.blank
    context = (-1,) * (model.context_size - 1) + (blank,)
    dec = model.decoder_out(context)[0]
    ids, fired = [], 0
    for frame in range(len(enc)):
        h = hidden(enc[frame], dec)
        logits = weight @ h + bias
        token = int(logits.argmax())
        if gate is not None and penalty and token == blank:
            if gate.scores(h[None], logits[None], blank)[0] > gate.tau:
                fired += 1
                logits = logits.copy()
                logits[blank] -= penalty
                token = int(logits.argmax())
        if token != blank and token != model.unk:
            ids.append(token)
            context = context[1:] + (token,)
            dec = model.decoder_out(context)[0]
    return {"ids": ids, "fired": fired, "frames": len(enc)}


def text_fold(text_key: str, folds: int = 2, salt: str = "trigger-fold-v1") -> int:
    return int(sha256(f"{salt}\n{text_key}".encode("utf-8")).hexdigest(), 16) % folds
