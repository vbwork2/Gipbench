"""Paired bootstrap and release gates for Gipformer experiments."""

import numpy as np

from gipformer_mech.metrics import summarize


def paired_cs_test(baseline: list[dict], candidate: list[dict], seed: int, rounds: int = 1000) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("Bootstrap requires nonempty, paired utterance scores.")
    denominators = np.array([row["cs_words"] for row in baseline])
    if not denominators.sum() or any(
        first["cs_words"] != second["cs_words"] for first, second in zip(baseline, candidate, strict=True)
    ):
        raise ValueError("Paired evaluations must share the same CS reference counts.")
    gains = np.array([
        first["cs_errors"] - second["cs_errors"] for first, second in zip(baseline, candidate, strict=True)
    ])
    observed = gains.sum() / denominators.sum()
    random = np.random.default_rng(seed)
    samples = random.integers(0, len(gains), size=(rounds, len(gains)))
    n = denominators[samples].sum(axis=1)
    resampled = gains[samples[n > 0]].sum(axis=1) / n[n > 0]
    tail = (resampled - observed >= observed).sum()
    return {
        "gain": float(observed),
        "p_value": float((1 + tail) / (len(resampled) + 1)),
        "interval": [float(value) for value in np.percentile(resampled, [2.5, 97.5])],
    }


def gates(base_held: list[dict], candidate_held: list[dict], base_clean: list[dict],
          candidate_clean: list[dict], evidence: dict, seed: int) -> dict:
    observed = paired_cs_test(base_held, candidate_held, seed)
    if not base_clean or len(base_clean) != len(candidate_clean):
        raise ValueError("Clean safety requires nonempty, paired control scores.")
    broken = sum(
        summarize([first])["wer_global"] == 0 and summarize([second])["wer_global"] > 0
        for first, second in zip(base_clean, candidate_clean, strict=True)
    )
    damage = broken / len(base_clean)
    base, candidate = summarize(base_held), summarize(candidate_held)
    insertion = candidate["ins_rate"] - base["ins_rate"]
    vietnamese = candidate["n_wer"] - base["n_wer"]
    controls = evidence.get("random_fix_rates", [])
    candidate_rate = evidence.get("candidate_fix_rate")
    p95 = float(np.percentile(controls, 95)) if len(controls) >= 64 else None
    return {
        "G1": {"pass": observed["gain"] > 0 and observed["p_value"] < 0.05, **observed},
        "G2": {"pass": damage < 0.005, "damage": damage, "utterances_degraded": broken},
        "G3": {"pass": insertion < 0.003, "insertion_delta": insertion},
        "G4": {"pass": vietnamese <= 0.002, "n_wer_delta": vietnamese},
        "G5": {
            "pass": p95 is not None and candidate_rate is not None and candidate_rate > p95,
            "random_control_count": len(controls),
            "random_p95": p95,
            "candidate_fix_rate": candidate_rate,
        },
    }
