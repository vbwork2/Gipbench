"""Compute corpus-level ASR errors and attribute word errors to VI or CS."""

from collections import Counter
from collections.abc import Iterable

import jiwer

from gipbench.core.align import attribute
from gipformer_mech.text_norm import cs_positions, tokens


def score(reference: str, hypothesis: str, case: str, cs_terms: list[str]) -> dict:
    ref, hyp = tokens(reference, case), tokens(hypothesis, case)
    positions = cs_positions(ref, cs_terms)
    spans = [-1] * len(ref)
    for position in positions:
        spans[position] = position
    result = jiwer.process_words(" ".join(ref), " ".join(hyp))
    ops = []
    for chunk in result.alignments[0]:
        ref_indices = range(chunk.ref_start_idx, chunk.ref_end_idx)
        hyp_indices = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
        if chunk.type in ("equal", "substitute"):
            kind = "C" if chunk.type == "equal" else "S"
            ops.extend((kind, ri, hi) for ri, hi in zip(ref_indices, hyp_indices, strict=True))
        elif chunk.type == "delete":
            ops.extend(("D", ri, None) for ri in ref_indices)
        else:
            ops.extend(("I", None, hi) for hi in hyp_indices)
    attributed, sub, delete, insert = attribute(ops, spans)
    ref_chars, hyp_chars = "".join(ref), "".join(hyp)
    chars = jiwer.process_characters(ref_chars, hyp_chars)
    return {
        "reference_words": len(ref),
        "reference_chars": len(ref_chars),
        "vi_words": len(ref) - len(positions),
        "cs_words": len(positions),
        "substitutions": sub,
        "deletions": delete,
        "insertions": insert,
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
        "vi_errors": attributed.get(-1, 0),
        "cs_errors": sum(attributed.get(position, 0) for position in positions),
        "cs_tokens": [ref[position] for position in positions],
        "cs_word_errors": [
            {"ref_index": ri, "ref": ref[ri], "hyp": hyp[hi] if hi is not None else "", "op": kind}
            for kind, ri, hi in ops if kind in ("S", "D") and spans[ri] >= 0
        ],
    }


def summarize(rows: Iterable[dict]) -> dict:
    counts = Counter()
    for row in rows:
        counts.update({key: value for key, value in row.items() if isinstance(value, int)})

    def ratio(value: int, key: str) -> float | None:
        return value / counts[key] if counts[key] else None

    return {
        "wer_global": ratio(counts["substitutions"] + counts["deletions"] + counts["insertions"], "reference_words"),
        "cer_global": ratio(counts["char_errors"], "reference_chars"),
        "n_wer": ratio(counts["vi_errors"], "vi_words"),
        "cs_wer": ratio(counts["cs_errors"], "cs_words"),
        "ins_rate": ratio(counts["insertions"], "reference_words"),
        "del_rate": ratio(counts["deletions"], "reference_words"),
        "sub_rate": ratio(counts["substitutions"], "reference_words"),
    }
