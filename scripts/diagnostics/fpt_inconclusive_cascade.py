"""Test the cascade reading of the INCONCLUSIVE FPT spans from the stored Phase 2 ranks (development only).

Question: `phase2_inconclusive_breakdown.md` found that 72 of 165 INCONCLUSIVE spans have every piece in
the top ten once the predictor sees the reference prefix, and that in nearly all of them the first missing
piece is not the word's first piece. Is that a cascade, meaning an earlier piece of the same word lost its
frame and the emitted prefix then changed the context for later pieces? Or is the missing piece missing
even though the model had the right context?

Rules, fixed before the run:
* population: spans whose current `logit_lens.classify` label is INCONCLUSIVE (greedy prefix, final stack);
* m = index of the first piece whose final-stack `full` best rank is above 10;
* span type, in this order:
  - START: m = 0, the word's first piece is already outside the top ten;
  - LOCAL: m > 0 and every piece before m has best rank 1, so the context was right as far as the ranks show;
  - CHAIN: m > 0 and at least one piece before m has best rank 2 to 10, so an earlier piece lost to a competitor;
* for a CHAIN span the root is the first piece before m with best rank above 1. Its leader is the first
  entry of the stored top ten at that piece's best frame: blank (id 0, `<blk>`) or another piece;
* reachable = every piece is within the top ten under the greedy prefix or under the reference prefix.

Registered prediction: the cascade reading holds if CHAIN is more than half of the reachable spans AND its
share among reachable spans is higher than among the other spans. If CHAIN is half or fewer of the
reachable spans, the reading is not supported and reachability comes from something else.

Limits: a best rank of 1 is the best rank inside the probed window, not proof that the piece was emitted, so
LOCAL can contain spans whose emission was in fact wrong. The reference prefix is an oracle context.
Descriptive and in-sample on development data. Nothing is selected and no repair is tested.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.logit_lens import classify

BLANK = 0


def rank(piece: dict, mode: str = "full") -> int:
    return piece["stacks"][-1][mode]["best_rank"]


def spans(findings: list[dict]) -> list[dict]:
    return [error for row in findings for error in row["errors"] if classify(error["pieces"]) == "INCONCLUSIVE"]


def kind(error: dict) -> tuple[str, str | None]:
    pieces = error["pieces"]
    first_missing = next(i for i, piece in enumerate(pieces) if rank(piece) > 10)
    if first_missing == 0:
        return "START", None
    root = next((piece for piece in pieces[:first_missing] if rank(piece) > 1), None)
    if root is None:
        return "LOCAL", None
    leader = root["stacks"][-1]["full"]["top10"][0]
    if (leader["token_id"] == BLANK) != (leader["piece"] == "<blk>"):
        raise ValueError(f"Blank id {BLANK} does not match the stored piece {leader['piece']!r}.")
    return "CHAIN", "blank" if leader["token_id"] == BLANK else "piece"


def reachable(error: dict) -> bool:
    return all(rank(piece) <= 10 or rank(piece, "reference_full") <= 10 for piece in error["pieces"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_inconclusive_cascade.md")
    args = parser.parse_args()

    groups = {
        "Bug-mining": spans(json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"]),
        "Other development": spans([json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines()]),
    }
    groups["Pooled"] = groups["Bug-mining"] + groups["Other development"]

    def table(errors: list[dict]) -> Counter:
        return Counter((kind(error)[0], reachable(error)) for error in errors)

    counts = {name: table(errors) for name, errors in groups.items()}
    roots = {name: Counter(kind(error)[1] for error in errors if kind(error)[0] == "CHAIN")
             for name, errors in groups.items()}
    pooled = counts["Pooled"]
    reach_total = sum(count for (_, flag), count in pooled.items() if flag)
    other_total = sum(count for (_, flag), count in pooled.items() if not flag)
    chain_reach, chain_other = pooled[("CHAIN", True)], pooled[("CHAIN", False)]
    share_reach, share_other = chain_reach / reach_total, chain_other / other_total
    supported = share_reach > 0.5 and share_reach > share_other

    lines = [
        "# Cascade test for the INCONCLUSIVE FPT spans",
        "",
        "Development only, from stored ranks (no model run). Heldout untouched. The rules and the prediction are in the "
        "docstring of `scripts/diagnostics/fpt_inconclusive_cascade.py` and were fixed before the run.",
        "",
        "Span types. START: the word's first piece is already outside the top ten. LOCAL: the first missing piece "
        "follows pieces that all have rank 1. CHAIN: an earlier piece of the word had rank 2 to 10, so it lost its frame "
        "to a competitor before the first missing piece.",
        "",
        "| Span type | " + " | ".join(f"{name} reachable | {name} not reachable" for name in groups) + " |",
        "|---|" + "---:|" * (2 * len(groups)),
        *(f"| {label} | " + " | ".join(f"{counts[name][(label, True)]} | {counts[name][(label, False)]}"
                                       for name in groups) + " |" for label in ("START", "LOCAL", "CHAIN")),
        "",
        "Reachable means every piece is in the top ten under the greedy prefix or under the reference prefix (oracle context).",
        "",
        f"Pooled: CHAIN is {chain_reach} of {reach_total} reachable spans ({share_reach:.1%}) and {chain_other} of "
        f"{other_total} other spans ({share_other:.1%}).",
        "",
        "## Root of the CHAIN spans (the first earlier piece that lost its frame)",
        "",
        "| Leader at the root piece | " + " | ".join(groups) + " |",
        "|---|" + "---:|" * len(groups),
        "| Blank | " + " | ".join(str(roots[name]["blank"]) for name in groups) + " |",
        "| Another piece | " + " | ".join(str(roots[name]["piece"]) for name in groups) + " |",
        "",
        "## Reading",
        "",
        "Registered prediction (CHAIN above half of the reachable spans and above its share among the others): "
        f"{'supported' if supported else 'not supported'}.",
        "",
        "A best rank of 1 is the best inside the probed window and does not prove the piece was emitted, so LOCAL may "
        "contain spans whose emission was wrong. The reference prefix is an oracle context. This test describes where "
        "the first loss happens. It does not test any repair.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
