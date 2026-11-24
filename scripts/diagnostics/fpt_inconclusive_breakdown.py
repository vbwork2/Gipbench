"""Break down the INCONCLUSIVE FPT spans from the stored Phase 2 ranks (development only, no model, no audio).

Two jobs:
* match each current INCONCLUSIVE span to the +/-1 frame review profile in
  `phase2_inconclusive_review.md` (all / some / no pieces in the final-stack top ten), span by span,
  by recomputing `logit_lens.classify` on the stored ranks;
* describe how far the missing pieces are, and whether they are reachable once the predictor
  context is the reference prefix (the oracle context of `reference_full`).

Descriptive only. No threshold is tested here and nothing is selected. "Reachable with the
reference prefix" is an oracle reading: it says the piece is available if the preceding pieces were
right, not that a repair exists.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.logit_lens import classify


def rank(piece: dict, mode: str = "full") -> int:
    return piece["stacks"][-1][mode]["best_rank"]


def profile(error: dict) -> str:
    hits = sum(rank(piece) <= 10 for piece in error["pieces"])
    return "all" if hits == len(error["pieces"]) else "some" if hits else "none"


def spans(findings: list[dict]) -> list[dict]:
    return [error for row in findings for error in row["errors"]]


def describe(inconclusive: list[dict]) -> dict:
    outside = Counter(sum(rank(piece) > 10 for piece in error["pieces"]) for error in inconclusive)
    reachable = [error for error in inconclusive
                 if all(rank(piece) <= 10 or rank(piece, "reference_full") <= 10 for piece in error["pieces"])]
    first_missing = Counter(min(i for i, piece in enumerate(error["pieces"]) if rank(piece) > 10)
                            for error in reachable)
    return {
        "spans": len(inconclusive),
        "profile": Counter(profile(error) for error in inconclusive),
        "one_piece_outside": outside[1],
        "two_pieces_outside": outside[2],
        "three_or_more_outside": sum(count for key, count in outside.items() if key >= 3),
        "within_top20": sum(all(rank(piece) <= 20 for piece in error["pieces"]) for error in inconclusive),
        "within_top50": sum(all(rank(piece) <= 50 for piece in error["pieces"]) for error in inconclusive),
        "farthest_over_100": sum(max(rank(piece) for piece in error["pieces"]) > 100 for error in inconclusive),
        "reachable_with_reference_prefix": len(reachable),
        "reachable_first_missing_at_word_start": first_missing[0],
        "reachable_first_missing_later": sum(count for key, count in first_missing.items() if key > 0),
        "tokenizer_unknown": sum(any(piece["unknown"] for piece in error["pieces"]) for error in inconclusive),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_inconclusive_breakdown.md")
    args = parser.parse_args()

    groups = {
        "Bug-mining": spans(json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"]),
        "Other development": spans([json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines()]),
    }
    groups["Pooled"] = groups["Bug-mining"] + groups["Other development"]
    categories = {name: Counter(classify(error["pieces"]) for error in errors) for name, errors in groups.items()}
    stale = {name: sum(error["category"] != classify(error["pieces"]) for error in errors)
             for name, errors in groups.items()}
    # Review population: the stored bug-mining category is INCONCLUSIVE for what is now NEAR_MISS or INCONCLUSIVE.
    old = [error for error in groups["Bug-mining"] if error["category"] == "INCONCLUSIVE"]
    review = Counter(profile(error) for error in old)
    matched = Counter((profile(error), classify(error["pieces"])) for error in old)
    details = {name: describe([error for error in errors if classify(error["pieces"]) == "INCONCLUSIVE"])
               for name, errors in groups.items()}

    total = len(groups["Pooled"])
    near, override = categories["Pooled"]["NEAR_MISS"], categories["Pooled"]["PREDICTOR_OVERRIDE_BUG"]
    reach = details["Pooled"]["reachable_with_reference_prefix"]
    top20 = details["Pooled"]["within_top20"]

    def row(label: str, key: str) -> str:
        return f"| {label} | " + " | ".join(str(details[name][key]) for name in groups) + " |"

    lines = [
        "# Breakdown of the INCONCLUSIVE FPT spans",
        "",
        "Development only, from stored ranks (no model run). Heldout untouched. Rules are in the docstring of "
        "`scripts/diagnostics/fpt_inconclusive_breakdown.py`. Descriptive: no threshold is tested here.",
        "",
        "## Match with the review profiles (bug-mining, span by span)",
        "",
        f"`phase2_inconclusive_review.md` sampled from {sum(review.values())} spans that were INCONCLUSIVE before the "
        f"NEAR_MISS category existed ({review['all']} all pieces, {review['some']} some, {review['none']} none in the "
        "final-stack top ten at +/-1 frame). Recomputing `classify` on the stored ranks gives:",
        "",
        "| Review profile | Now NEAR_MISS | Now INCONCLUSIVE |",
        "|---|---:|---:|",
        *(f"| {name} | {matched[(key, 'NEAR_MISS')]} | {matched[(key, 'INCONCLUSIVE')]} |"
          for name, key in (("All pieces", "all"), ("Some pieces", "some"), ("No pieces", "none"))),
        "",
        f"The profile is the same quantity as the NEAR_MISS test, so the {matched[('all', 'NEAR_MISS')]} \"all\" spans are "
        f"the near misses and the {matched[('some', 'INCONCLUSIVE')] + matched[('none', 'INCONCLUSIVE')]} remaining "
        "INCONCLUSIVE spans are exactly the \"some\" and \"none\" ones. Stored categories that differ from the recomputed "
        f"ones: bug-mining {stale['Bug-mining']}, other development {stale['Other development']} (the old INCONCLUSIVE "
        "spans that are now NEAR_MISS).",
        "",
        "## What the INCONCLUSIVE spans look like",
        "",
        "| | " + " | ".join(groups) + " |",
        "|---|" + "---:|" * len(groups),
        row("INCONCLUSIVE spans", "spans"),
        *(f"| Final-stack profile \"{key}\" | " + " | ".join(str(details[name]["profile"][key]) for name in groups) + " |"
          for key in ("some", "none")),
        row("Exactly one piece outside the top ten", "one_piece_outside"),
        row("Two pieces outside", "two_pieces_outside"),
        row("Three or more outside", "three_or_more_outside"),
        row("Every piece within the top 20", "within_top20"),
        row("Every piece within the top 50", "within_top50"),
        row("Worst missing piece below rank 100", "farthest_over_100"),
        row("Span contains a tokenizer unknown", "tokenizer_unknown"),
        row("All pieces in the top ten if the predictor had the reference prefix", "reachable_with_reference_prefix"),
        row("... and the first missing piece is the word's first piece", "reachable_first_missing_at_word_start"),
        row("... and the first missing piece comes later in the word", "reachable_first_missing_later"),
        "",
        "## Where this leaves the post-encoder share (pooled, " + str(total) + " spans)",
        "",
        "Alternative readings of the same table, not a decision. They differ by what is allowed to count as "
        "\"after the encoder\". Each of the last two rows adds a different set of INCONCLUSIVE spans to the first "
        "row; the two sets overlap, so the rows are not cumulative.",
        "",
        "| Reading | Spans | Share |",
        "|---|---:|---:|",
        f"| NEAR_MISS + PREDICTOR_OVERRIDE_BUG (the `classify` labels) | {near + override} | {(near + override) / total:.1%} |",
        f"| First row plus INCONCLUSIVE spans with every piece within the top 20 | {near + override + top20} | "
        f"{(near + override + top20) / total:.1%} |",
        f"| First row plus INCONCLUSIVE spans reachable with the reference prefix (oracle context) | "
        f"{near + override + reach} | {(near + override + reach) / total:.1%} |",
        "",
        "The last reading uses an oracle context, so it measures availability and not repairability. The top-20 reading "
        "changes the 10 in the NEAR_MISS rule, which was fixed before the first run. Neither is a preregistered test.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
