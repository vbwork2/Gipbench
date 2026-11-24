"""Profile the gap-like FPT error spans against the near-miss spans (development only, stored ranks, no model).

`TODO_rea.md` item 3.3. The locked span list (`fpt_span_lock.py`) has 63 gap_like spans (START, LOCAL and
TRUE_GAP) next to 293 mechanism spans. This script asks what the gap-like spans have in common, using only
the stored ranks and the development reference texts.

EXPLORATORY, not a test. The features below were chosen after reading 14 gap-like spans (rare English words
spelled out in many one- or two-letter pieces, blank leading at the missing piece). Nothing here is
registered, no threshold is applied, and the comparison cannot confirm a hypothesis it was built from. Its
use is to decide what a later, registered check (for example a tokenizer check) should look at.

Groups: near_miss; chain_reachable and chain_unreachable (CHAIN spans split by whether every piece is in the
top ten under the reference prefix); start_local (START and LOCAL); true_gap. The weakest piece of a span
is its piece with the largest final-stack `full` best rank. "Occurrences" counts the development pool rows
whose reference contains the word (the pool is the only text available here, so this is a rarity proxy within
the development pool, not within the model's training data).
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.text_norm import tokens

ORDER = ("near_miss", "chain_reachable", "chain_unreachable", "start_local", "true_gap")


def rank(piece: dict, mode: str = "full") -> int:
    return piece["stacks"][-1][mode]["best_rank"]


def group_name(row: dict) -> str | None:
    if row["group"] == "near_miss":
        return "near_miss"
    if row["group"] == "chain":
        return "chain_reachable" if row["reachable_with_reference_prefix"] else "chain_unreachable"
    if row["category"] == "TRUE_GAP":
        return "true_gap"
    return "start_local" if row["group"] == "gap_like" else None


def letters(piece: str) -> int:
    return len(piece.replace("▁", ""))


def share(count: int, total: int) -> str:
    return f"{count} ({count / total:.0%})" if total else "0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=ROOT / "reports/phase2_span_lock.jsonl")
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--case", default="upper")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_gap_profile.md")
    args = parser.parse_args()

    lock = [json.loads(line) for line in args.lock.read_text(encoding="utf-8").splitlines()]
    findings = json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"]
    findings += [json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines()]
    errors = {(item["utt_id"], error["word_index"]): error for item in findings for error in item["errors"]}
    pool_rows = [json.loads(line) for line in (args.manifests / "full_scored.jsonl").read_text(encoding="utf-8").splitlines()]
    occurrences = Counter(word for row in pool_rows for word in set(tokens(row["reference"], args.case)))

    groups = {name: [] for name in ORDER}
    for row in lock:
        name = group_name(row)
        if name:
            groups[name].append((row, errors[(row["utt_id"], row["word_index"])]))

    def stats(items: list[tuple[dict, dict]]) -> dict:
        weakest = [max(error["pieces"], key=rank) for _, error in items]
        sizes = [len(error["pieces"]) for _, error in items]
        words = [row["reference_word"] for row, _ in items]
        return {
            "spans": len(items),
            "words": len(set(words)),
            "median_pieces": median(sizes),
            "five_or_more_pieces": sum(size >= 5 for size in sizes),
            "median_weakest_rank": median(rank(piece) for piece in weakest),
            "weakest_over_100": sum(rank(piece) > 100 for piece in weakest),
            "weakest_acoustic_over_100": sum(rank(piece, "acoustic") > 100 for piece in weakest),
            "weakest_short": sum(letters(piece["target_piece"]) <= 2 for piece in weakest),
            "blank_leads_weakest": sum(piece["stacks"][-1]["full"]["top10"][0]["piece"] == "<blk>" for piece in weakest),
            "word_in_one_row": sum(occurrences[word] == 1 for word in words),
            "word_in_five_or_more_rows": sum(occurrences[word] >= 5 for word in words),
            "weakest_pieces": Counter(piece["target_piece"] for piece in weakest),
        }

    table = {name: stats(items) for name, items in groups.items() if items}
    names = [name for name in ORDER if name in table]

    def row(label: str, key: str, fraction: bool = True) -> str:
        cells = [share(table[name][key], table[name]["spans"]) if fraction else str(table[name][key]) for name in names]
        return f"| {label} | " + " | ".join(cells) + " |"

    gap_like = Counter()
    for name in ("start_local", "true_gap", "chain_unreachable"):
        if name in table:
            gap_like.update(table[name]["weakest_pieces"])

    lines = [
        "# Profile of the gap-like FPT error spans",
        "",
        "Development only, from stored ranks and development reference texts. Heldout untouched. **Exploratory**: the "
        "features were chosen after reading 14 gap-like spans, nothing is registered and no threshold is applied. "
        "See the docstring of `scripts/diagnostics/fpt_gap_profile.py`.",
        "",
        "| | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
        row("Spans", "spans", False),
        row("Distinct reference words", "words", False),
        row("Median pieces per word", "median_pieces", False),
        row("Words of five or more pieces", "five_or_more_pieces"),
        row("Median rank of the weakest piece", "median_weakest_rank", False),
        row("Weakest piece below rank 100", "weakest_over_100"),
        row("Weakest piece below rank 100 acoustically (no predictor context)", "weakest_acoustic_over_100"),
        row("Weakest piece has at most two letters", "weakest_short"),
        row("Blank leads at the weakest piece", "blank_leads_weakest"),
        row("Word occurs in one development row only", "word_in_one_row"),
        row("Word occurs in five or more development rows", "word_in_five_or_more_rows"),
        "",
        "Weakest pieces most often seen in the spans that stay missing even with the reference prefix (start_local, "
        "true_gap, chain_unreachable): " + ", ".join(f"`{piece}` ({count})" for piece, count in gap_like.most_common(10)) + ".",
        "",
        "Reading limits: the weakest piece of a near miss is still in the top ten, so its rank row compares across "
        "groups but not as a pass mark. The occurrence count is a rarity proxy inside the development pool. The "
        "training frequency of the pieces and words is not available here, so whether these pieces are rare in the "
        "model's training data is not answered. A tokenizer check on the training text is the next step if the "
        "profile points that way.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
