"""Report the Phase 2 diagnosis of the extra pool's diagnostic part next to the sealed development pool.

`TODO_rea.md` item 1.3. Reads stored ranks only, no model or audio:
* extra: `artifacts/extra_near_miss/findings.jsonl` from `fpt_pool_near_miss.py --extra`, which reads
  `extra_diagnostic.jsonl` only. The new heldout is never opened;
* development: `reports/phase2_bug_vs_gap.json` (bug mining) and `artifacts/pool_near_miss/findings.jsonl`
  (other development recordings).

Counting rules are the ones already fixed: category = `logit_lens.classify` with the greedy prefix,
blank-led near miss = `logit_lens.blank_led_near_miss`, a blank-led piece = a near-miss piece below rank 1
whose final-stack leader is blank (the Phase 4 event unit), cascade types from
`fpt_inconclusive_cascade.py`. The cascade prediction registered there is re-checked on the extra pool
as a replicate, with its rule unchanged. The unbiased development comparison is "other development",
because bug mining was selected for repeated confusions.

The development counts must reproduce the published ones (`phase2_pool_near_miss.md`); the script stops
if they do not.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpt_inconclusive_cascade import BLANK, kind, reachable
from gipbench.core.fileio import read_jsonl_rows as read_rows
from gipformer_mech.logit_lens import blank_led_near_miss, classify

CATEGORIES = [("NEAR_MISS", "Near miss"), ("ENCODER_BUG", "Encoder bug"),
              ("PREDICTOR_OVERRIDE_BUG", "Predictor override"), ("TRUE_GAP", "True gap"),
              ("INCONCLUSIVE", "Inconclusive")]
PUBLISHED = {"spans": 365, "near_miss": 183, "blank_led_spans": 117, "blank_led_pieces": 204}


def relabel(findings: list[dict]) -> list[dict]:
    return [{**row, "errors": [{**error, "category": classify(error["pieces"])} for error in row["errors"]]}
            for row in findings]


def blank_led_pieces(error: dict) -> int:
    if error["category"] != "NEAR_MISS":
        return 0
    return sum(piece["stacks"][-1]["full"]["best_rank"] > 1
               and piece["stacks"][-1]["full"]["top10"][0]["token_id"] == BLANK for piece in error["pieces"])


def profile(findings: list[dict]) -> dict:
    errors = [error for row in findings for error in row["errors"]]
    inconclusive = [error for error in errors if error["category"] == "INCONCLUSIVE"]
    return {
        "recordings": len(findings),
        "spans": len(errors),
        "categories": Counter(error["category"] for error in errors),
        "blank_led_spans": sum(blank_led_near_miss(error, BLANK) for error in errors),
        "blank_led_pieces": sum(map(blank_led_pieces, errors)),
        "blank_led_recordings": sum(any(blank_led_pieces(error) for error in row["errors"]) for row in findings),
        "near_miss_words": len({error["reference_word"] for error in errors if error["category"] == "NEAR_MISS"}),
        "cascade": Counter((kind(error)[0], reachable(error)) for error in inconclusive),
        "roots": Counter(kind(error)[1] for error in inconclusive if kind(error)[0] == "CHAIN"),
    }


def share(count: int, total: int) -> str:
    return f"{count} ({count / total:.1%})" if total else "0"


def cascade_test(cascade: Counter) -> tuple[bool, str]:
    reach = sum(count for (_, flag), count in cascade.items() if flag)
    other = sum(count for (_, flag), count in cascade.items() if not flag)
    chain_reach, chain_other = cascade[("CHAIN", True)], cascade[("CHAIN", False)]
    share_reach = chain_reach / reach if reach else 0.0
    share_other = chain_other / other if other else 0.0
    held = share_reach > 0.5 and share_reach > share_other
    return held, (f"CHAIN is {chain_reach} of {reach} reachable spans ({share_reach:.1%}) and {chain_other} of "
                  f"{other} other spans ({share_other:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extra", type=Path, default=ROOT / "artifacts/extra_near_miss/findings.jsonl")
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_extra_near_miss.md")
    parser.add_argument("--check-only", action="store_true", help="Check the development counts and stop.")
    args = parser.parse_args()

    bug = relabel(json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"])
    other = relabel(read_rows(args.pool))
    pooled = profile(bug + other)
    got = {"spans": pooled["spans"], "near_miss": pooled["categories"]["NEAR_MISS"],
           "blank_led_spans": pooled["blank_led_spans"], "blank_led_pieces": pooled["blank_led_pieces"]}
    if got != PUBLISHED:
        raise ValueError(f"Development counts {got} do not reproduce the published {PUBLISHED}.")
    if args.check_only:
        print(f"Development counts reproduce the published ones: {got}")
        return

    extra = relabel(read_rows(args.extra))
    lock = json.loads(args.split_lock.read_text(encoding="utf-8"))["diagnostic"]
    groups = {"Other development": profile(other), "Development pooled": pooled, "Extra diagnostic": profile(extra)}
    ex = groups["Extra diagnostic"]
    if (ex["recordings"], ex["spans"]) != (lock["rows_with_cs_errors"], lock["cs_error_spans"]):
        raise ValueError(f"Extra findings ({ex['recordings']} rec., {ex['spans']} spans) do not cover the split lock.")

    names = list(groups)
    row = lambda label, cells: f"| {label} | " + " | ".join(cells) + " |"
    lines = [
        "# Phase 2 diagnosis on the extra FPT pool (diagnostic part)",
        "",
        "Diagnostic part of the extra pool only (`extra_diagnostic.jsonl`, sha256 checked against "
        "`phase1_extra_split_lock.json`). The new heldout was not opened. Same logit lens, emitted-greedy-prefix "
        "context and categories as Phase 2; rules in the docstrings of `fpt_pool_near_miss.py` and "
        "`fpt_extra_report.py`. The development columns reproduce `phase2_pool_near_miss.md`.",
        "",
        "| | " + " | ".join(f"{name} ({groups[name]['recordings']} rec.)" for name in names) + " |",
        "|---|" + "---:|" * len(names),
        row("Spans", [str(groups[name]["spans"]) for name in names]),
        *(row(label, [share(groups[name]["categories"][key], groups[name]["spans"]) for name in names])
          for key, label in CATEGORIES),
        row("Near-miss spans where blank leads a non-top piece", [str(groups[name]["blank_led_spans"]) for name in names]),
        row("Blank-led near-miss pieces (Phase 4 event unit)", [str(groups[name]["blank_led_pieces"]) for name in names]),
        row("Recordings with such a piece", [str(groups[name]["blank_led_recordings"]) for name in names]),
        row("Distinct reference words among near-miss spans", [str(groups[name]["near_miss_words"]) for name in names]),
        "",
        "## Cascade types of the INCONCLUSIVE spans (replicate of `phase2_inconclusive_cascade.md`)",
        "",
        "| Span type | " + " | ".join(f"{name} reachable | {name} not reachable" for name in names) + " |",
        "|---|" + "---:|" * (2 * len(names)),
        *(row(label, [f"{groups[name]['cascade'][(label, True)]} | {groups[name]['cascade'][(label, False)]}"
                      for name in names]) for label in ("START", "LOCAL", "CHAIN")),
        "",
        "| CHAIN root leader | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
        row("Blank", [str(groups[name]["roots"]["blank"]) for name in names]),
        row("Another piece", [str(groups[name]["roots"]["piece"]) for name in names]),
        "",
    ]
    for name in ("Development pooled", "Extra diagnostic"):
        held, text = cascade_test(groups[name]["cascade"])
        lines.append(f"- {name}: {text}. Registered cascade prediction: {'holds' if held else 'does not hold'}.")
    gap_like = ex["categories"]["TRUE_GAP"] + sum(
        count for (label, _), count in ex["cascade"].items() if label in ("START", "LOCAL"))
    chain = sum(count for (label, _), count in ex["cascade"].items() if label == "CHAIN")
    lines += [
        "",
        "## Bug or gap on the extra pool (same reading as `TODO_rea.md`)",
        "",
        f"- Post-encoder, near miss: {share(ex['categories']['NEAR_MISS'], ex['spans'])}.",
        f"- Post-encoder, chain: {share(chain, ex['spans'])}.",
        "- Encoder bug and predictor override: "
        f"{share(ex['categories']['ENCODER_BUG'] + ex['categories']['PREDICTOR_OVERRIDE_BUG'], ex['spans'])}.",
        f"- Gap-like (true gap, START, LOCAL): {share(gap_like, ex['spans'])}.",
        "",
        "Descriptive. The extra pool includes 24 recordings (26 spans) whose text also occurs in the development pool; "
        "they were forced to the diagnostic side by the split. Blank-led pieces are not yet replayed through the "
        "Joiner (item 1.4), so they are candidate events, not replicated ones.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
