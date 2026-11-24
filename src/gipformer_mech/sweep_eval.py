"""Shared scoring, mechanism check and selection rule for development decoder sweeps."""

import json
from pathlib import Path

from gipformer_mech.acceptance import paired_cs_test
from gipformer_mech.logit_lens import classify
from gipformer_mech.metrics import score, summarize

CLEAN_DAMAGE_LIMIT = 0.005


def phase2_spans(path: Path, bug_ids: set[str]) -> dict[tuple[str, int], str]:
    """(utt_id, reference word index) -> NEAR_MISS or OTHER, from the stored Phase 2 ranks."""
    report = json.loads(path.read_text(encoding="utf-8"))
    spans = {
        (row["utt_id"], error["word_index"]): "NEAR_MISS" if classify(error["pieces"]) == "NEAR_MISS" else "OTHER"
        for row in report["findings"] for error in row["errors"]
    }
    if {utt for utt, _ in spans} - bug_ids:
        raise ValueError("The Phase 2 report names recordings outside the sealed bug-mining split.")
    return spans


def cs_error_positions(rows: list[dict], scores: dict[str, dict]) -> set[tuple[str, int]]:
    return {(row["utt_id"], error["ref_index"]) for row in rows for error in scores[row["utt_id"]]["cs_word_errors"]}


def mechanism(spans: dict, base: set, candidate: set) -> dict:
    """Fix rate of Phase 2 spans that were wrong at the baseline, split by category."""
    result = {}
    for category in ("NEAR_MISS", "OTHER"):
        wrong = {key for key, value in spans.items() if value == category} & base
        fixed = wrong - candidate
        result[category] = {"wrong_at_zero": len(wrong), "fixed": len(fixed),
                            "fix_rate": len(fixed) / len(wrong) if wrong else None}
    return result


def score_run(sets: dict[str, list[dict]], hyps: dict[int, str], case: str, known: list[str],
              spans: dict, zero: dict | None, seed: int) -> dict:
    """Metrics of one decoder setting; `zero` is the matched baseline run, or None for the baseline itself."""
    rows = [row for members in sets.values() for row in members]
    scores = {row["utt_id"]: score(row["reference"], hyps[row["audio_idx"]], case, known) for row in rows}
    run = {"scores": scores, **{split: summarize(scores[row["utt_id"]] for row in members)
                               for split, members in sets.items()}}
    run["cs_positions"] = cs_error_positions(sets["bug_mining"], scores)
    base = zero or run
    correct = [row["utt_id"] for row in sets["clean_control"]
               if summarize([base["scores"][row["utt_id"]]])["wer_global"] == 0]
    run["clean_damage"] = sum(summarize([scores[utt]])["wer_global"] > 0 for utt in correct) / len(sets["clean_control"])
    run["mechanism"] = mechanism(spans, base["cs_positions"], run["cs_positions"])
    run["random_cs_test"] = paired_cs_test(
        [base["scores"][row["utt_id"]] for row in sets["random_dev"]],
        [scores[row["utt_id"]] for row in sets["random_dev"]], seed,
    ) if zero else None
    return run


def eligible(run: dict, zero: dict) -> bool:
    """Clean damage below 0.5%, random-sample WER no higher than baseline, lower bug-mining CS-WER."""
    return (run["clean_damage"] < CLEAN_DAMAGE_LIMIT
            and run["random_dev"]["wer_global"] <= zero["random_dev"]["wer_global"]
            and run["bug_mining"]["cs_wer"] < zero["bug_mining"]["cs_wer"])


def pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


TABLE_HEADER = [
    "| Setting | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | "
    "Random del | Clean damage | Near-miss fixed | Other fixed |",
    "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]


def table_row(label: str, run: dict) -> str:
    near, other = run["mechanism"]["NEAR_MISS"], run["mechanism"]["OTHER"]
    return (
        f"| {label} | {pct(run['bug_mining']['cs_wer'])} | {pct(run['bug_mining']['wer_global'])} | "
        f"{pct(run['bug_mining']['n_wer'])} | {pct(run['random_dev']['wer_global'])} | "
        f"{pct(run['random_dev']['n_wer'])} | {pct(run['random_dev']['cs_wer'])} | "
        f"{pct(run['random_dev']['ins_rate'])} | {pct(run['random_dev']['del_rate'])} | "
        f"{pct(run['clean_damage'])} | {near['fixed']}/{near['wrong_at_zero']} | {other['fixed']}/{other['wrong_at_zero']} |"
    )


def cs_test_line(label: str, run: dict) -> str:
    test = run["random_cs_test"]
    return (f"- {label}: {test['gain'] * 100:+.2f} pp, p={test['p_value']:.4f}, 95% CI "
            f"[{test['interval'][0] * 100:+.2f}, {test['interval'][1] * 100:+.2f}] pp")
