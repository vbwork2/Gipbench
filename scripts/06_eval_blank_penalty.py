"""Sweep the RNN-T blank penalty on development data and check that it acts on near-miss spans.

Selection rule, fixed before the sweep: for each decoding method, a penalty is eligible when it
changes fewer than 0.5% of the clean-control recordings that were correct at penalty 0 and does
not raise WER on the random development sample. Among eligible penalties the largest bug-mining
CS-WER gain wins, with ties going to the smaller penalty. The heldout split is never opened.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.acceptance import paired_cs_test
from gipformer_mech.fpt import decode_rows, recognizer, settings
from gipformer_mech.logit_lens import classify
from gipformer_mech.metrics import score, summarize
from gipformer_mech.splits import development_rows, development_sample

CLEAN_DAMAGE_LIMIT = 0.005
METHODS = {"greedy": "greedy_search", "beam": "modified_beam_search"}


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
    """Fix rate of Phase 2 spans that were wrong at penalty 0, split by category."""
    result = {}
    for category in ("NEAR_MISS", "OTHER"):
        wrong = {key for key, value in spans.items() if value == category} & base
        fixed = wrong - candidate
        result[category] = {"wrong_at_zero": len(wrong), "fixed": len(fixed),
                            "fix_rate": len(fixed) / len(wrong) if wrong else None}
    return result


def decode(args, rows, config, case, method, penalty) -> dict[int, str]:
    cache = args.work_dir / f"{method}_{penalty:g}.json"
    ids = sorted(row["audio_idx"] for row in rows)
    if cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored["audio_idx"] == ids:
            return {int(key): value for key, value in stored["hyps"].items()}
    run = {**config, "decode": {**config["decode"], "blank_penalty": penalty, "threads": args.threads}}
    rec = recognizer(args.model_dir, args.work_dir, run, case, METHODS[method])
    hyps = decode_rows(rows, args.data_dir, rec, config["decode"]["batch_size"])
    cache.write_text(json.dumps({"method": method, "blank_penalty": penalty, "audio_idx": ids,
                                 "hyps": {str(key): hyps[key] for key in ids}}, ensure_ascii=False), encoding="utf-8")
    return hyps


def evaluate(args) -> dict:
    config, case = settings()
    known = config["text"]["cs_terms"]
    sets = {split: development_rows(args.manifests, split) for split in ("bug_mining", "clean_control")}
    sets["random_dev"] = development_sample(args.manifests, args.random_size, config["seed"])
    ids = [row["utt_id"] for rows in sets.values() for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Development sets overlap.")
    spans = phase2_spans(args.phase2, {row["utt_id"] for row in sets["bug_mining"]})
    rows = [row for split in sets.values() for row in split]
    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for method in args.methods:
        runs = []
        zero = None
        for penalty in sorted(set([0.0, *args.penalties])):
            hyps = decode(args, rows, config, case, method, penalty)
            scores = {row["utt_id"]: score(row["reference"], hyps[row["audio_idx"]], case, known) for row in rows}
            run = {"blank_penalty": penalty, "scores": scores}
            for split, members in sets.items():
                run[split] = summarize(scores[row["utt_id"]] for row in members)
            run["cs_positions"] = cs_error_positions(sets["bug_mining"], scores)
            if zero is None:
                zero = run
            correct = [row["utt_id"] for row in sets["clean_control"]
                       if summarize([zero["scores"][row["utt_id"]]])["wer_global"] == 0]
            run["clean_damage"] = sum(
                summarize([scores[utt]])["wer_global"] > 0 for utt in correct
            ) / len(sets["clean_control"])
            run["mechanism"] = mechanism(spans, zero["cs_positions"], run["cs_positions"])
            run["random_cs_test"] = paired_cs_test(
                [zero["scores"][row["utt_id"]] for row in sets["random_dev"]],
                [scores[row["utt_id"]] for row in sets["random_dev"]], config["seed"],
            ) if penalty else None
            runs.append(run)
        eligible = [
            run for run in runs[1:]
            if run["clean_damage"] < CLEAN_DAMAGE_LIMIT
            and run["random_dev"]["wer_global"] <= zero["random_dev"]["wer_global"]
            and run["bug_mining"]["cs_wer"] < zero["bug_mining"]["cs_wer"]
        ]
        chosen = min(eligible, key=lambda run: (run["bug_mining"]["cs_wer"], run["blank_penalty"]), default=None)
        results[method] = {"runs": runs, "chosen": chosen["blank_penalty"] if chosen else None}
    return {"sets": {split: len(members) for split, members in sets.items()},
            "random_cs_utterances": sum(bool(row["cs_tokens"]) for row in sets["random_dev"]),
            "span_categories": dict(Counter(spans.values())), "results": results}


def pct(value) -> str:
    return "n/a" if value is None else f"{value:.2%}"


def write_report(path: Path, outcome: dict) -> None:
    sets = outcome["sets"]
    lines = [
        "# Phase 3b blank-penalty sweep",
        "",
        f"Development only: bug mining {sets['bug_mining']}, clean control {sets['clean_control']} and a seeded random "
        f"sample of {sets['random_dev']} other development recordings ({outcome['random_cs_utterances']} with CS words). "
        "The heldout stayed sealed. INT8 ONNX, same batches for every penalty.",
        "",
        "Selection rule, fixed before the sweep: clean-control damage below 0.5%, random-sample WER no higher than at "
        "penalty 0, then the lowest bug-mining CS-WER, with ties going to the smaller penalty.",
        "",
        f"Phase 2 spans re-classified from stored ranks: {outcome['span_categories']}. The fix rate counts spans that "
        "were wrong at penalty 0 with the same decoder and became correct. If the penalty acts through the "
        "near-miss mechanism, the near-miss fix rate should clearly exceed the rate for other spans.",
    ]
    for method, result in outcome["results"].items():
        lines += [
            "",
            f"## {method}",
            "",
            "| Penalty | Bug CS-WER | Bug WER | Bug N-WER | Random WER | Random N-WER | Random CS-WER | Random ins | "
            "Random del | Clean damage | Near-miss fixed | Other fixed |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for run in result["runs"]:
            near, other = run["mechanism"]["NEAR_MISS"], run["mechanism"]["OTHER"]
            lines.append(
                f"| {run['blank_penalty']:g} | {pct(run['bug_mining']['cs_wer'])} | {pct(run['bug_mining']['wer_global'])} | "
                f"{pct(run['bug_mining']['n_wer'])} | {pct(run['random_dev']['wer_global'])} | "
                f"{pct(run['random_dev']['n_wer'])} | {pct(run['random_dev']['cs_wer'])} | "
                f"{pct(run['random_dev']['ins_rate'])} | {pct(run['random_dev']['del_rate'])} | "
                f"{pct(run['clean_damage'])} | {near['fixed']}/{near['wrong_at_zero']} | "
                f"{other['fixed']}/{other['wrong_at_zero']} |"
            )
        tests = [run for run in result["runs"] if run["random_cs_test"]]
        lines += ["", "Random-sample CS-WER gain against penalty 0 (paired bootstrap, 1,000 rounds):", ""]
        lines += [
            f"- {run['blank_penalty']:g}: {run['random_cs_test']['gain'] * 100:+.2f} pp, "
            f"p={run['random_cs_test']['p_value']:.4f}, 95% CI "
            f"[{run['random_cs_test']['interval'][0] * 100:+.2f}, {run['random_cs_test']['interval'][1] * 100:+.2f}] pp"
            for run in tests
        ]
        chosen = result["chosen"]
        lines += ["", f"Selected penalty: {chosen:g}." if chosen is not None else "No penalty met the selection rule."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--phase2", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "artifacts/blank_penalty")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase3b_blank_penalty.md")
    parser.add_argument("--methods", nargs="+", choices=sorted(METHODS), default=["greedy", "beam"])
    parser.add_argument("--penalties", nargs="+", type=float, default=[0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    parser.add_argument("--random-size", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    outcome = evaluate(args)
    write_report(args.report, outcome)
    summary = {method: {"chosen": result["chosen"], "runs": [
        {key: run[key] for key in ("blank_penalty", "bug_mining", "random_dev", "clean_damage", "mechanism")}
        for run in result["runs"]
    ]} for method, result in outcome["results"].items()}
    (args.work_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
