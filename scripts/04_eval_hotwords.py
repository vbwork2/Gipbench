"""Sweep hotword scores on bug-mining and clean controls using the same beam decoder."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import sentencepiece

from gipformer_mech.fpt import decode_rows, recognizer, settings
from gipformer_mech.metrics import score, summarize
from gipformer_mech.splits import development_rows
from gipformer_mech.text_norm import tokens


def run(args: argparse.Namespace) -> None:
    config, case = settings()
    manifests = args.manifests
    if not (manifests / "heldout_lock.json").is_file():
        raise ValueError("Seal the development and heldout splits before tuning hotwords.")
    bug_rows = development_rows(manifests, "bug_mining")
    clean_rows = development_rows(manifests, "clean_control")
    if {row["utt_id"] for row in bug_rows} & {row["utt_id"] for row in clean_rows}:
        raise ValueError("Bug-mining and clean-control utterances overlap.")
    candidates = {token for row in bug_rows for token in row["cs_tokens"]}
    known = config["text"]["cs_terms"]
    sp = sentencepiece.SentencePieceProcessor(model_file=str(args.model_dir / "bpe.model"))
    hotwords = {
        word for word in candidates
        if (ids := sp.encode(word, out_type=int)) and sp.unk_id() not in ids
    }
    if not hotwords or not clean_rows:
        raise ValueError("Hotword tuning requires representable CS terms and a nonempty clean control.")
    args.hotwords.parent.mkdir(parents=True, exist_ok=True)
    args.hotwords.write_text("".join(f"{word}\n" for word in sorted(hotwords)), encoding="utf-8")
    method = "modified_beam_search"
    rows = bug_rows + clean_rows
    plain = recognizer(args.model_dir, args.report.parent, config, case, method)
    hypotheses = decode_rows(rows, args.data_dir, plain, config["decode"]["batch_size"])
    baseline = summarize(score(row["reference"], hypotheses[row["audio_idx"]], case, known) for row in bug_rows)
    if baseline["cs_wer"] is None:
        raise ValueError("Bug-mining has no scored CS reference words.")
    baseline_clean = {
        row["audio_idx"] for row in clean_rows
        if summarize([score(row["reference"], hypotheses[row["audio_idx"]], case, known)])["wer_global"] == 0
    }
    results = []
    for strength in config["decode"]["hotwords_scores"]:
        rec = recognizer(args.model_dir, args.report.parent, config, case, method, args.hotwords, strength)
        hypotheses = decode_rows(rows, args.data_dir, rec, config["decode"]["batch_size"])
        bug = summarize(score(row["reference"], hypotheses[row["audio_idx"]], case, known) for row in bug_rows)
        bad = false_triggers = 0
        for row in clean_rows:
            index = row["audio_idx"]
            hyp = hypotheses[index]
            if index in baseline_clean and tokens(row["reference"], case) != tokens(hyp, case):
                bad += 1
            false_triggers += bool(hotwords & (set(tokens(hyp, case)) - set(tokens(row["reference"], case))))
        results.append({
            "strength": strength, "cs_wer": bug["cs_wer"], "gain": baseline["cs_wer"] - bug["cs_wer"],
            "damage": bad / len(clean_rows), "false_triggers": false_triggers / len(clean_rows),
        })
    eligible = [row for row in results if row["damage"] < 0.005 and row["gain"] > 0]
    chosen = max(eligible, key=lambda row: (row["gain"], -row["damage"], -row["strength"]), default=None)
    text = [
        "# Phase 3 hotword sweep",
        "",
        f"Unbiased beam CS-WER: {baseline['cs_wer']:.2%}.",
        f"Terms used: {len(hotwords)}. Omitted unknown BPE terms: {len(candidates - hotwords)}.",
        f"Clean-control utterances already changed by unbiased beam: {len(clean_rows) - len(baseline_clean)}.",
        "The unbiased beam and hotword runs use the same batches over both development splits.",
        "",
        "| Score | CS-WER | CS-WER gain | Newly changed clean | False hotword triggers | Pareto frontier |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for row in results:
        dominated = any(
            other["gain"] >= row["gain"] and other["damage"] <= row["damage"]
            and (other["gain"] > row["gain"] or other["damage"] < row["damage"]) for other in results
        )
        text.append(
            f"| {row['strength']} | {row['cs_wer']:.2%} | {row['gain']:.2%} | {row['damage']:.2%} | "
            f"{row['false_triggers']:.2%} | {'no' if dominated else 'yes'} |"
        )
    text.extend(["", f"Selected score: {chosen['strength']}." if chosen else "No score improved CS-WER within the clean damage budget."])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--hotwords", type=Path, default=ROOT / "data/hotwords/domain_hotwords.txt")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase3_hotwords.md")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
