"""Evaluate the unmodified greedy and beam decoders on development data."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.fpt import decode_rows, read_rows, recognizer, settings, write_rows
from gipformer_mech.metrics import score, summarize
from gipformer_mech.splits import development_rows

ORDER = ("wer_global", "cer_global", "n_wer", "cs_wer", "ins_rate", "del_rate", "sub_rate")


def table_row(split: str, decoder: str, count: int, summary: dict) -> str:
    fields = [f"{summary[key]:.2%}" if summary[key] is not None else "n/a" for key in ORDER]
    return "| " + " | ".join([split, decoder, str(count), *fields]) + " |"


def evaluate(args: argparse.Namespace) -> None:
    config, case = settings()
    root = args.manifests
    terms = config["text"]["cs_terms"]
    if (root / "heldout_test.jsonl").exists():
        sets = {split: development_rows(root, split) for split in ("bug_mining", "clean_control")}
        provisional = False
    else:
        sets = {"provisional_pool": read_rows(root / "full_scored.jsonl")}
        provisional = True
    text = [
        "# Gipformer 1.5 ONNX baseline",
        "",
        f"Dataset {config['dataset']['repo']} at revision {config['dataset']['revision']}.",
        "",
        "This run covers the provisional scored pool. No heldout results or speaker-disjoint claims apply."
        if provisional else "This run covers development splits. Heldout is reserved for Phase 7.",
        "",
        "| Split | Decoder | Utterances | WER | CER | N-WER | CS-WER | Ins | Del | Sub |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rec = recognizer(args.model_dir, root, config, case, "modified_beam_search")
    for split, rows in sets.items():
        if args.limit:
            rows = rows[:args.limit]
        greedy = summarize(row["score_counts"] for row in rows)
        text.append(table_row(split, "greedy", len(rows), greedy))
        beam_hyps = decode_rows(rows, args.data_dir, rec, config["decode"]["batch_size"])
        beam = []
        for row in rows:
            hyp = beam_hyps[row["audio_idx"]]
            counts = score(row["reference"], hyp, case, terms)
            beam.append({"utt_id": row["utt_id"], "beam_hyp": hyp, "score_counts": counts})
        write_rows(args.prediction_dir / f"{split}_beam.jsonl", beam)
        text.append(table_row(split, f"modified beam ({config['decode']['beam']})", len(rows), summarize(
            row["score_counts"] for row in beam
        )))
        if provisional:
            coverage = Counter(term for row in rows for term in set(row["cs_tokens"]))
            text.extend([
                "",
                f"Lexical CS candidates occur in {sum(bool(row['cs_tokens']) for row in rows):,} utterances.",
                f"Greedy returned zero word errors on {sum(row['baseline_wer'] == 0 for row in rows):,} utterances.",
                "Top detected terms by utterance: " + (
                    ", ".join(f"{word} ({n})" for word, n in coverage.most_common(10)) or "none"
                ),
                "",
                "CS attribution is lexical and candidate labels may need reference review.",
            ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--prediction-dir", type=Path, default=ROOT / "artifacts/baseline")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase1_baseline.md")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive.")
    evaluate(args)


if __name__ == "__main__":
    main()
