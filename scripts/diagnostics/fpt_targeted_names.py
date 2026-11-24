"""Verify the reviewed-name preset with the regular Gipformer ONNX adapter."""

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import soundfile

from gipbench.bench.transcribe import load_model_cfg
from gipbench.core.align import align
from gipbench.infer.adapters import build_adapter
from gipbench.infer.adapters.base import run_manifest
from gipformer_mech.fpt import audio_indices, manifest_waveform, parquet_files, read_rows, settings
from gipformer_mech.metrics import score, summarize
from gipformer_mech.text_norm import tokens
from gipformer_mech.splits import development_rows


def phrase_counts(words: list[str], phrases: list[tuple[str, ...]]) -> Counter:
    counts = Counter()
    for phrase in phrases:
        size = len(phrase)
        counts[phrase] = sum(tuple(words[i:i + size]) == phrase for i in range(len(words) - size + 1))
    return counts


def aligned_phrase(ref: list[str], hyp: list[str], start: int, count: int) -> tuple[bool, str]:
    indices = [
        (op, hi) for op, ri, hi in align(ref, hyp) if ri is not None and start <= ri < start + count
    ]
    if len(indices) != count:
        raise ValueError("Target phrase is outside the reference transcript.")
    emitted = [hi for _, hi in indices if hi is not None]
    if not emitted:
        return False, ""
    # Include the complete span, so a phrase with an intervening insertion does not pass.
    surface = " ".join(hyp[emitted[0]:emitted[-1] + 1])
    success = all(op == "C" for op, _ in indices) and emitted[-1] - emitted[0] + 1 == count
    return success, surface


def comparison(args) -> bool:
    pipeline, _ = settings()
    config = load_model_cfg(str(args.preset))
    if args.model_dir:
        config["model"]["dir"] = str(args.model_dir.resolve())
    case = config["decode"]["case"]
    phrases = [
        tuple(tokens(line, case))
        for line in Path(config["context_vocab"]["file"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not phrases or any(not phrase for phrase in phrases):
        raise ValueError("The reviewed name list must contain nonempty name phrases.")
    sample = read_rows(args.review_sample)
    bug = development_rows(args.manifests, "bug_mining")
    clean = development_rows(args.manifests, "clean_control")
    rows = bug + clean
    by_id = {row["utt_id"]: row for row in rows}
    by_index = {row["audio_idx"]: row for row in rows}
    if len(by_id) != len(rows) or len(by_index) != len(rows):
        raise ValueError("Development split recordings overlap.")
    label_words = {row["utt_id"]: tokens(row["reference"], case) for row in rows}

    targets = []
    for item in sample:
        index = item["word_index"]
        words = label_words[item["utt_id"]]
        matches = [phrase for phrase in phrases if tuple(words[index:index + len(phrase)]) == phrase]
        if matches:
            targets.append((item["utt_id"], index, matches[0]))
    if Counter(phrase for _, _, phrase in targets) != Counter(phrases):
        raise ValueError("Each name phrase must point to one reviewed development target.")

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    exported = set()
    for index, raw in audio_indices(parquet_files(args.data_dir), set(by_index)):
        row = by_index[index]
        waveform, rate = manifest_waveform(row, raw)
        soundfile.write(audio_dir / f"{row['utt_id']}.wav", waveform, rate, subtype="FLOAT")
        exported.add(index)
    if exported != by_index.keys():
        raise ValueError("Some development recordings could not be exported.")
    manifest = [
        {"id": row["utt_id"], "_audio_path": str(audio_dir / f"{row['utt_id']}.wav"), "duration": row["duration_sec"]}
        for row in rows
    ]

    base = {**config, "system": "gip15_targeted_names_control", "inference_mode": "open_vocab"}
    del base["context_vocab"]
    hypotheses = {}
    for name, model in (("baseline", base), ("candidate", config)):
        out = args.output_dir / name
        adapter = build_adapter(model, out)

        def progress(done, total, label=name):
            if done % 100 == 0 or done == total:
                print(f"{label}: {done}/{total} recordings", flush=True)

        run_manifest(adapter, manifest, out, batch=int(model["batch"]), resume=False, progress=progress)
        hypotheses[name] = {row["id"]: row["hyp"] for row in read_rows(out / "hyps.jsonl")}
    before, after = (
        {uid: tokens(hypothesis, case) for uid, hypothesis in hypotheses[name].items()}
        for name in ("baseline", "candidate")
    )

    decisions = []
    for uid, position, phrase in targets:
        old_match, old = aligned_phrase(label_words[uid], before[uid], position, len(phrase))
        new_match, new = aligned_phrase(label_words[uid], after[uid], position, len(phrase))
        decisions.append((uid, " ".join(phrase), old, new, old_match, new_match))

    degraded = false_triggers = 0
    for row in clean:
        uid = row["utt_id"]
        ref = label_words[uid]
        degraded += before[uid] == ref and after[uid] != ref
        false_triggers += bool(phrase_counts(after[uid], phrases) - (
            phrase_counts(ref, phrases) | phrase_counts(before[uid], phrases)
        ))
    metrics = {}
    for split, records in (("Bug mining", bug), ("Clean control", clean)):
        for name in ("baseline", "candidate"):
            metrics[split, name] = summarize(
                score(row["reference"], hypotheses[name][row["utt_id"]], case, pipeline["text"]["cs_terms"])
                for row in records
            )
    clean_insertions = metrics["Clean control", "candidate"]["ins_rate"] - metrics["Clean control", "baseline"]["ins_rate"]
    passed = (
        all(row[-1] for row in decisions)
        and degraded / len(clean) < 0.005 and clean_insertions < 0.003 and false_triggers == 0
    )
    lines = [
        "# Gipformer FPT reviewed-name development probe",
        "",
        f"In-sample probe, criteria **{'met' if passed else 'not met'}**. "
        f"Same INT8 decoder and batch size of {config['batch']} in both runs. The candidate supplies {len(phrases)} reviewed name "
        f"phrases at hotword score {config['context_vocab']['score']:g}.",
        "",
        "| Development recording | Target phrase | Baseline span | Candidate span |",
        "|---|---|---|---|",
        *[f"| {uid} | {phrase} | {old or '(deleted)'} | {new or '(deleted)'} |"
          for uid, phrase, old, new, _, _ in decisions],
        "",
        "| Split | Decoder | WER | N-WER | CS-WER | Insertion rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (split, name), summary in metrics.items():
        values = [summary[key] for key in ("wer_global", "n_wer", "cs_wer", "ins_rate")]
        formatted = " | ".join(f"{value:.2%}" if value is not None else "N/A" for value in values)
        lines.append(f"| {split} | {name} | {formatted} |")
    lines.extend([
        "",
        f"- Exact target phrases {sum(row[-2] for row in decisions)}/{len(decisions)} to "
        f"{sum(row[-1] for row in decisions)}/{len(decisions)}.",
        f"- Newly incorrect clean recordings {degraded}/{len(clean)}.",
        f"- Clean recordings with a newly introduced erroneous target phrase {false_triggers}/{len(clean)}.",
        f"- Clean insertion-rate change {clean_insertions * 100:+.3f} percentage points.",
        "",
        "The name list was selected from the same four reviewed recordings that the target table scores, so the "
        "exact-phrase gain above is in-sample and shows only that the biasing mechanism can reach those spans. It "
        "is not evidence that the preset generalizes. The clean-control rows check for collateral damage. Names "
        "that did not come from these recordings were not tested here. The heldout was outside the run.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--preset", type=Path, default=ROOT / "configs/diagnostics/fpt/gip15_mbs4_fpt_reviewed_names.yaml")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--review-sample", type=Path, default=ROOT / "artifacts/phase2_review/sample.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/targeted_decode/regression")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/fpt_targeted_name_probe.md")
    raise SystemExit(0 if comparison(parser.parse_args()) else 1)


if __name__ == "__main__":
    main()
