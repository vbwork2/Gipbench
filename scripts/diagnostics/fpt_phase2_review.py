"""Review a stratified sample of inconclusive FPT errors on development audio."""

import argparse
from collections import Counter, defaultdict, deque
import json
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipbench.core.align import align
from gipformer_mech.fpt import audio_indices, manifest_waveform, parquet_files, read_rows, settings, write_rows
from gipformer_mech.model_audit import checksum
from gipformer_mech.rnnt_loader import resample
from gipformer_mech.splits import development_rows
from gipformer_mech.text_norm import tokens

MODES = ("full", "reference_full", "acoustic")
RADII = (1, 2, 4)
PROFILES = ("all", "some", "none")


def profile(ranks: list[int]) -> str:
    hits = sum(rank <= 10 for rank in ranks)
    if hits == len(ranks):
        return "all"
    return "some" if hits else "none"


def sample_cases(diagnosis: dict, rows: dict[str, dict], case: str, cs_terms: list[str],
                 count: int, seed: int) -> tuple[list[dict], Counter]:
    groups = defaultdict(list)
    known = {term.casefold() for term in cs_terms}
    for utterance in diagnosis["findings"]:
        row = rows[utterance["utt_id"]]
        for error in utterance["errors"]:
            if error["category"] != "INCONCLUSIVE":
                continue
            index = error["word_index"]
            ref = tokens(row["reference"], case)
            if ref[index] != error["reference_word"]:
                raise ValueError(f"Reference mismatch in {row['utt_id']} at word {index}.")
            pieces = [{
                "id": piece["target_id"], "text": piece["target_piece"], "frame": piece["frame"],
                "published_ranks": {mode: piece["stacks"][-1][mode]["best_rank"] for mode in MODES},
            } for piece in error["pieces"]]
            group = profile([piece["published_ranks"]["full"] for piece in pieces])
            groups[group].append({
                "utt_id": row["utt_id"], "word_index": index, "word": ref[index], "profile": group,
                "label_source": "configured" if ref[index].casefold() in known else "orthographic",
                "pieces": pieces, "reference": row["reference"], "gipformer": row["baseline_hyp"],
            })
    population = Counter({key: len(values) for key, values in groups.items()})
    random = Random(seed)
    queues = {}
    for group in PROFILES:
        values = sorted(groups[group], key=lambda item: (item["utt_id"], item["word_index"]))
        random.shuffle(values)
        queues[group] = deque(values)
    sample, chosen = [], set()
    while len(sample) < count and any(queues.values()):
        for group in PROFILES:
            queue = queues[group]
            while queue and queue[0]["utt_id"] in chosen:
                queue.popleft()
            if queue:
                selected = queue.popleft()
                sample.append(selected)
                chosen.add(selected["utt_id"])
                if len(sample) == count:
                    break
    if len(sample) != count:
        raise ValueError(f"Requested {count} distinct utterances; only {len(sample)} available.")
    return sample, population


def ordered_support(options: list[list[int]]) -> list[int] | None:
    """Find top-k support on distinct, increasing frames for all pieces."""
    path = []
    for frames in options:
        after = path[-1] if path else -1
        earliest = next((frame for frame in frames if frame > after), None)
        if earliest is None:
            return None
        path.append(earliest)
    return path


def summarize_window(pieces: list[dict], radius: int, mode: str) -> dict:
    options, best = [], []
    for piece in pieces:
        frames_ranks = [
            (frame, rank) for frame, rank in zip(piece["frames"], piece[mode], strict=True)
            if abs(frame - piece["frame"]) <= radius
        ]
        best.append(min(rank for _, rank in frames_ranks))
        options.append([frame for frame, rank in frames_ranks if rank <= 10])
    return {"best_ranks": best, "ordered_top10_frames": ordered_support(options)}


def prepare(args, rows, cfg, case) -> None:
    import soundfile

    diagnosis = json.loads(args.diagnosis.read_text(encoding="utf-8"))
    sample, population = sample_cases(diagnosis, rows, case, cfg["text"]["cs_terms"], args.count, cfg["seed"])
    args.review_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.review_dir / "sample.jsonl"
    if manifest.is_file() and read_rows(manifest) != sample:
        raise ValueError("The review directory already contains a different sample. Use a new --review-dir.")
    audio_dir = args.review_dir / "audio"
    audio_dir.mkdir(exist_ok=True)
    selected = {rows[item["utt_id"]]["audio_idx"]: item for item in sample}
    exported = set()
    for index, raw in audio_indices(parquet_files(args.data_dir), set(selected)):
        item = selected[index]
        waveform, rate = manifest_waveform(rows[item["utt_id"]], raw)
        path = audio_dir / f"{item['utt_id']}.wav"
        soundfile.write(path, resample(waveform, rate, 16000), 16000, subtype="PCM_16")
        exported.add(index)
    if exported != selected.keys():
        raise ValueError("A selected development recording could not be exported.")
    write_rows(manifest, sample)
    write_rows(args.review_dir / "oracle_input.jsonl", ({
        "id": item["utt_id"], "duration": rows[item["utt_id"]]["duration_sec"],
        "_audio_path": (audio_dir / f"{item['utt_id']}.wav").relative_to(ROOT).as_posix(),
    } for item in sample))
    (args.review_dir / "population.json").write_text(json.dumps(population, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(sample)} distinct development recordings at {args.review_dir}")


def validate_sample(sample: list[dict], rows: dict[str, dict], case: str) -> None:
    for item in sample:
        row = rows[item["utt_id"]]
        words = tokens(row["reference"], case)
        if row["reference"] != item["reference"] or words[item["word_index"]] != item["word"]:
            raise ValueError(f"Sample reference changed for {item['utt_id']}.")


def probe(args, rows, cfg, case) -> None:
    import torch

    from gipformer_mech.logit_lens import greedy_contexts
    from gipformer_mech.rnnt_loader import encode, load_model

    sample = read_rows(args.review_dir / "sample.jsonl")
    validate_sample(sample, rows, case)
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["decode"]["threads"])
    model, sp = load_model(args.model_dir, args.icefall, model_map, args.device)
    selected = {rows[item["utt_id"]]["audio_idx"]: item for item in sample}
    outputs = {}
    context = model_map["predictor"]["context_size"]
    initial = [-1] * (context - 1) + [model_map["blank_id"]]
    with torch.inference_mode():
        for index, raw in audio_indices(parquet_files(args.data_dir), set(selected)):
            item = selected[index]
            waveform, sample_rate = manifest_waveform(rows[item["utt_id"]], raw)
            target_rate = model_map["sample_rate"]
            final, lengths = encode(model, resample(waveform, sample_rate, target_rate), model_map, args.device)
            predictor = greedy_contexts(model, final, lengths, initial)
            acoustic = model.joiner.encoder_proj(final[0])
            words = tokens(item["reference"], case)
            prefix = initial + sp.encode(" ".join(words[:item["word_index"]]), out_type=int)
            target_ids = sp.encode(item["word"], out_type=int)
            if target_ids != [piece["id"] for piece in item["pieces"]]:
                raise ValueError(f"Tokenization changed for {item['utt_id']}.")
            pieces = []
            frame_count = int(lengths[0])
            for piece in item["pieces"]:
                anchor, target = piece["frame"], piece["id"]
                if not 0 <= anchor < frame_count:
                    raise ValueError(f"Anchor {anchor} is outside the encoder output for {item['utt_id']}.")
                frames = list(range(max(0, anchor - max(RADII)), min(frame_count, anchor + max(RADII) + 1)))
                dec_input = torch.tensor([prefix[-context:]], device=final.device)
                ref_predictor = model.joiner.decoder_proj(model.decoder(dec_input, need_pad=False)[0, 0])
                projected = acoustic[frames]
                sums = (projected + predictor[frames], projected + ref_predictor, projected)
                ranks = {}
                for mode, features in zip(MODES, sums, strict=True):
                    logits = model.joiner.output_linear(torch.tanh(features))
                    ranks[mode] = ((logits > logits[:, target:target + 1]).sum(-1) + 1).tolist()
                pieces.append({"frame": anchor, "frames": frames, **ranks})
                prefix.append(target)
            outputs[item["utt_id"]] = {
                "utt_id": item["utt_id"], "frame_count": frame_count, "traces": pieces,
                "windows": {
                    str(radius): {mode: summarize_window(pieces, radius, mode) for mode in MODES} for radius in RADII
                },
            }
    write_rows(args.review_dir / "temporal_probe.jsonl", (outputs[item["utt_id"]] for item in sample))
    print(f"Replayed {len(outputs)} recordings and evaluated all three windows.")


def reference_status(item: dict, hyp: str, case: str) -> tuple[str, str]:
    reference, hypothesis = tokens(item["reference"], case), tokens(hyp, case)
    index = item["word_index"]
    for op, first, second in align(reference, hypothesis):
        if first == index:
            return op, hypothesis[second] if second is not None else ""
    raise ValueError(f"No aligned target word for {item['utt_id']}.")


def report(args, rows, cfg, case) -> None:
    sample = read_rows(args.review_dir / "sample.jsonl")
    validate_sample(sample, rows, case)
    population = json.loads((args.review_dir / "population.json").read_text(encoding="utf-8"))
    replay = {row["utt_id"]: row for row in read_rows(args.review_dir / "temporal_probe.jsonl")}
    probes = {key: row["windows"] for key, row in replay.items()}
    if probes.keys() != {item["utt_id"] for item in sample}:
        raise ValueError("Temporal probe results must match the sampled utterances.")
    oracle_results = {}
    oracle_sources = {}
    wav_hashes = {
        item["utt_id"]: checksum(args.review_dir / "audio" / f"{item['utt_id']}.wav") for item in sample
    }
    for name in ("qwen", "whisper"):
        path = args.review_dir / f"{name}.jsonl"
        if path.is_file():
            predictions = read_rows(path)
            hyps = {row["id"]: row["hyp"] for row in predictions}
            if hyps.keys() != probes.keys() or len(predictions) != len(sample):
                raise ValueError(f"The {name} transcriptions do not cover the sampled utterances.")
            if any(row["audio_sha256"] != wav_hashes[row["id"]] for row in predictions):
                raise ValueError(f"The {name} output appears to come from a different set of audio files.")
            sources = {Path(row["model"]).name for row in predictions}
            if len(sources) != 1:
                raise ValueError(f"Multiple model snapshots in the {name} output.")
            oracle_sources[name] = sources.pop()
            oracle_results[name] = {item["utt_id"]: reference_status(item, hyps[item["utt_id"]], case) for item in sample}

    selected = Counter(item["profile"] for item in sample)
    profile_labels = {"all": "All pieces", "some": "Some pieces", "none": "No pieces"}
    windows = []
    consensus = Counter()
    corroborated = []
    reproduced = 0
    unordered = 0
    top_one = 0
    reference_only = 0
    for item in sample:
        by_radius = probes[item["utt_id"]]
        narrow = by_radius["1"]
        reproduced += all(
            narrow[mode]["best_ranks"] == [piece["published_ranks"][mode] for piece in item["pieces"]]
            for mode in MODES
        )
        unordered += item["profile"] == "all" and narrow["full"]["ordered_top10_frames"] is None
        wide = by_radius[str(max(RADII))]
        reference_only += (
            wide["reference_full"]["ordered_top10_frames"] is not None
            and wide["full"]["ordered_top10_frames"] is None
        )
        top_one += ordered_support([
            [frame for frame, rank in zip(piece["frames"], piece["full"], strict=True) if rank == 1]
            for piece in replay[item["utt_id"]]["traces"]
        ]) is not None
        windows.append(by_radius)
        judgments = [result[item["utt_id"]] for result in oracle_results.values()]
        if len(judgments) == 2:
            matches = sum(op == "C" for op, _ in judgments)
            if matches:
                consensus["both" if matches == 2 else "one"] += 1
                if matches == 2 and narrow["full"]["ordered_top10_frames"] is not None:
                    corroborated.append(item)
            else:
                first, second = judgments
                consensus["same_alternative" if first == second else "different_alternatives"] += 1

    lines = [
        "# Follow-up review of inconclusive FPT errors",
        "",
        f"Development-only audit on {cfg['dataset']['repo']} at revision {cfg['dataset']['revision']}. "
        f"Seed {cfg['seed']} selected {len(sample)} distinct recordings from {sum(population.values())} inconclusive spans.",
        "",
        "| Final-stack top-10 pieces at ±1 frame | Inconclusive population | Reviewed recordings |",
        "|---|---:|---:|",
        *[f"| {profile_labels[key]} | {population.get(key, 0)} | {selected[key]} |" for key in PROFILES],
        "",
        f"The replay reproduced all published final-stack ranks in {reproduced}/{len(sample)} recordings. "
        f"At ±1, {selected['all'] - unordered}/{selected['all']} cases with every piece individually in the top ten "
        "also had top-ten support on distinct increasing frames.",
        "",
        "| Window | Greedy-prefix ordered top 10 | Reference-prefix ordered top 10 | Acoustic ordered top 10 |",
        "|---|---:|---:|---:|",
    ]
    for radius in RADII:
        counts = [sum(run[str(radius)][mode]["ordered_top10_frames"] is not None for run in windows) for mode in MODES]
        lines.append(f"| ±{radius} | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines.extend([
        "",
        f"At ±{max(RADII)} frames, {top_one}/{len(sample)} recordings had ordered rank-one support "
        f"with the greedy prefix. {reference_only}/{len(sample)} had ordered top-ten support exclusively "
        "with the reference prefix.",
        "",
        "Ordered top-ten support requires a distinct increasing frame for each target piece within its window. "
        "It measures the availability of the pieces, while complete greedy recognition also depends on their scores "
        "and the intervening emissions.",
        "",
        "## Independent ASR cross-check",
        "",
    ])
    names = {"qwen": "Qwen3-ASR-0.6B", "whisper": "faster-whisper large-v3"}
    for name, result in oracle_results.items():
        hits = sum(op == "C" for op, _ in result.values())
        lines.append(
            f"- {names[name]} ({oracle_sources[name]}). The aligned hypothesis matched the normalized reference "
            f"word in {hits}/{len(sample)} recordings."
        )
    if len(oracle_results) == 2:
        lines.append(
            f"- Both matched in {consensus['both']}, one matched in {consensus['one']}, "
            f"and neither matched in {consensus['same_alternative'] + consensus['different_alternatives']}. "
            f"Of the last group, {consensus['same_alternative']} had the same aligned substitution or deletion."
        )
    supplemental = args.review_dir / "qwen_vi_probe.jsonl"
    if supplemental.is_file():
        if "qwen" not in oracle_results:
            raise ValueError("The language probe requires the primary Qwen results.")
        by_id = {item["utt_id"]: item for item in sample}
        for prediction in read_rows(supplemental):
            key = prediction["id"]
            if prediction["audio_sha256"] != wav_hashes.get(key):
                raise ValueError(f"Supplemental Qwen audio checksum mismatch for {key}.")
            if Path(prediction["model"]).name != oracle_sources["qwen"]:
                raise ValueError("The language probe must use the primary Qwen snapshot.")
            item = by_id[key]
            initial = oracle_results["qwen"][key][0] == "C"
            forced = reference_status(item, prediction["hyp"], case)[0] == "C"
            lines.append(
                f"- Qwen language sensitivity on {key} ({item['word']}). "
                f"Automatic language {'matched' if initial else 'missed'} the target, "
                f"and the Vietnamese setting {'matched' if forced else 'missed'} it. "
                "This is a targeted follow-up outside the aggregate Qwen count."
            )
    configured = sum(item["label_source"] == "configured" for item in sample)
    lines.extend([
        "",
        f"The orthographic rule selected {len(sample) - configured}/{len(sample)} target labels. "
        "ASR agreement triages label review. A human listening verdict is still pending.",
    ])
    if corroborated:
        source = json.loads(args.diagnosis.read_text(encoding="utf-8"))
        original = {
            (row["utt_id"], error["word_index"]): error
            for row in source["findings"] for error in row["errors"]
        }
        lines.extend([
            "",
            "## Ranked competitors on corroborated targets",
            "",
            "Both independent ASRs matched these targets, and all pieces had ordered top-ten support with "
            "the greedy prefix. Each competitor leads at the best target frame within ±1.",
            "",
            "| Recording | Target word | Pieces below rank one and their leading competitor |",
            "|---|---|---|",
        ])
        for item in corroborated:
            error = original[item["utt_id"], item["word_index"]]
            rivals = []
            for piece in error["pieces"]:
                peak = piece["stacks"][-1]["full"]
                if peak["best_rank"] > 1:
                    rival = peak["top10"][0]["piece"]
                    rivals.append(f"`{piece['target_piece']}` (rank {peak['best_rank']}) vs `{rival}`")
            lines.append(f"| {item['utt_id']} | {item['word']} | {', '.join(rivals) or 'All pieces reach rank one'} |")
    lines.extend([
        "",
        "## Sample details",
        "",
        "| Recording | Target | Lexical rule | Greedy-prefix ordered top 10 at ±1 / ±4 | Qwen | Whisper |",
        "|---|---|---|---|---|---|",
    ])
    for item in sample:
        key = item["utt_id"]
        temporal = ["yes" if probes[key][str(radius)]["full"]["ordered_top10_frames"] else "no" for radius in (1, 4)]
        labels = []
        for name in ("qwen", "whisper"):
            op, word = oracle_results.get(name, {}).get(key, ("", ""))
            labels.append({"C": "match", "S": f"substitution ({word})", "D": "deletion"}.get(op, "pending"))
        lines.append(f"| {key} | {item['word']} | {item['label_source']} | {' / '.join(temporal)} | {' | '.join(labels)} |")
    lines.extend([
        "",
        "Full references, baseline hypotheses, transcriptions and WAVs are in the local review directory. "
        "The recording ID matches the WAV basename.",
        "",
        "The sample deliberately overrepresents spans with no final-stack top-ten pieces. Its counts describe this review "
        "sample. The original Phase 2 population remains the input to the encoder, Joiner and fine-tuning decisions.",
    ])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.report}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "report"))
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--diagnosis", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--review-dir", type=Path, default=ROOT / "artifacts/phase2_review")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase2_inconclusive_review.md")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--icefall", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--count", type=int, default=24)
    args = parser.parse_args()
    args.review_dir = args.review_dir.resolve()
    if args.stage == "prepare" and not args.review_dir.is_relative_to(ROOT):
        parser.error("--review-dir must be inside the worktree for portable audio paths.")
    if args.stage == "probe" and args.icefall is None:
        parser.error("--icefall is required for a temporal probe.")
    if args.count < 1:
        parser.error("--count must be positive.")
    config, case = settings()
    rows = {row["utt_id"]: row for row in development_rows(args.manifests, "bug_mining")}
    {"prepare": prepare, "probe": probe, "report": report}[args.stage](args, rows, config, case)


if __name__ == "__main__":
    main()
