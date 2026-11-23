"""Create disjoint FPT manifests when speaker assignments are available."""

from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from random import Random
import csv
import io
import json
import stat

from gipformer_mech.fpt import read_rows, write_rows
from gipformer_mech.model_audit import checksum


def speaker_assignments(path: Path) -> dict[str, str]:
    raw = path.read_text(encoding="utf-8-sig")
    with io.StringIO(raw) as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames != ["utt_id", "speaker_id"]:
            raise ValueError("Speaker TSV header must be utt_id followed by speaker_id.")
        speakers = {}
        for line in reader:
            utterance = (line["utt_id"] or "").strip()
            speaker = (line["speaker_id"] or "").strip()
            if not utterance or speaker.casefold() in {"", "unknown", "unk", "none", "n/a"} or utterance in speakers:
                raise ValueError("Speaker TSV contains an unknown speaker, missing field or duplicate utterance.")
            speakers[utterance] = speaker
    if len(set(speakers.values())) < 3:
        raise ValueError("At least three verified speaker groups are needed for the three splits.")
    return speakers


def select(rows: list[dict], sizes: dict, seed: int) -> dict[str, list[dict]]:
    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker_id"]].append(row)
    random = Random(seed)
    speakers = sorted(by_speaker)
    heldout_size = sizes["heldout_size"]
    minimum_cs = sizes["heldout_min_cs"]
    clean_size = sizes["clean_size"]
    bug_size = sizes["bug_size"]
    confusions = Counter(
        (err["ref"], err["hyp"]) for row in rows for err in row["score_counts"]["cs_word_errors"]
    )
    for _ in range(128):
        random.shuffle(speakers)
        held_pool, bug_pool, clean_pool = [], [], []
        held_cs = 0
        for speaker in speakers:
            group = by_speaker[speaker]
            if len(held_pool) < heldout_size or held_cs < minimum_cs:
                held_pool.extend(group)
                held_cs += sum(bool(row["cs_tokens"]) for row in group)
            elif len(bug_pool) < bug_size:
                bug_pool.extend(row for row in group if row["baseline_wer"] > 0)
            else:
                clean_pool.extend(row for row in group if row["baseline_wer"] == 0 and not row["cs_tokens"])
        if min(len(held_pool) - heldout_size, held_cs - minimum_cs,
               len(bug_pool) - bug_size, len(clean_pool) - clean_size) < 0:
            continue
        reserved = random.sample([row for row in held_pool if row["cs_tokens"]], minimum_cs)
        reserved_ids = {row["utt_id"] for row in reserved}
        held = reserved + random.sample(
            [row for row in held_pool if row["utt_id"] not in reserved_ids], heldout_size - minimum_cs
        )
        random.shuffle(bug_pool)
        bug_pool.sort(key=lambda row: (
            bool(row["score_counts"]["cs_errors"]),
            max((confusions[(err["ref"], err["hyp"])] for err in row["score_counts"]["cs_word_errors"]), default=0),
            row["baseline_wer"],
        ), reverse=True)
        return {
            "bug_mining": bug_pool[:bug_size],
            "clean_control": random.sample(clean_pool, clean_size),
            "heldout_test": sorted(held, key=lambda row: row["utt_id"]),
        }
    raise ValueError("The available speaker groups cannot produce the requested development sizes and heldout CS quota.")


def seal(directory: Path, speaker_map: Path, config: dict, rejected_labels: Path | None) -> None:
    path = directory / "heldout_test.jsonl"
    lock = directory / "heldout_lock.json"
    if path.exists() or lock.exists():
        raise FileExistsError("A heldout split already exists in this run directory.")
    pool_path = directory / "full_scored.jsonl"
    rows = read_rows(pool_path)
    speakers = speaker_assignments(speaker_map)
    rejected = {row["utt_id"]: row["reason"] for row in read_rows(rejected_labels)} if rejected_labels else {}
    missing = {row["utt_id"] for row in rows} - speakers.keys() - rejected.keys()
    if missing:
        raise ValueError(f"Speaker TSV lacks assignments for {len(missing)} scored utterances.")
    pool = [
        {**row, "speaker_id": speakers[row["utt_id"]]}
        for row in rows if row["utt_id"] not in rejected
    ]
    splits = select(pool, config["dataset"], config["seed"])
    for left, right in combinations(splits.values(), 2):
        for key in ("utt_id", "speaker_id", "audio_sha256"):
            if {row[key] for row in left} & {row[key] for row in right}:
                raise ValueError(f"Overlap detected for {key}.")
    held_speakers = {row["speaker_id"] for row in splits["heldout_test"]}
    audit_path = directory / "label_audit_dropped.jsonl"
    dropped = read_rows(audit_path)
    dropped.extend({**row, "reason": rejected[row["utt_id"]]} for row in rows if row["utt_id"] in rejected)
    write_rows(audit_path, (row for row in dropped if speakers.get(row["utt_id"]) not in held_speakers))
    write_rows(pool_path, (row for row in pool if row["speaker_id"] not in held_speakers))
    for split, selected in splits.items():
        write_rows(directory / f"{split}.jsonl", selected)
    lock.write_text(json.dumps({
        "speaker_evidence": "verified",
        "speaker_map_sha256": checksum(speaker_map),
        "heldout_sha256": checksum(path),
        "development_sha256": {
            split: checksum(directory / f"{split}.jsonl") for split in ("bug_mining", "clean_control")
        },
        "development_pool_sha256": checksum(pool_path),
        "heldout_size": len(splits["heldout_test"]),
        "heldout_cs_count": sum(bool(row["cs_tokens"]) for row in splits["heldout_test"]),
        "dataset_revision": config["dataset"]["revision"],
    }, indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IREAD)
    lock.chmod(stat.S_IREAD)


def development_rows(directory: Path, split: str) -> list[dict]:
    if split not in {"bug_mining", "clean_control"}:
        raise ValueError(f"Development access does not support {split}.")
    lock = json.loads((directory / "heldout_lock.json").read_text(encoding="utf-8"))
    manifest = directory / f"{split}.jsonl"
    if checksum(manifest) != lock["development_sha256"][split]:
        raise ValueError(f"The {split} manifest differs from the sealed copy.")
    return read_rows(manifest)
