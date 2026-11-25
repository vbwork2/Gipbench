"""Audit pinned source cards and prepare a local provenance-review queue."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from gipbench.core import release as R
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO

KEYS = ("vietspeech", "vss", "code_switch_om", "lsvsc")


def om_queue(shard, spec, out):
    import pyarrow.parquet as pq

    if R.sha256(shard) != spec["pilot_sha256"]:
        raise ValueError("OM pilot shard differs from the registered revision")
    fields = ["id", "youtube_url", "channel", "duration", "label_source", "text", "audio", "audio_sha256",
              "recording_rights_evidence", "reviewer", "speaker", "domain", "speech_type", "language", "corrected_text", "decision"]
    seen, videos = set(), set()
    seconds = queued = 0.0
    with (out / "om_review.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=16, columns=fields[:7] + ["youtube_id"]):
            for row in batch.to_pylist():
                data = row["audio"]["bytes"]
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                seconds += row["duration"]
                video = row.pop("youtube_id")
                if video in videos or len(videos) >= 24:
                    continue
                videos.add(video)
                audio = out / f"om-{digest[:24]}.flac"
                audio.write_bytes(data)
                writer.writerow(dict(row, audio=audio.name, audio_sha256=digest))
                queued += row["duration"]
    return dict(unique_recordings=len(seen), acquired_hours=seconds / 3600, queued_rows=len(videos),
                queued_hours=queued / 3600, reviewed_hours=0, eligible_hours_added=0, sampled_hours=0,
                revision=spec["revision"], shard_sha256=spec["pilot_sha256"])


def source_cards(sources):
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    api = HfApi()
    cards = {}
    for key in KEYS:
        spec = sources[key]
        try:
            info = api.dataset_info(spec["repo"], revision=spec["revision"], token=False)
        except HfHubHTTPError as exc:
            cards[key] = {"repo": spec["repo"], "requested_revision": spec["revision"],
                          "http_status": exc.response.status_code if exc.response is not None else None}
            continue
        cards[key] = {"repo": info.id, "revision": info.sha, "gated": info.gated, "private": info.private,
                      "card_license": info.card_data.license if info.card_data else None}
    return cards


def review_queue(sources, out):
    fields = ("dataset", "id", "source", "speaker", "domain", "duration", "path", "label_source",
              "recording_rights", "verified_speaker", "verified_domain", "transcript_decision")
    manifest = REPO / "data/work/vietnote/pilots/lsvsc/metadata_screened.jsonl"
    manifests = {"lsvsc": manifest, "vss": R.resolve(sources["vss"]["local_pool"]).manifest}
    summary = {}
    with (out / "provenance_review.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, path in manifests.items():
            found = []
            for row in read_jsonl(path):
                audio = Path(row["_audio_path"])
                if not audio.is_file() or row.get("domain") == "medical":
                    continue
                found.append(row)
                writer.writerow({"dataset": key, **{field: row.get(field) for field in fields[1:6]},
                                 "path": audio.resolve().relative_to(REPO).as_posix(),
                                 "label_source": row.get("label_source")})
            summary[key] = {"manifest_sha256": R.sha256(path), "queued_rows": len(found),
                            "queued_hours": sum(float(row["duration"]) for row in found) / 3600,
                            "domains": dict(Counter(row.get("domain") or "unknown" for row in found)),
                            "eligible_hours_added": 0}
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--om-shard", type=Path)
    args = parser.parse_args()
    registry = REPO / "configs/dataset/vietnote_sources.yaml"
    sources = yaml.safe_load(registry.read_text(encoding="utf-8"))["sources"]
    cards = {} if args.offline else source_cards(sources)
    args.out.mkdir(parents=True, exist_ok=False)
    status = {"source_registry_sha256": R.sha256(registry), "cards": cards, "local_queue": review_queue(sources, args.out)}
    if args.om_shard:
        status["local_queue"]["code_switch_om"] = om_queue(args.om_shard, sources["code_switch_om"], args.out)
    (args.out / "result.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
