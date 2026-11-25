"""Audit released and candidate VietNote data using the GipBench inventory."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.paths import REPO, path
from gipbench.core.textnorm import Catalog
from gipbench.dataset.report import inventory as eda
from gipbench.dataset.synth.backends import analyze

CORPUS_POOLS = (
    "pool/vimedcss_train@v4", "pool/vss_cs@v5", "pool/distill@v6",
    "pool/vivos_train@v3", "pool/cv_other@v3", "pool/confusable_train@v3", "pool/tts_train@e4_v1",
    "pool/vietnote_vimedcss_cs@v1", "pool/vietnote_librispeech_en@v1", "pool/vietnote_fleurs_vi@v1",
)


def identify_audio(rows, known_hashes):
    seen = set()
    unique = []
    missing = 0
    for row in rows:
        p = row["_path"]
        if not p or not Path(p).is_file():
            missing += 1
            continue
        identity = known_hashes.get(p) or p
        if identity not in seen:
            unique.append(row)
            seen.add(identity)
    return unique, missing


def acoustic_sample(rows, count, salt):
    from gipbench.core.audiofp import load_mono

    ordered = sorted(rows, key=lambda r: hashlib.sha256(f"{salt}:{r['id']}".encode()).hexdigest())
    bad = Counter()
    measurements = []
    for row in ordered[:count]:
        try:
            x, sr = load_mono(row["_path"])
            if not len(x):
                bad["empty"] += 1
                continue
            measurements.append({"id": row["id"], **analyze(x.tolist(), sr)})
        except (RuntimeError, OSError, ValueError):
            bad["decode_failed"] += 1
    return {"decoded": len(measurements), "defects": dict(bad), "measurements": measurements}


def fragmentation(terms):
    import sentencepiece as spm

    result = {}
    for name, case, filename in (("vietnote78", str.lower, "vietnote78/bpe.model"),
                                 ("gip15", str.upper, "gip15/bpe.model")):
        p = REPO / "models" / filename
        if not p.is_file():
            continue
        sp = spm.SentencePieceProcessor(model_file=str(p))
        fragments = [sp.encode(case(word), out_type=int) for word in terms]
        result[name] = {
            "unique_words": len(fragments), "pieces_per_word": eda.quantile([len(x) for x in fragments], .5),
            "p90_pieces_per_word": eda.quantile([len(x) for x in fragments], .9),
            "words_with_unknown_piece": sum(sp.unk_id() in f for f in fragments),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", nargs="*", default=CORPUS_POOLS)
    parser.add_argument("--candidates", nargs="*", type=Path, default=[])
    parser.add_argument("--audio-sample", type=int, default=12)
    parser.add_argument("--out", type=Path, default=path("data_root") / "work/vietnote/corpus.json")
    args = parser.parse_args()
    cat = Catalog.load(str(R.resolve("catalog/train-v1").catalog))
    sources, missing, flags = {}, {}, {}
    for ref in args.pool:
        try:
            pool = R.resolve(ref)
            sources[ref] = pool.manifest
            flags[ref] = pool.meta.get("flags") or {}
        except R.ReleaseError as exc:
            missing[ref] = str(exc)
    sources.update({str(p): p.resolve() for p in args.candidates})
    by_source = {ref: eda.load_manifest(str(p), cat) for ref, p in sources.items()}
    known_hashes = {r["_path"]: r["audio_sha256"] for rows in by_source.values() for r in rows if r.get("audio_sha256")}
    report = {"pools": {}, "unresolved_releases": missing, "manifest_provenance": eda.PROVENANCE}
    all_released = []
    for ref, rows in by_source.items():
        release_label = flags.get(ref, {}).get("label_source")
        if release_label:
            for row in rows:
                if not row.get("label_source"):
                    row["label_source"] = release_label
        unique, unavailable = identify_audio(rows, known_hashes)
        view = eda.summarize(rows, ref, ref, str(sources[ref]), cat)
        view.update(rows_missing_audio=unavailable, unique_audio_rows=len(unique),
                    unique_audio_hours=sum(r["_duration"] for r in unique) / 3600,
                    matched_by_existing_audio_hash=sum(r["_path"] in known_hashes for r in rows),
                    distinct_recording_ids=len({r.get("source") for r in unique if r.get("source")}),
                    mixed_language_rows=sum(r["_tokens"] > r["_en_tokens"] > 0 for r in unique),
                    upstream_dialect_rows=dict(Counter(r["provenance"]["dialect"] for r in unique
                                                       if (r.get("provenance") or {}).get("dialect"))),
                    unique_synthetic_hours=sum(r["_duration"] for r in unique if r.get("synthetic")) / 3600,
                    release_flags=flags.get(ref, {}),
                    acoustic_sample=acoustic_sample(unique, args.audio_sample, ref),
                    registered_pool=ref in args.pool)
        report["pools"][ref] = view
        if ref in args.pool:
            all_released.extend(unique)
    union, _ = identify_audio(all_released, known_hashes)
    words = Counter(w for r in union for span in r["_spans"] for w in span.split())
    durations = defaultdict(float)
    for row in union:
        label = row.get("label_source") or "unknown"
        durations[label.split(":")[0]] += row["_duration"] / 3600
    report["released_union"] = {
        "unique_audio_rows": len(union), "unique_audio_hours": sum(r["_duration"] for r in union) / 3600,
        "duplicate_rows_removed": sum(len(by_source[ref]) - report["pools"][ref]["rows_missing_audio"]
                                      for ref in args.pool if ref in by_source) - len(union),
        "identity": "Recorded SHA256 where available, resolved audio path otherwise",
        "hours_by_label_source_prefix": dict(durations),
        "unique_tagged_english_words": len(words), "rare_tagged_english_words_le_4": sum(n <= 4 for n in words.values()),
        "tokenizer_fragmentation": fragmentation(words), "sampling_exposure": None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(union)} unique available recordings across released pools; report: {args.out}")


if __name__ == "__main__":
    main()
