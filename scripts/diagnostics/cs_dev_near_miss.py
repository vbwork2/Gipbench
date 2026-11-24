"""Cross-dataset check of the FPT Phase 2 diagnosis on the interim CS dev set (`data/dev/dev_interim_cs.jsonl`).

The FPT findings (near miss about 50%, cascades about 30%, encoder bug about 2%, gap-like about 15%) come
from one read-speech corpus. This set is different: natural speech from YouTube talks over eight domains
(VietSuperSpeech), 323 rows from 61 sources, labels listened to by a person (reviewed from a Zipformer-30M
pseudo-label). It is a dev set, `train_only`, never part of a locked bench. The locked benches (G0, C0, C1,
E0) are not used here: they are the final tests of direction B.

Stages, run in order:
* `decode`: baseline greedy INT8 with sherpa-onnx (the FPT harness: `configs/pipeline.yaml` decode
  settings, batch `decode.batch_size`, manifest order), then `metrics.score`;
* `diagnose`: `logit_lens.investigate` on every row with a CS substitution or deletion, resumable;
* `report`: `reports/phase2_dev_interim_near_miss.md`.

Registered before the diagnose stage, with the FPT rules unchanged:
* span = one CS substitution or deletion from `metrics.score` on the reference `text`, CS words found by
  `cs_terms` and `en_like` as for FPT, so the categories compare. The `<en>` tags are not used to choose
  spans; how many tagged words that rule misses, and how many untagged words it adds, is reported;
* category = `logit_lens.classify` with the emitted greedy prefix; cascade types and the cascade
  prediction as in `fpt_inconclusive_cascade.py`;
* predictions, each read on its own:
  H1 near miss is at least 30% of spans (the Phase 2 score-calibration threshold);
  H2 encoder bug is below 25% of spans (the encoder-sweep threshold);
  H3 the cascade prediction holds (CHAIN above half of the reachable INCONCLUSIVE spans and above its
     share among the others).
* reading: H1 failing means the bug-dominant reading is specific to FPT. The gap-like share (true gap,
  START, LOCAL) is reported without a threshold; none was set for it on FPT.
Nothing here selects or tunes anything, and this set stays out of the training and selection of any
trigger (TODO_rea 2.2) so that it can check one later.

`--dataset vimedcss_train` (registered before its decode stage) runs the same check, rules and
predictions unchanged, on the ViMedCSS train pool (`data/pools/vimedcss_train_v4/manifest.jsonl`):
3,868 rows, 7.5 h of medical Vietnamese-English speech from YouTube, original human labels and no ASR
agreement filter, so its gap-like share is not biased the way the interim dev set's is. The reference
is `text_tagged` with the `<en>` tags removed. If more than 1,200 rows have a CS error, the diagnose
stage takes a seeded sample of 1,200 of them (`random.Random(0).sample` over the rows sorted by id);
the report says so. It is a training pool of direction B, not a locked bench (E0 holds the ViMedCSS
test split, which is not read).
"""

import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gipformer_mech.fpt import read_rows, read_waveform, recognizer, settings, transcribe, write_rows
from gipformer_mech.metrics import score, summarize
from gipformer_mech.text_norm import cs_positions, tokens

TAG = re.compile(r"<en>(.*?)</en>", re.S)
SAMPLE_LIMIT = 1200
VSS_BIAS = [
    "## Selection bias (read before comparing the gap share)",
    "",
    "The rows come from the `vss_cs` pool, whose labels started as a Zipformer-30M pseudo-label. Every row kept "
    "has every English span confirmed by at least one of two ASR hypotheses (`agreement.confirmed`, 471 of 471 "
    "spans), and a person then reviewed it. English words that ASR systems do not hear were filtered out before "
    "this set existed. The low gap-like share is therefore partly built into the set and does not compare with "
    "FPT. The bug-side predictions (H1 to H3) are still informative: on words that an ASR can hear, Gipformer's "
    "CS errors are mostly near misses and cascades.",
]
VIMEDCSS_NOTE = [
    "## Labels",
    "",
    "Original human labels of the ViMedCSS dataset, no ASR agreement filter, so the gap-like share is not biased "
    "toward words an ASR can hear. Medical terms (drug, disease and anatomy names) are frequent. This is the train "
    "split, a training pool of direction B; the ViMedCSS test split (bench E0) was not read.",
]
DATASETS = {
    "dev_interim_cs": {
        "manifest": "data/dev/dev_interim_cs.jsonl", "report": "reports/phase2_dev_interim_near_miss.md",
        "title": "the interim CS dev set",
        "about": "`data/dev/dev_interim_cs.jsonl`: natural speech from YouTube talks (VietSuperSpeech), 323 rows, "
                 "61 sources, labels listened to by a person.",
        "note": VSS_BIAS,
    },
    "vimedcss_train": {
        "manifest": "data/pools/vimedcss_train_v4/manifest.jsonl", "report": "reports/phase2_vimedcss_train_near_miss.md",
        "title": "the ViMedCSS train pool",
        "about": "`data/pools/vimedcss_train_v4/manifest.jsonl`: medical Vietnamese-English speech from YouTube, "
                 "3,868 rows, 7.5 h, original human labels.",
        "note": VIMEDCSS_NOTE,
    },
}


def reference(row: dict) -> str:
    return row["text"] if row.get("text") else TAG.sub(r"\1", row["text_tagged"])


def tagged_words(text_tagged: str, case: str) -> list[str]:
    return [word for span in TAG.findall(text_tagged) for word in tokens(span, case)]


def tag_coverage(row: dict, case: str, cs_terms: list[str]) -> dict:
    words = tokens(reference(row), case)
    found = Counter(words[index] for index in cs_positions(words, cs_terms))
    tagged = Counter(tagged_words(row["text_tagged"], case))
    return {"tagged": sum(tagged.values()), "rule": sum(found.values()),
            "tagged_missed_by_rule": sum((tagged - found).values()),
            "rule_untagged": sum((found - tagged).values())}


def audio_path(manifest: Path, row: dict) -> Path:
    return (manifest.parent / row["audio"]).resolve()


def decode(args, config: dict, case: str) -> None:
    rows = read_rows(args.manifest)
    rec = recognizer(args.model_dir, args.out_dir, config, case, "greedy_search")
    batch = config["decode"]["batch_size"]
    cs_terms = config["text"]["cs_terms"]
    out = []
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        waves = [read_waveform(audio_path(args.manifest, row).read_bytes()) for row in chunk]
        for row, hyp in zip(chunk, transcribe(rec, waves), strict=True):
            out.append({"id": row["id"], "audio": row["audio"], "reference": reference(row),
                        "text_tagged": row["text_tagged"], "domain": row.get("domain"), "source": row.get("source"),
                        "baseline_hyp": hyp, "score_counts": score(reference(row), hyp, case, cs_terms),
                        "tags": tag_coverage(row, case, cs_terms)})
        print(f"Decoded {len(out)} of {len(rows)}", flush=True)
    write_rows(args.out_dir / "scored.jsonl", out)
    cs = [row for row in out if row["score_counts"]["cs_word_errors"]]
    tags = Counter()
    for row in out:
        tags.update(row["tags"])
    summary = {"rows": len(out), "rows_with_cs_errors": len(cs),
               "cs_error_spans": sum(len(row["score_counts"]["cs_word_errors"]) for row in cs),
               "greedy": summarize(row["score_counts"] for row in out), "tags": dict(tags)}
    (args.out_dir / "decode_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def diagnose(args, config: dict, case: str) -> None:
    import torch

    from gipformer_mech.logit_lens import investigate
    from gipformer_mech.rnnt_loader import load_model, resample

    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["decode"]["threads"])
    rows = sorted((row for row in read_rows(args.out_dir / "scored.jsonl") if row["score_counts"]["cs_word_errors"]),
                  key=lambda row: row["id"])
    if len(rows) > SAMPLE_LIMIT:
        rows = sorted(Random(0).sample(rows, SAMPLE_LIMIT), key=lambda row: row["id"])
    (args.out_dir / "diagnosed_ids.json").write_text(json.dumps([row["id"] for row in rows]) + "\n", encoding="utf-8")
    out = args.out_dir / "findings.jsonl"
    done = {json.loads(line)["utt_id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()} \
        if out.exists() else set()
    todo = [row for row in rows if row["id"] not in done]
    print(f"{len(rows)} rows with CS errors, {len(done)} done, {len(todo)} to run", flush=True)
    model, sp = load_model(args.model_dir, args.icefall, model_map, args.device)
    with out.open("a", encoding="utf-8") as stream:
        for count, row in enumerate(todo, 1):
            waveform, rate = read_waveform(audio_path(args.manifest, row).read_bytes())
            waveform = resample(waveform, rate, model_map["sample_rate"])
            positions = [error["ref_index"] for error in row["score_counts"]["cs_word_errors"]]
            errors = investigate(model, sp, model_map, waveform, tokens(row["reference"], case), positions, args.device)
            stream.write(json.dumps({"utt_id": row["id"], "errors": errors}, ensure_ascii=False,
                                    separators=(",", ":")) + "\n")
            stream.flush()
            if count % 10 == 0:
                print(f"Diagnosed {count} of {len(todo)}", flush=True)


def report(args) -> None:
    from fpt_extra_report import cascade_test, profile, relabel, share

    findings = relabel([json.loads(line) for line in (args.out_dir / "findings.jsonl").read_text(
        encoding="utf-8").splitlines() if line.strip()])
    decoded = json.loads((args.out_dir / "decode_summary.json").read_text(encoding="utf-8"))
    ids = set(json.loads((args.out_dir / "diagnosed_ids.json").read_text(encoding="utf-8")))
    scored = {row["id"]: row for row in read_rows(args.out_dir / "scored.jsonl")}
    expected = sum(len(scored[key]["score_counts"]["cs_word_errors"]) for key in ids)
    p = profile(findings)
    if {row["utt_id"] for row in findings} != ids or p["spans"] != expected:
        raise ValueError("Findings do not cover every diagnosed CS error.")
    spec = DATASETS[args.dataset]
    sampled = (f" The diagnosis covers a seeded sample of {len(ids)} of these rows ({p['spans']} spans)."
               if len(ids) < decoded["rows_with_cs_errors"] else "")
    n, cat = p["spans"], p["categories"]
    chain = sum(count for (label, _), count in p["cascade"].items() if label == "CHAIN")
    gap_like = cat["TRUE_GAP"] + sum(count for (label, _), count in p["cascade"].items() if label in ("START", "LOCAL"))
    h3, h3_text = cascade_test(p["cascade"])
    h1, h2 = cat["NEAR_MISS"] >= 0.30 * n, cat["ENCODER_BUG"] < 0.25 * n
    tags, greedy = decoded["tags"], decoded["greedy"]
    lines = [
        f"# Phase 2 diagnosis on {spec['title']} (cross-dataset check)",
        "",
        f"{spec['about']} Rules and predictions were registered in the docstring of "
        "`scripts/diagnostics/cs_dev_near_miss.py` before the diagnosis ran. Locked benches not used.",
        "",
        f"Baseline greedy INT8: WER {greedy['wer_global']:.2%}, CS-WER {greedy['cs_wer']:.2%}. "
        f"{decoded['rows_with_cs_errors']} rows with a CS substitution or deletion, {decoded['cs_error_spans']} spans."
        f"{sampled}",
        "",
        f"CS words by the FPT rule (`cs_terms` + `en_like`): {tags['rule']}. Words inside `<en>` tags: {tags['tagged']}. "
        f"Tagged words the rule misses: {tags['tagged_missed_by_rule']}; rule words without a tag: {tags['rule_untagged']}.",
        "",
        "| Category | Spans | FPT development pool | FPT extra diagnostic |",
        "|---|---:|---:|---:|",
        f"| Near miss | {share(cat['NEAR_MISS'], n)} | 50.1% | 51.3% |",
        f"| Encoder bug | {share(cat['ENCODER_BUG'], n)} | 1.4% | 2.8% |",
        f"| Predictor override | {share(cat['PREDICTOR_OVERRIDE_BUG'], n)} | 1.1% | 1.2% |",
        f"| True gap | {share(cat['TRUE_GAP'], n)} | 2.2% | 2.0% |",
        f"| Inconclusive | {share(cat['INCONCLUSIVE'], n)} | 45.2% | 42.8% |",
        f"| of which chain | {share(chain, n)} | 30.1% | 29.9% |",
        f"| Gap-like (true gap, START, LOCAL) | {share(gap_like, n)} | 17.3% | 14.8% |",
        "",
        f"Blank leads a non-top piece in {p['blank_led_spans']} near-miss spans; {p['blank_led_pieces']} blank-led pieces in "
        f"{p['blank_led_recordings']} rows; {p['near_miss_words']} distinct near-miss words.",
        "",
        "| Span type | Reachable | Not reachable |",
        "|---|---:|---:|",
        *(f"| {label} | {p['cascade'][(label, True)]} | {p['cascade'][(label, False)]} |" for label in ("START", "LOCAL", "CHAIN")),
        "",
        "## Registered predictions",
        "",
        f"- H1 near miss at least 30%: {'holds' if h1 else 'fails'} ({cat['NEAR_MISS'] / n:.1%}).",
        f"- H2 encoder bug below 25%: {'holds' if h2 else 'fails'} ({cat['ENCODER_BUG'] / n:.1%}).",
        f"- H3 cascade prediction: {'holds' if h3 else 'fails'} ({h3_text}).",
        "",
        *spec["note"],
        "",
        f"Descriptive beyond these three. One span is about {1 / n:.2%} of the total.",
    ]
    out = ROOT / spec["report"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("decode", "diagnose", "report"))
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="dev_interim_cs")
    parser.add_argument("--manifest", type=Path, help="Default: the dataset's manifest.")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/gip15")
    parser.add_argument("--icefall", type=Path)
    parser.add_argument("--out-dir", type=Path, help="Default: artifacts/<dataset>_near_miss.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.manifest = args.manifest or ROOT / DATASETS[args.dataset]["manifest"]
    args.out_dir = args.out_dir or ROOT / f"artifacts/{args.dataset}_near_miss"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config, case = settings()
    if args.stage == "decode":
        decode(args, config, case)
    elif args.stage == "diagnose":
        if args.icefall is None:
            parser.error("diagnose needs --icefall")
        diagnose(args, config, case)
    else:
        report(args)


if __name__ == "__main__":
    main()
