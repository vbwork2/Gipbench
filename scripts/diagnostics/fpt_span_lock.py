"""Lock the FPT development error spans that a trained checkpoint (X2 and later) is judged against.

`TODO_rea.md` item 4.1: report how many near-miss spans a checkpoint repairs next to its WER, on a span
list and a prediction fixed BEFORE any checkpoint is decoded. This script freezes the list from the stored
Phase 2 ranks (no model, no audio, development recordings only; the heldout is not read). Run it once,
commit its outputs, then do not rerun it with different rules.

Span = one CS substitution or deletion of the baseline greedy INT8 hypothesis, identified by
(utt_id, word_index), where word_index indexes `tokens(reference, case)` as `metrics.score` does.

Groups (from `logit_lens.classify` with the emitted greedy prefix, and the cascade types of
`fpt_inconclusive_cascade.py`):
* near_miss: NEAR_MISS;
* chain: INCONCLUSIVE with an earlier piece of the word at rank 2 to 10 (type CHAIN);
* other_post_encoder: PREDICTOR_OVERRIDE_BUG and ENCODER_BUG;
* gap_like: TRUE_GAP and INCONCLUSIVE of type START or LOCAL.
The group "mechanism" is near_miss + chain. The group "contrast" is gap_like.

Registered definitions for the later comparison:
* a span is FIXED when the checkpoint's hypothesis for that utterance, scored by `metrics.score` with the
  same case and CS terms, has no CS substitution or deletion at that word_index;
* the fix rate of a group is fixed spans over spans;
* intervals are 95% from a PAIRED bootstrap over recordings (utt_id), 10,000 draws, seed 0: each draw
  resamples the recordings once and every group, the Vietnamese control and the replicate control are
  computed on that same draw, so a difference between two groups is taken per draw. A recording can hold
  spans of several groups, so unpaired intervals would be wrong;
* the checkpoint is decoded with the same decoder as the baseline (greedy INT8, batch 16, original pool
  order). The baseline model is ALWAYS decoded again in the same harness as a replicate control, whether
  or not the decoder matches. The replicate control gives the "fixed" rate that comes from the harness
  alone; greedy hypotheses depend on batch composition (known from Phase 3c);
* Vietnamese control: every non-CS reference word that the baseline greedy hypothesis of the same 279
  recordings substitutes or deletes, derived at scoring time from the stored baseline hypotheses of the
  locked development pool (its sha256 is in the output file), with the same FIXED rule. It is not part of
  this lock file, it is derived by that fixed rule, and it measures how much a checkpoint repairs errors
  in general;
* BROKEN: a CS reference word of the same 279 recordings that the baseline gets right and the checkpoint
  substitutes or deletes. Its count is always reported next to the fix counts;
* no span is added, removed or regrouped after this lock.

Registered prediction (Joiner and decoder miscalibration, which X2 can change, versus a missing piece that
it cannot):
  P1a the fix rate of "mechanism" is above the replicate control: lower end of the interval of the
      per-draw difference (mechanism minus replicate) above zero;
  P1b the fix rate of "mechanism" is above the Vietnamese control: lower end of the interval of the
      per-draw difference (mechanism minus Vietnamese) above zero;
  P2  the fix rate of "mechanism" minus that of "gap_like" has a lower interval end above zero.
Reading:
* P1a, P1b and P2 hold: the miscalibration reading is supported and gap_like is what remains for X3 or a
  tokenizer check.
* P1a holds and P1b fails: X2 improves errors in general and the CS mechanism is not shown to be the
  thing it repairs.
* P1a holds and P2 fails: X2 repairs both alike and the groups are not distinct (see the training-data
  condition below before reading P2).
* P1a fails: X2 does not repair the near misses and the reading is not supported, even if WER improves.
* Churn: when BROKEN is at least the number of fixed "mechanism" spans, the result is reported as churn
  and not as support, whatever P1 and P2 give.
No magnitude threshold is registered: the sizes (293 against 63) support a direction, and a size claim
would need its own registration. P1b replaces a threshold as the guard against a general improvement.

Descriptive breakdown, always reported and never used for P1 or P2: the fix rate of near_miss, of chain
split by `reachable_with_reference_prefix` (true 67, false 43), and of other_post_encoder. The unreachable
chain spans sit closer to gap_like by rank (median weakest piece 272, see `phase2_gap_profile.md`), so
they may dilute "mechanism"; this split shows by how much.

Conditions for the result to count:
* the training data of the checkpoint contains none of the development or heldout recordings or
  reference texts of this pool (check before reading the result);
* the composition of that training data is recorded with the result: whether it holds CS text, and which
  gap_like reference words occur in its text. If any do, P2 is also reported on the gap_like spans whose
  word is absent from the training text, because a checkpoint that has seen those words can repair
  gap_like for a reason other than miscalibration, and a failed P2 then does not mean the groups are
  alike;
* the lock checksums in the output file still match.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from fpt_inconclusive_cascade import BLANK, kind, reachable
from gipbench.core.fileio import sha256_file as sha
from gipformer_mech.logit_lens import blank_led_near_miss, classify


def group_of(category: str, span_type: str | None) -> str:
    if category == "NEAR_MISS":
        return "near_miss"
    if category == "INCONCLUSIVE":
        return "chain" if span_type == "CHAIN" else "gap_like"
    return "gap_like" if category == "TRUE_GAP" else "other_post_encoder"


def locked_rows(findings: list[dict], source: str) -> list[dict]:
    rows = []
    for item in findings:
        for error in item["errors"]:
            category = classify(error["pieces"])
            span_type = kind(error)[0] if category == "INCONCLUSIVE" else None
            rows.append({
                "utt_id": item["utt_id"],
                "word_index": error["word_index"],
                "reference_word": error["reference_word"],
                "source": source,
                "category": category,
                "span_type": span_type,
                "group": group_of(category, span_type),
                "blank_led": blank_led_near_miss(error, BLANK) if category == "NEAR_MISS" else None,
                "reachable_with_reference_prefix": reachable(error) if category == "INCONCLUSIVE" else None,
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bug-mining", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--pool", type=Path, default=ROOT / "artifacts/pool_near_miss/findings.jsonl")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase2_span_lock.jsonl")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"{args.output} exists. The lock is written once.")

    pool_lock = json.loads((args.manifests / "heldout_lock.json").read_text(encoding="utf-8"))
    if sha(args.manifests / "full_scored.jsonl") != pool_lock["development_pool_sha256"]:
        raise ValueError("The development pool differs from the sealed copy.")
    rows = locked_rows(json.loads(args.bug_mining.read_text(encoding="utf-8"))["findings"], "bug_mining")
    rows += locked_rows([json.loads(line) for line in args.pool.read_text(encoding="utf-8").splitlines()],
                        "other_development")
    rows.sort(key=lambda row: (row["utt_id"], row["word_index"]))
    keys = [(row["utt_id"], row["word_index"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("A span appears twice.")

    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    summary = {
        "spans": len(rows),
        "recordings": len({row["utt_id"] for row in rows}),
        "groups": dict(Counter(row["group"] for row in rows)),
        "categories": dict(Counter(row["category"] for row in rows)),
        "span_types": dict(Counter(row["span_type"] for row in rows if row["span_type"])),
        "mechanism_spans": sum(row["group"] in ("near_miss", "chain") for row in rows),
        "spans_sha256": sha(args.output),
        "development_pool_sha256": pool_lock["development_pool_sha256"],
        "input_sha256": {"bug_mining": sha(args.bug_mining), "pool": sha(args.pool)},
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
