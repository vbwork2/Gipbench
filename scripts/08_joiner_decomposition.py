"""Decompose the blank-over-piece margin of the Phase 2 near misses over the 512 Joiner hidden units.

Question: is the blank-over-piece gap on near-miss frames carried by a few hidden units that matter little
elsewhere (a neuron-style switch), or spread over many units (a diffuse score)?

Fixed before the analysis ran:
- Events. Phase 2 NEAR_MISS pieces (greedy-prefix context) that were below rank one with blank leading, at the
  Phase 2 frame of the best final-stack rank. An event is replicated when, in this analysis's own greedy replay
  (INT8 encoder and decoder outputs, fp32 output layer), blank is the argmax at that frame and the piece is in the
  top ten. The analysis stops if fewer than 50 events replicate or fewer than half of them do.
- Ordinary frames. Frames from clean-control recordings where the replay emits blank serve to select units. Frames
  of the seeded random development sample (never used for selection) serve to measure side effects.
- Selection by two folds over recordings, so no unit set is evaluated on its own selection data. For each held-out
  fold, the K = 10 units with the largest excess contribution (mean event contribution minus mean ordinary-frame
  contribution) are chosen on the other fold.
- Pass rule, all required, otherwise the unit-level direction is closed:
  P1  pooled held-out median share of the blank-over-piece margin carried by the K units is at least 0.5;
  P2  zeroing the K units changes the per-frame argmax on fewer than 1% of random-sample frames, for both folds;
  P3  fewer than 1% of 1,000 random K-unit sets reach the held-out median share, for both folds.
Passing only authorizes a hook experiment with a random-unit control. The heldout split is never opened.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from gipformer_mech import joiner_probe as jp
from gipformer_mech.fpt import settings
from gipformer_mech.logit_lens import classify
from gipformer_mech.onnx_greedy import OnnxTransducer
from gipformer_mech.splits import development_rows, development_sample

K = 10
DRAWS = 1000
MIN_EVENTS = 50
MIN_REPLICATED_SHARE = 0.5
P1_MIN_SHARE = 0.5
P2_MAX_CHANGE = 0.01
P3_MAX_P = 0.01
CURVE = (1, 5, 10, 20, 50, 100, 200)


def load_events(path: Path, blank: int, audio_idx: dict[str, int]) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    events = []
    for row in report["findings"]:
        for error in row["errors"]:
            if classify(error["pieces"]) != "NEAR_MISS":
                continue
            for index, piece in enumerate(error["pieces"]):
                last = piece["stacks"][-1]["full"]
                if last["best_rank"] > 1 and last["top10"][0]["token_id"] == blank:
                    events.append({"utt_id": row["utt_id"], "audio_idx": audio_idx[row["utt_id"]],
                                   "word_index": error["word_index"], "piece": index,
                                   "target": piece["target_id"], "frame": last["frame"]})
    return events


def replay_set(model, rows: list[dict], encoder_dir: Path, weight, bias) -> dict[int, dict]:
    return {row["audio_idx"]: jp.replay(model, np.load(encoder_dir / f"{row['audio_idx']}.npy"), weight, bias)
            for row in rows}


def fidelity(model, rows: list[dict], encoder_dir: Path, weight, bias, count: int = 30) -> dict:
    """Agreement of the fp32 output layer with the deployed INT8 joiner along the replay path."""
    agree = frames = 0
    worst = []
    for row in rows[:count]:
        enc = np.load(encoder_dir / f"{row['audio_idx']}.npy")
        out = jp.replay(model, enc, weight, bias, keep_decoder=True)
        for frame in range(len(enc)):
            logit = model.joiner.run(None, {"encoder_out": enc[frame:frame + 1], "decoder_out": out["decoder"][frame:frame + 1]})[0][0]
            mine = weight @ out["hidden"][frame] + bias
            agree += int(logit.argmax() == mine.argmax())
            worst.append(float(np.abs(logit - mine).max()))
        frames += len(enc)
    return {"recordings": min(count, len(rows)), "frames": frames, "argmax_agreement": agree / frames,
            "median_max_logit_gap": float(np.median(worst))}


def evaluate(args) -> dict:
    config, _ = settings()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    weight, bias = jp.load_output_layer(args.model_dir, model_map)
    blank = model_map["blank_id"]
    model = OnnxTransducer(args.model_dir, threads=args.threads)
    bug = development_rows(args.manifests, "bug_mining")
    clean = development_rows(args.manifests, "clean_control")
    sample = development_sample(args.manifests, args.random_size, config["seed"])
    events = load_events(args.phase2, blank, {row["utt_id"]: row["audio_idx"] for row in bug})

    bug_states = replay_set(model, bug, args.encoder_dir, weight, bias)
    rows = []
    for event in events:
        out = bug_states[event["audio_idx"]]
        if event["frame"] >= len(out["hidden"]):
            continue
        h = out["hidden"][event["frame"]]
        logits = weight @ h + bias
        rank = int((logits > logits[event["target"]]).sum()) + 1
        rows.append({**event, "h": h, "blank_leads": int(out["tokens"][event["frame"]]) == blank, "rank": rank})
    replicated = [row for row in rows if row["blank_leads"] and row["rank"] <= 10]
    outcome = {"phase2_events": len(events), "replicated": len(replicated),
               "fidelity": fidelity(model, sample, args.encoder_dir, weight, bias)}
    if len(replicated) < MIN_EVENTS or len(replicated) / len(events) < MIN_REPLICATED_SHARE:
        outcome["stopped"] = "Too few Phase 2 events replicate under this analysis's own greedy replay."
        return outcome

    targets = np.array([row["target"] for row in replicated])
    h_events = np.stack([row["h"] for row in replicated])
    c_events = jp.contributions(h_events, weight, blank, targets)
    margin = jp.margins(c_events, bias, blank, targets)
    folds = np.array([jp.fold_of(row["utt_id"]) for row in replicated])

    clean_states = replay_set(model, clean, args.encoder_dir, weight, bias)
    clean_h = np.concatenate([out["hidden"][out["tokens"] == blank] for out in clean_states.values()])
    ordinary_mean = jp.mean_contribution(clean_h, weight, bias, blank)
    sample_states = replay_set(model, sample, args.encoder_dir, weight, bias)
    sample_h = np.concatenate([out["hidden"] for out in sample_states.values()])
    outcome.update(clean_blank_frames=len(clean_h), random_frames=len(sample_h), margin_quantiles={
        str(q): float(np.quantile(margin, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)})

    folds_out, pooled = [], []
    for fold in (0, 1):
        train, test = folds != fold, folds == fold
        units = jp.select_units(c_events[train].mean(axis=0), ordinary_mean, K)
        shares = jp.share_explained(c_events[test], margin[test], units)
        random_shares = jp.random_unit_shares(c_events[test], margin[test], K, DRAWS, seed=fold)
        after = jp.ablated_logits(h_events[test], weight, bias, units).argmax(axis=1)
        pooled.append(shares)
        folds_out.append({
            "fold": fold, "train_events": int(train.sum()), "test_events": int(test.sum()),
            "units": [int(unit) for unit in units], "median_share": float(np.median(shares)),
            "random_median_share": float(np.median(random_shares)), "random_p95": float(np.quantile(random_shares, 0.95)),
            "p_value": float((1 + (random_shares >= np.median(shares)).sum()) / (DRAWS + 1)),
            "target_top1_after_ablation": float((after == targets[test]).mean()),
            "harm": jp.argmax_change(sample_h, weight, bias, units, blank),
        })
    pooled_median = float(np.median(np.concatenate(pooled)))
    all_excess = c_events.mean(axis=0) - ordinary_mean
    all_units = jp.select_units(c_events.mean(axis=0), ordinary_mean, K)
    outcome.update(
        folds=folds_out, pooled_median_share=pooled_median,
        curve=jp.concentration_curve(all_excess, CURVE),
        participation_ratio=jp.participation_ratio(all_excess),
        in_sample_units=[{"unit": int(unit), "event_mean": float(c_events.mean(axis=0)[unit]),
                          "ordinary_mean": float(ordinary_mean[unit]), "blank_weight": float(weight[blank, unit])}
                         for unit in all_units],
        in_sample_share=float(np.median(jp.share_explained(c_events, margin, all_units))),
        checks={
            "P1": pooled_median >= P1_MIN_SHARE,
            "P2": all(item["harm"]["change_rate"] < P2_MAX_CHANGE for item in folds_out),
            "P3": all(item["p_value"] < P3_MAX_P for item in folds_out),
        })
    outcome["passed"] = all(outcome["checks"].values())
    return outcome


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(path: Path, outcome: dict) -> None:
    fidelity_ = outcome["fidelity"]
    lines = [
        "# Phase 4 Joiner unit analysis",
        "",
        "Development only. The heldout stayed sealed. The rules are in the docstring of "
        "`scripts/08_joiner_decomposition.py` and were fixed before the analysis ran.",
        "",
        "The decomposition is exact for the fp32 output layer applied to the INT8 encoder and decoder outputs. "
        f"On {fidelity_['frames']} frames of {fidelity_['recordings']} random-sample recordings this layer picks the same "
        f"argmax as the deployed INT8 Joiner on {pct(fidelity_['argmax_agreement'])} of frames. The largest logit "
        f"difference per frame has a median of {fidelity_['median_max_logit_gap']:.2f} nats.",
        "",
        f"Phase 2 events (near-miss pieces below rank one with blank leading): {outcome['phase2_events']}. "
        f"Replicated under this analysis's own greedy replay: {outcome['replicated']}.",
    ]
    if "stopped" in outcome:
        lines += ["", outcome["stopped"]]
    else:
        margin = outcome["margin_quantiles"]
        lines += [
            f"Blank-over-piece margin at replicated events, quantiles 10/25/50/75/90%: "
            + " / ".join(f"{margin[q]:.1f}" for q in ("0.1", "0.25", "0.5", "0.75", "0.9")) + " nats.",
            f"Ordinary frames: {outcome['clean_blank_frames']} blank frames from clean controls (selection); "
            f"{outcome['random_frames']} frames from the random sample (side effects).",
            "",
            "## Concentration",
            "",
            "Share of the summed positive excess contribution (event minus ordinary frames) carried by the top-k units "
            "of 512, using all events (descriptive, in-sample):",
            "",
            "| k | " + " | ".join(str(k) for k in outcome["curve"]) + " |",
            "|---|" + "---:|" * len(outcome["curve"]),
            "| share | " + " | ".join(pct(v) for v in outcome["curve"].values()) + " |",
            "",
            f"Effective number of units (participation ratio of the excess): {outcome['participation_ratio']:.0f} of 512. "
            f"A uniform spread would give 512 and a single unit would give 1.",
            "",
            f"## Held-out test with K = {K} units",
            "",
            "| Held-out fold | Events | Median share carried | Random 10-unit sets: median, 95th pct | p | "
            "Target top-1 after zeroing | Random-sample frames with changed argmax |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for item in outcome["folds"]:
            harm = item["harm"]
            lines.append(
                f"| {item['fold']} | {item['test_events']} | {pct(item['median_share'])} | "
                f"{pct(item['random_median_share'])}, {pct(item['random_p95'])} | {item['p_value']:.4f} | "
                f"{pct(item['target_top1_after_ablation'])} | {pct(harm['change_rate'])} "
                f"({harm['blank_changed']} blank, {harm['piece_changed']} piece frames) |"
            )
        lines += [
            "",
            f"Pooled held-out median share: {pct(outcome['pooled_median_share'])}.",
            "",
            "| Rule | Result |",
            "|---|---|",
            f"| P1 pooled median share at least {P1_MIN_SHARE:.0%} | {'pass' if outcome['checks']['P1'] else 'fail'} |",
            f"| P2 changed-argmax rate below {P2_MAX_CHANGE:.0%}, both folds | {'pass' if outcome['checks']['P2'] else 'fail'} |",
            f"| P3 random-set p below {P3_MAX_P}, both folds | {'pass' if outcome['checks']['P3'] else 'fail'} |",
            "",
            "Units chosen on all events, in-sample, for reference only "
            f"(median share {pct(outcome['in_sample_share'])}):",
            "",
            "| Unit | Mean contribution at events | Mean at ordinary blank frames | Blank output weight |",
            "|---:|---:|---:|---:|",
            *[f"| {u['unit']} | {u['event_mean']:+.3f} | {u['ordinary_mean']:+.3f} | {u['blank_weight']:+.3f} |"
              for u in outcome["in_sample_units"]],
            "",
            "**All three rules passed.** A hook experiment with random-unit controls is authorized."
            if outcome["passed"] else
            "**At least one rule failed.** The unit-level direction is closed.",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--manifests", type=Path, default=ROOT / "data/manifests")
    parser.add_argument("--phase2", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--encoder-dir", type=Path, default=ROOT / "artifacts/conditional_blank/encoder",
                        help="Encoder outputs cached by scripts/07_eval_conditional_blank.py.")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase4_joiner_units.md")
    parser.add_argument("--summary", type=Path, default=ROOT / "artifacts/joiner_units.json")
    parser.add_argument("--random-size", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    outcome = evaluate(args)
    write_report(args.report, outcome)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(outcome, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
