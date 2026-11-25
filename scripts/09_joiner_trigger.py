"""TODO_rea 2.2: a learned trigger on the Joiner hidden state that lowers the blank logit, against matched controls.

Question: can a linear probe on h = tanh(enc + dec) tell near-miss event frames from ordinary blank frames
well enough that a blank penalty applied only where it fires repairs near misses without the damage that the
global penalty (Phase 3b) and the word-continuation gate (Phase 3c) caused? h is an affine function of the
full logit vector (W is 2000 x 512 of rank 512), so the probe reads the shape of the whole output
distribution; TODO_rea 2.1 only rules out gates on the blank-versus-best rank or margin.

Registered before any probe was trained (measured beforehand, descriptively: 654 replicated events in the
diagnostic part, 264 of them with the target as the best non-blank piece, in 219 spans):

Data, diagnostic part of the extra pool only (`extra_diagnostic.jsonl`, hash checked against
`phase1_extra_split_lock.json`; the new heldout, the old heldout, `dev_interim_cs` and ViMedCSS are not read):
* event rows: rows with at least one replicated event in `artifacts/event_replay/events.jsonl`;
* clean rows: rows with CS tokens, no error at the sherpa baseline, not event rows; damage is counted on the
  ones that are also error-free in this harness at penalty 0;
* random rows: `random.Random(0).sample` of 300 of the remaining rows, sorted by utt_id;
* folds: two, by `trigger.text_fold` of the normalized reference text, so one text never sits on both sides.

Harness: `trigger.gated_replay` (INT8 encoder and decoder outputs, fp32 output layer, one symbol per frame).
Its penalty-0 run is the baseline for every comparison; sherpa hypotheses are not compared against.

Primary spans: NEAR_MISS spans (stored Phase 2 ranks of the extra pool) of event rows that hold a replicated
event in this harness and are wrong at the baseline. Fixed = no CS substitution or deletion at that word
index. Reported subsets: spans with a rank-2 event (the ceiling of a blank penalty) and spans whose word
does not occur among the training fold's primary spans.

Per held-out fold k, trained on the other fold only:
* probe: class-balanced L2 logistic regression (`trigger.fit_probe`, l2 = 1) of replicated event frames
  (positives) against blank-argmax frames of the clean and random rows (negatives);
* grid on the training fold: penalty in {4, 8} x firing rate in {0.1%, 0.5%} of the negatives, tau from the
  training negatives (`trigger.threshold_for_rate`). Eligible: clean damage below 0.5%, random-row WER no
  higher than the baseline, at least one primary fix. Choose the most primary fixes, ties to the smaller
  penalty, then the lower rate. No eligible setting: the fold fails;
* controls on the held-out fold, at the chosen penalty and rate, tau from the training negatives:
  margin (smallest blank-over-best margins) and random directions, seeds 1 to 5.
Pass, on BOTH held-out folds:
  P1 primary fixes of the probe above the margin control and above every random seed;
  P2 clean damage below 0.5% and random-row WER no higher than the baseline;
  P3 CS words fixed minus CS words newly broken (all held-out rows) above zero.
Stop rule: fewer than 100 primary spans with a rank-2 event in the diagnostic part. A pass authorizes one
confirmation on the new heldout (`phase1_extra_split_lock.json`), which needs its own registration; the
cross-domain sets are then a descriptive transfer check of that one setting, never a selection step.
"""

import argparse
from collections import defaultdict
import json
from multiprocessing import get_context
from pathlib import Path
from random import Random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

import numpy as np

from gipbench.core.fileio import sha256_file as file_sha
from gipformer_mech import joiner_probe as jp
from gipformer_mech import trigger as tg
from gipformer_mech.fpt import read_rows, settings
from gipformer_mech.logit_lens import classify
from gipformer_mech.metrics import score, summarize
from gipformer_mech.onnx_greedy import OnnxTransducer, pieces_to_text
from gipformer_mech.sweep_eval import CLEAN_DAMAGE_LIMIT
from gipformer_mech.text_norm import tokens

PENALTIES = (4.0, 8.0)
RATES = (0.001, 0.005)
RANDOM_SEEDS = (1, 2, 3, 4, 5)
RANDOM_ROWS = 300
MIN_CEILING = 100

_STATE = {}


def _init(model_dir: str, encoder_dir: str) -> None:
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    model = OnnxTransducer(Path(model_dir), threads=1)
    weight, bias = jp.load_output_layer(Path(model_dir), model_map)
    _STATE.update(model=model, weight=weight, bias=bias, encoder_dir=Path(encoder_dir))


def _encoder(audio_idx: int) -> np.ndarray:
    return np.load(_STATE["encoder_dir"] / f"{audio_idx}.npy")


def _collect(task):
    """Baseline replay: text, blank-frame features for negatives, and features at every listed event frame."""
    utt_id, audio_idx, frames = task
    model, weight, bias = _STATE["model"], _STATE["weight"], _STATE["bias"]
    out = jp.replay(model, _encoder(audio_idx), weight, bias)
    ids = [int(t) for t in out["tokens"] if t != model.blank and t != model.unk]
    logits = out["hidden"] @ weight.T + bias
    blank_frames = out["tokens"] == model.blank
    events = []
    for frame, target in frames:
        if frame >= len(out["hidden"]):
            events.append({"frame": frame, "target": target, "rank": None, "replicated": False, "h": None})
            continue
        rank = int((logits[frame] > logits[frame, target]).sum()) + 1
        events.append({"frame": frame, "target": target, "rank": rank,
                       "replicated": bool(blank_frames[frame]) and rank <= 10, "h": out["hidden"][frame]})
    return {"utt_id": utt_id, "text": pieces_to_text(ids, model.pieces), "events": events,
            "neg_h": out["hidden"][blank_frames], "neg_logits": logits[blank_frames]}


def _decode(task):
    utt_id, audio_idx, gate, penalty = task
    model = _STATE["model"]
    out = tg.gated_replay(model, _encoder(audio_idx), _STATE["weight"], _STATE["bias"], gate, penalty)
    return utt_id, pieces_to_text(out["ids"], model.pieces), out["fired"], out["frames"]


def build_sets(args, case: str) -> dict:
    lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    path = args.extra_dir / "extra_diagnostic.jsonl"
    if file_sha(path) != lock["diagnostic"]["sha256"]:
        raise ValueError("extra_diagnostic.jsonl differs from the split lock.")
    rows = {row["utt_id"]: row for row in read_rows(path)}
    events = [event for event in read_rows(args.events) if event["group"] == "extra_diagnostic" and event["replicated"]]
    event_ids = sorted({event["utt_id"] for event in events})
    counts = lambda row: row["score_counts"]
    clean = sorted(utt for utt, row in rows.items() if utt not in set(event_ids) and row["cs_tokens"]
                   and counts(row)["substitutions"] + counts(row)["deletions"] + counts(row)["insertions"] == 0)
    rest = sorted(set(rows) - set(event_ids) - set(clean))
    random_ids = sorted(Random(0).sample(rest, RANDOM_ROWS))
    near = {}
    for item in read_rows(args.findings):
        for error in item["errors"]:
            if classify(error["pieces"]) == "NEAR_MISS":
                near[(item["utt_id"], error["word_index"])] = error["reference_word"]
    fold = {utt: tg.text_fold(" ".join(tokens(rows[utt]["reference"], case))) for utt in event_ids + clean + random_ids}
    return {"rows": rows, "events": events, "event": event_ids, "clean": clean, "random": random_ids,
            "near": near, "fold": fold}


def metrics(ids: list[str], sets: dict, hyps: dict, base_wrong: set, primary: dict, clean_ok: set, case: str,
            cs_terms: list[str], train_words: set[str] | None = None) -> dict:
    rows, idset, random_set = sets["rows"], set(ids), set(sets["random"])
    scores = {utt: score(rows[utt]["reference"], hyps[utt], case, cs_terms) for utt in ids}
    wrong = {(utt, e["ref_index"]) for utt in ids for e in scores[utt]["cs_word_errors"]}
    base_here = {key for key in base_wrong if key[0] in idset}
    keys = [key for key in primary if key[0] in idset]
    fixed = set(keys) - wrong
    clean_here = [u for u in ids if u in clean_ok]
    result = {
        "primary": len(keys), "primary_fixed": len(fixed),
        "rank2": sum(primary[key]["rank2"] for key in keys),
        "rank2_fixed": sum(primary[key]["rank2"] for key in fixed),
        "cs_fixed": len(base_here - wrong), "cs_broken": len(wrong - base_here),
        "random_wer": summarize(scores[u] for u in ids if u in random_set)["wer_global"],
        "clean_damage": sum(summarize([scores[u]])["wer_global"] > 0 for u in clean_here) / max(1, len(clean_here)),
    }
    if train_words is not None:
        disjoint = [key for key in keys if primary[key]["word"] not in train_words]
        result.update(disjoint=len(disjoint), disjoint_fixed=sum(key in fixed for key in disjoint))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "artifacts/fpt_fosd")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/base_model")
    parser.add_argument("--extra-dir", type=Path, default=ROOT / "data/manifests_extra")
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--events", type=Path, default=ROOT / "artifacts/event_replay/events.jsonl")
    parser.add_argument("--findings", type=Path, default=ROOT / "artifacts/extra_near_miss/findings.jsonl")
    parser.add_argument("--encoder-dir", type=Path, default=ROOT / "artifacts/event_replay/encoder")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/joiner_trigger.json")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase5_joiner_trigger.md")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    # Fork the workers before any ONNX session exists in this process.
    pool = get_context("fork").Pool(args.workers, _init, (str(args.model_dir), str(args.encoder_dir)))
    config, case = settings()
    cs_terms = config["text"]["cs_terms"]
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    blank = model_map["blank_id"]
    sets = build_sets(args, case)
    rows = sets["rows"]
    event_set = set(sets["event"])
    all_ids = sets["event"] + sets["clean"] + sets["random"]
    from fpt_event_replay import encode_missing
    encode_missing(OnnxTransducer(args.model_dir, threads=args.workers), {u: rows[u] for u in all_ids},
                   args.encoder_dir, args.data_dir)

    frames = defaultdict(list)
    for event in sets["events"]:
        frames[event["utt_id"]].append((event["frame"], event["target"], event["word_index"]))
    collected = {item["utt_id"]: item for item in pool.imap_unordered(
        _collect, [(u, rows[u]["audio_idx"], [(f, t) for f, t, _ in frames.get(u, [])]) for u in all_ids], chunksize=4)}
    print(f"Baseline replayed on {len(collected)} rows", flush=True)
    base_hyps = {u: collected[u]["text"] for u in all_ids}
    base_scores = {u: score(rows[u]["reference"], base_hyps[u], case, cs_terms) for u in all_ids}
    base_wrong = {(u, e["ref_index"]) for u in all_ids for e in base_scores[u]["cs_word_errors"]}
    clean_ok = {u for u in sets["clean"] if summarize([base_scores[u]])["wer_global"] == 0}

    primary = {}
    for utt in sets["event"]:
        for (_, _, word_index), event in zip(frames[utt], collected[utt]["events"], strict=True):
            key = (utt, word_index)
            if event["replicated"] and key in sets["near"] and key in base_wrong:
                entry = primary.setdefault(key, {"rank2": False, "word": sets["near"][key]})
                entry["rank2"] |= event["rank"] == 2
    ceiling = sum(entry["rank2"] for entry in primary.values())
    print(f"Primary spans {len(primary)}, with a rank-2 event {ceiling}, clean rows usable {len(clean_ok)}", flush=True)
    outcome = {"primary_spans": len(primary), "rank2_spans": ceiling, "clean_rows": len(clean_ok), "folds": []}
    if ceiling < MIN_CEILING:
        outcome["stopped"] = "Fewer than 100 primary spans with a rank-2 event."
        args.out.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        print(outcome["stopped"])
        return

    def decode(ids, gate, penalty):
        hyps, fired, total = {}, 0, 0
        tasks = [(u, rows[u]["audio_idx"], gate, penalty) for u in ids]
        for utt, text, f, n in pool.imap_unordered(_decode, tasks, chunksize=4):
            hyps[utt], fired, total = text, fired + f, total + n
        return hyps, fired / max(1, total)

    for held in (0, 1):
        train = [u for u in all_ids if sets["fold"][u] != held]
        test = [u for u in all_ids if sets["fold"][u] == held]
        pos = np.stack([e["h"] for u in train if u in event_set for e in collected[u]["events"] if e["replicated"]])
        neg_h = np.concatenate([collected[u]["neg_h"] for u in train if u not in event_set])
        neg_logits = np.concatenate([collected[u]["neg_logits"] for u in train if u not in event_set])
        w, c, mean, std = tg.fit_probe(pos, neg_h)
        probe_scores = tg.Gate("probe", 0.0, w, c, mean, std).scores(neg_h, neg_logits, blank)
        train_words = {primary[key]["word"] for key in primary if sets["fold"][key[0]] != held}
        base_train = metrics(train, sets, base_hyps, base_wrong, primary, clean_ok, case, cs_terms)
        grid = []
        for penalty in PENALTIES:
            for rate in RATES:
                gate = tg.Gate("probe", tg.threshold_for_rate(probe_scores, rate), w, c, mean, std)
                hyps, fired = decode(train, gate, penalty)
                m = metrics(train, sets, hyps, base_wrong, primary, clean_ok, case, cs_terms)
                m.update(penalty=penalty, rate=rate, fired=fired,
                         eligible=bool(m["clean_damage"] < CLEAN_DAMAGE_LIMIT
                                       and m["random_wer"] <= base_train["random_wer"] and m["primary_fixed"] > 0))
                grid.append(m)
                print(f"fold {held} train penalty {penalty} rate {rate}: {m}", flush=True)
        fold = {"held_out": held, "train_rows": len(train), "test_rows": len(test), "positives": len(pos),
                "negatives": len(neg_h), "train_baseline": base_train, "grid": grid}
        eligible = [m for m in grid if m["eligible"]]
        if not eligible:
            fold.update(passed=False, reason="No eligible setting on the training fold.")
            outcome["folds"].append(fold)
            continue
        best = sorted(eligible, key=lambda m: (-m["primary_fixed"], m["penalty"], m["rate"]))[0]
        penalty, rate = best["penalty"], best["rate"]
        gates = {"probe": tg.Gate("probe", tg.threshold_for_rate(probe_scores, rate), w, c, mean, std)}
        margin_scores = tg.Gate("margin", 0.0).scores(neg_h, neg_logits, blank)
        gates["margin"] = tg.Gate("margin", tg.threshold_for_rate(margin_scores, rate))
        for seed in RANDOM_SEEDS:
            rw = tg.random_direction(len(w), seed)
            rs = tg.Gate("random", 0.0, rw, 0.0, mean, std).scores(neg_h, neg_logits, blank)
            gates[f"random_{seed}"] = tg.Gate("random", tg.threshold_for_rate(rs, rate), rw, 0.0, mean, std)
        fold["chosen"] = {"penalty": penalty, "rate": rate}
        fold["test_baseline"] = metrics(test, sets, base_hyps, base_wrong, primary, clean_ok, case, cs_terms, train_words)
        fold["test"] = {}
        for name, gate in gates.items():
            hyps, fired = decode(test, gate, penalty)
            fold["test"][name] = {**metrics(test, sets, hyps, base_wrong, primary, clean_ok, case, cs_terms, train_words),
                                  "fired": fired}
            print(f"fold {held} test {name}: {fold['test'][name]}", flush=True)
        t = fold["test"]
        randoms = [t[f"random_{seed}"]["primary_fixed"] for seed in RANDOM_SEEDS]
        fold["P1"] = bool(t["probe"]["primary_fixed"] > t["margin"]["primary_fixed"] and t["probe"]["primary_fixed"] > max(randoms))
        fold["P2"] = bool(t["probe"]["clean_damage"] < CLEAN_DAMAGE_LIMIT
                          and t["probe"]["random_wer"] <= fold["test_baseline"]["random_wer"])
        fold["P3"] = bool(t["probe"]["cs_fixed"] - t["probe"]["cs_broken"] > 0)
        fold["passed"] = fold["P1"] and fold["P2"] and fold["P3"]
        outcome["folds"].append(fold)
    pool.close()
    outcome["passed"] = len(outcome["folds"]) == 2 and all(f["passed"] for f in outcome["folds"])
    args.out.write_text(json.dumps(outcome, indent=2, default=float) + "\n", encoding="utf-8")
    write_report(args.report, outcome)
    print(f"Passed: {outcome['passed']}")


def write_report(path: Path, outcome: dict) -> None:
    pct = lambda v: f"{v:.2%}"
    lines = [
        "# Phase 5: learned Joiner trigger for near misses (TODO_rea 2.2)",
        "",
        "Diagnostic part of the extra pool only. Rules, grid, controls and pass rules are in the docstring of "
        "`scripts/09_joiner_trigger.py` and were committed before any probe was trained. Neither heldout was read.",
        "",
        f"Primary spans (near misses with a replicated blank-led event, wrong at the penalty-0 baseline): "
        f"{outcome['primary_spans']}, of which {outcome['rank2_spans']} hold a rank-2 event (the ceiling of a blank "
        f"penalty). Clean rows usable for damage: {outcome['clean_rows']}.",
        "",
    ]
    for fold in outcome["folds"]:
        lines += [f"## Held-out fold {fold['held_out']}", "",
                  f"Train rows {fold['train_rows']}, test rows {fold['test_rows']}, positives {fold['positives']}, "
                  f"negatives {fold['negatives']}.", "",
                  "Training-fold grid:", "",
                  "| Penalty | Rate | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Eligible |",
                  "|---:|---:|---:|---:|---:|---:|---:|---|",
                  *(f"| {m['penalty']:g} | {m['rate']:.1%} | {m['primary_fixed']}/{m['primary']} | {m['cs_fixed']} | "
                    f"{m['cs_broken']} | {pct(m['clean_damage'])} | {pct(m['random_wer'])} | {m['eligible']} |"
                    for m in fold["grid"]),
                  f"| baseline | | | | | | {pct(fold['train_baseline']['random_wer'])} | |", ""]
        if "chosen" not in fold:
            lines += [f"Fold fails: {fold['reason']}", ""]
            continue
        b = fold["test_baseline"]
        lines += [f"Chosen on the training fold: penalty {fold['chosen']['penalty']:g}, rate {fold['chosen']['rate']:.1%}.", "",
                  "| Held-out gate | Primary fixed | Rank-2 fixed | Word-disjoint fixed | CS fixed | CS broken | "
                  "Clean damage | Random WER | Fired frames |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
                  f"| baseline | 0/{b['primary']} | 0/{b['rank2']} | 0/{b['disjoint']} | | | | {pct(b['random_wer'])} | |",
                  *(f"| {name} | {m['primary_fixed']}/{m['primary']} | {m['rank2_fixed']}/{m['rank2']} | "
                    f"{m['disjoint_fixed']}/{m['disjoint']} | {m['cs_fixed']} | {m['cs_broken']} | {pct(m['clean_damage'])} | "
                    f"{pct(m['random_wer'])} | {pct(m['fired'])} |" for name, m in fold["test"].items()),
                  "", f"P1 {fold['P1']}, P2 {fold['P2']}, P3 {fold['P3']}: fold {'passes' if fold['passed'] else 'fails'}.", ""]
    lines += ["## Result", "",
              f"Registered pass (P1, P2 and P3 on both held-out folds): {'passed' if outcome['passed'] else 'not passed'}."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
