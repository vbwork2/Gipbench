"""TODO_rea 2.3: a static edit of the Joiner output rows of near-miss target pieces, confined to directions that
ordinary frames barely use. The last registered test of direction A.

Question: Phase 4 zeroed whole hidden units and changed the argmax on 8-30% of ordinary frames; Phase 5 lowered
blank only, which cannot reach events whose target is below rank 2. Can a weight edit that raises the logit of
the target pieces, restricted to the low-energy subspace of ordinary hidden states (the `out_proj` idea of
`gipformer_mech.weight_surgeon`), repair held-out near misses without that damage?

Registered before any edit was fitted or decoded.

Data and harness: exactly those of `scripts/09_joiner_trigger.py` (`build_sets`: diagnostic part of the extra
pool only, hash checked against `phase1_extra_split_lock.json`; event, clean and seeded random rows; two folds by
normalized reference text; the same primary NEAR_MISS spans and the same `metrics`). The harness is
`joiner_probe.replay` (INT8 encoder and decoder outputs, fp32 output layer, one symbol per frame) with the edited
output weight; its unedited run is the baseline. Neither heldout, `dev_interim_cs` nor ViMedCSS is read.

Frames, from the baseline trajectory:
* event frames: listed events of event rows that are replicated in the file and in this harness (blank is the
  argmax and the target is in the top ten); deficit = blank logit - target logit;
* ordinary frames: every frame (blank and piece) of the event, clean and random rows that lies farther than
  WINDOW = 10 frames from any event listed for that row (replicated or not).

Edit, fitted on the training fold only. C = ordinary-frame second moment H^T H / n; B = eigenvectors of its
smallest round(r * 512) eigenvalues. For each target piece t of the training events, with m_t the mean event
hidden vector and d_t the mean deficit: u_t = B B^T m_t and W[t] += alpha * d_t * u_t / |u_t|^2, so the edit
raises the target by alpha * d_t on average over its training events. Bias and all other rows are unchanged.
Grid: r in {0.25, 0.5} x alpha in {1, 2}.

Static stage (no decoding), per fold and setting, on baseline-trajectory frames:
  (a) share of event frames whose argmax becomes the target; (b) argmax change rate on ordinary frames.
A setting is statically admissible when, on the training fold, (a) >= 10% and (b) < 1% (the Phase 4 limit).
No admissible setting: the fold fails without decoding. Held-out (a) and (b) and the share of held-out events
whose target piece was edited are reported for every setting.

Decode stage, per held-out fold k, admissible settings only:
* training fold: eligible when clean damage < 0.5%, random-row WER and event-row VI errors (Vietnamese word errors
  summed over event rows, which Phase 5 did not measure) no higher than the baseline, and at least one primary
  fix. Choose the most primary fixes, ties to the smaller alpha, then the smaller r. None eligible: fold fails;
* held-out fold, chosen setting: the edit; controls at the same setting: random directions in the same subspace
  with the same per-row norms (seeds 1 to 5); descriptive: the unconstrained edit (B = identity, same alpha).
Pass, on BOTH held-out folds:
  P1 primary fixes above every random seed;
  P2 clean damage < 0.5%, random-row WER and event-row VI errors no higher than the baseline;
  P3 CS words fixed minus CS words newly broken (all held-out rows) above zero.

Reading, fixed now. A pass is evidence that the near-miss deficit is a fixable output-row calibration, supports
X2, and authorizes one confirmation on the new heldout under its own registration. A failure closes static edits
of `joiner.output_linear` for direction A. It does not close X2: X2 also trains the decoder, the encoder and
decoder projections and the whole output layer, on far more data than these ~430 spans.
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

import numpy as np

from gipformer_mech import joiner_probe as jp
from gipformer_mech.fpt import read_rows, settings
from gipformer_mech.logit_lens import classify
from gipformer_mech.metrics import score
from gipformer_mech.onnx_greedy import OnnxTransducer, pieces_to_text
from gipformer_mech.sweep_eval import CLEAN_DAMAGE_LIMIT

spec = importlib.util.spec_from_file_location("joiner_trigger", ROOT / "scripts/09_joiner_trigger.py")
p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p5)

RANK_RATIOS = (0.25, 0.5)
ALPHAS = (1.0, 2.0)
RANDOM_SEEDS = (1, 2, 3, 4, 5)
WINDOW = 10
STATIC_MIN_HIT = 0.10
STATIC_MAX_CHANGE = 0.01
MIN_PRIMARY = 100

_STATE = {}


def _init(model_dir: str, encoder_dir: str) -> None:
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    weight, bias = jp.load_output_layer(Path(model_dir), model_map)
    _STATE.update(model=OnnxTransducer(Path(model_dir), threads=1), weight=weight, bias=bias,
                  encoder_dir=Path(encoder_dir), loaded=None)


def _replay(task):
    """(utt_id, audio_idx, weight path or None, keep hidden) -> (utt_id, text, hidden or None, tokens)."""
    utt_id, audio_idx, weight_path, keep = task
    if weight_path is None:
        weight = _STATE["weight"]
    else:
        if _STATE["loaded"] != weight_path:
            _STATE["edited"], _STATE["loaded"] = np.load(weight_path), weight_path
        weight = _STATE["edited"]
    model = _STATE["model"]
    out = jp.replay(model, np.load(_STATE["encoder_dir"] / f"{audio_idx}.npy"), weight, _STATE["bias"])
    ids = [int(t) for t in out["tokens"] if t != model.blank and t != model.unk]
    return utt_id, pieces_to_text(ids, model.pieces), out["hidden"] if keep else None, out["tokens"]


def low_energy_basis(ordinary: np.ndarray, rank_ratio: float) -> np.ndarray:
    data = ordinary.astype(np.float64)
    _, vectors = np.linalg.eigh(data.T @ data / len(data))
    return vectors[:, :max(1, round(rank_ratio * data.shape[1]))]


def fit_edit(events: list[dict], basis: np.ndarray | None, alpha: float) -> dict[int, np.ndarray]:
    """Row deltas {piece: delta[512]}; basis None means the unconstrained edit."""
    by_target = defaultdict(list)
    for event in events:
        by_target[event["target"]].append(event)
    deltas = {}
    for target, group in sorted(by_target.items()):
        m = np.mean([e["h"] for e in group], axis=0).astype(np.float64)
        u = m if basis is None else basis @ (basis.T @ m)
        norm2 = float(u @ u)
        if norm2 > 1e-12:
            deltas[target] = alpha * float(np.mean([e["deficit"] for e in group])) * u / norm2
    return deltas


def random_like(deltas: dict[int, np.ndarray], basis: np.ndarray, seed: int) -> dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    out = {}
    for target, delta in sorted(deltas.items()):
        v = basis @ (basis.T @ rng.standard_normal(basis.shape[0]))
        out[target] = v * np.linalg.norm(delta) / np.linalg.norm(v)
    return out


def apply(weight: np.ndarray, deltas: dict[int, np.ndarray]) -> np.ndarray:
    edited = weight.copy()
    for target, delta in deltas.items():
        edited[target] += delta.astype(edited.dtype)
    return edited


def static(edited: np.ndarray, weight: np.ndarray, bias: np.ndarray, events: list[dict], ordinary: np.ndarray,
           chunk: int = 8192) -> dict:
    hit = np.mean([int(np.argmax(edited @ e["h"] + bias)) == e["target"] for e in events]) if events else 0.0
    changed = 0
    for start in range(0, len(ordinary), chunk):
        part = ordinary[start:start + chunk]
        changed += int(((part @ weight.T + bias).argmax(1) != (part @ edited.T + bias).argmax(1)).sum())
    return {"hit": float(hit), "change": changed / max(1, len(ordinary))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase6_joiner_null_edit.md")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    pool = get_context("fork").Pool(args.workers, _init, (str(args.model_dir), str(args.encoder_dir)))
    config, case = settings()
    cs_terms = config["text"]["cs_terms"]
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    blank = model_map["blank_id"]
    weight, bias = jp.load_output_layer(args.model_dir, model_map)
    sets = p5.build_sets(args, case)
    rows, event_set = sets["rows"], set(sets["event"])
    all_ids = sets["event"] + sets["clean"] + sets["random"]

    def decode(ids, weight_path, keep=False):
        tasks = [(u, rows[u]["audio_idx"], weight_path, keep) for u in ids]
        return {u: (text, h, tokens) for u, text, h, tokens in pool.imap_unordered(_replay, tasks, chunksize=4)}

    base = decode(all_ids, None, keep=True)
    print(f"Baseline replayed on {len(base)} rows", flush=True)
    base_hyps = {u: base[u][0] for u in all_ids}
    base_scores = {u: score(rows[u]["reference"], base_hyps[u], case, cs_terms) for u in all_ids}
    base_wrong = {(u, e["ref_index"]) for u in all_ids for e in base_scores[u]["cs_word_errors"]}
    clean_ok = {u for u in sets["clean"] if base_scores[u]["substitutions"] + base_scores[u]["deletions"]
                + base_scores[u]["insertions"] == 0}

    listed = defaultdict(list)
    for event in read_rows(args.events):
        if event["group"] == "extra_diagnostic":
            listed[event["utt_id"]].append(event)
    events, ordinary = [], {}
    for utt in all_ids:
        _, h, tokens = base[utt]
        far = np.ones(len(h), dtype=bool)
        for event in listed.get(utt, []):
            far &= np.abs(np.arange(len(h)) - event["frame"]) > WINDOW
            frame, target = event["frame"], event["target"]
            if utt in event_set and event["replicated"] and frame < len(h) and tokens[frame] == blank:
                logits = weight @ h[frame] + bias
                rank = int((logits > logits[target]).sum()) + 1
                if rank <= 10:
                    events.append({"utt_id": utt, "word_index": event["word_index"], "target": target,
                                   "rank": rank, "h": h[frame].copy(),
                                   "deficit": float(logits[blank] - logits[target])})
        ordinary[utt] = h[far]
    primary = {}
    for event in events:
        key = (event["utt_id"], event["word_index"])
        if key in sets["near"] and key in base_wrong:
            entry = primary.setdefault(key, {"rank2": False, "word": sets["near"][key]})
            entry["rank2"] |= event["rank"] == 2
    print(f"Events {len(events)}, primary spans {len(primary)}, clean rows usable {len(clean_ok)}", flush=True)
    outcome = {"events": len(events), "primary_spans": len(primary), "clean_rows": len(clean_ok), "folds": []}
    if len(primary) < MIN_PRIMARY:
        outcome["stopped"] = f"Fewer than {MIN_PRIMARY} primary spans."
        args.out.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
        return

    def measure(ids, hyps):
        m = p5.metrics(ids, sets, hyps, base_wrong, primary, clean_ok, case, cs_terms)
        m["event_vi"] = sum(score(rows[u]["reference"], hyps[u], case, cs_terms)["vi_errors"]
                            for u in ids if u in event_set)
        return m

    tmp = Path(tempfile.mkdtemp(prefix="null_edit_"))
    counter = iter(range(10 ** 6))

    def run(ids, deltas):
        path = tmp / f"w{next(counter)}.npy"
        np.save(path, apply(weight, deltas))
        out = decode(ids, str(path))
        return measure(ids, {u: out[u][0] for u in ids})

    for held in (0, 1):
        train = [u for u in all_ids if sets["fold"][u] != held]
        test = [u for u in all_ids if sets["fold"][u] == held]
        ev_tr = [e for e in events if sets["fold"][e["utt_id"]] != held]
        ev_te = [e for e in events if sets["fold"][e["utt_id"]] == held]
        ord_tr = np.concatenate([ordinary[u] for u in train])
        ord_te = np.concatenate([ordinary[u] for u in test])
        bases = {r: low_energy_basis(ord_tr, r) for r in RANK_RATIOS}
        fold = {"held_out": held, "train_rows": len(train), "test_rows": len(test), "train_events": len(ev_tr),
                "test_events": len(ev_te), "ordinary_frames_train": len(ord_tr), "static": []}
        base_train = measure(train, base_hyps)
        fold["train_baseline"] = base_train
        admissible = []
        for r in RANK_RATIOS:
            for alpha in ALPHAS:
                deltas = fit_edit(ev_tr, bases[r], alpha)
                edited = apply(weight, deltas)
                s_tr = static(edited, weight, bias, ev_tr, ord_tr)
                s_te = static(edited, weight, bias, ev_te, ord_te)
                entry = {"r": r, "alpha": alpha, "tokens": len(deltas),
                         "test_events_covered": float(np.mean([e["target"] in deltas for e in ev_te])),
                         "train": s_tr, "test": s_te,
                         "admissible": bool(s_tr["hit"] >= STATIC_MIN_HIT and s_tr["change"] < STATIC_MAX_CHANGE)}
                fold["static"].append(entry)
                print(f"fold {held} static {entry}", flush=True)
                if entry["admissible"]:
                    admissible.append((r, alpha, deltas))
        if not admissible:
            fold.update(passed=False, reason="No statically admissible setting on the training fold.")
            outcome["folds"].append(fold)
            continue
        grid = []
        for r, alpha, deltas in admissible:
            m = run(train, deltas)
            m.update(r=r, alpha=alpha, eligible=bool(
                m["clean_damage"] < CLEAN_DAMAGE_LIMIT and m["random_wer"] <= base_train["random_wer"]
                and m["event_vi"] <= base_train["event_vi"] and m["primary_fixed"] > 0))
            grid.append(m)
            print(f"fold {held} train r {r} alpha {alpha}: {m}", flush=True)
        fold["grid"] = grid
        eligible = [m for m in grid if m["eligible"]]
        if not eligible:
            fold.update(passed=False, reason="No eligible setting on the training fold.")
            outcome["folds"].append(fold)
            continue
        best = sorted(eligible, key=lambda m: (-m["primary_fixed"], m["alpha"], m["r"]))[0]
        r, alpha = best["r"], best["alpha"]
        deltas = fit_edit(ev_tr, bases[r], alpha)
        variants = {"edit": deltas, "unconstrained": fit_edit(ev_tr, None, alpha)}
        variants.update({f"random_{seed}": random_like(deltas, bases[r], seed) for seed in RANDOM_SEEDS})
        test_base = p5.metrics(test, sets, base_hyps, base_wrong, primary, clean_ok, case, cs_terms)
        test_base["event_vi"] = sum(base_scores[u]["vi_errors"] for u in test if u in event_set)
        fold.update(chosen={"r": r, "alpha": alpha}, test_baseline=test_base, test={})
        for name, d in variants.items():
            fold["test"][name] = run(test, d)
            print(f"fold {held} test {name}: {fold['test'][name]}", flush=True)
        t = fold["test"]
        fold["P1"] = bool(t["edit"]["primary_fixed"] > max(t[f"random_{s}"]["primary_fixed"] for s in RANDOM_SEEDS))
        fold["P2"] = bool(t["edit"]["clean_damage"] < CLEAN_DAMAGE_LIMIT
                          and t["edit"]["random_wer"] <= test_base["random_wer"]
                          and t["edit"]["event_vi"] <= test_base["event_vi"])
        fold["P3"] = bool(t["edit"]["cs_fixed"] - t["edit"]["cs_broken"] > 0)
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
        "# Phase 6: static edit of Joiner output rows in the low-energy subspace (TODO_rea 2.3)", "",
        "Diagnostic part of the extra pool only. Rules, grid, controls, pass rules and how a result is read are in "
        "the docstring of `scripts/10_joiner_null_edit.py`, committed before any edit was fitted. Neither heldout "
        "was read.", "",
        f"Replicated events {outcome['events']}, primary spans {outcome['primary_spans']}, clean rows usable "
        f"{outcome['clean_rows']}.", "",
    ]
    for fold in outcome["folds"]:
        lines += [f"## Held-out fold {fold['held_out']}", "",
                  f"Train rows {fold['train_rows']} ({fold['train_events']} events, "
                  f"{fold['ordinary_frames_train']} ordinary frames), test rows {fold['test_rows']} "
                  f"({fold['test_events']} events).", "",
                  "Static stage (baseline-trajectory frames; admissible needs train hit >= 10% and train change < 1%):", "",
                  "| r | alpha | Pieces edited | Test events covered | Train hit | Train change | Test hit | Test change | Admissible |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                  *(f"| {s['r']:g} | {s['alpha']:g} | {s['tokens']} | {pct(s['test_events_covered'])} | "
                    f"{pct(s['train']['hit'])} | {pct(s['train']['change'])} | {pct(s['test']['hit'])} | "
                    f"{pct(s['test']['change'])} | {s['admissible']} |" for s in fold["static"]), ""]
        if "grid" in fold:
            b = fold["train_baseline"]
            lines += ["Training-fold decodes:", "",
                      "| r | alpha | Primary fixed | CS fixed | CS broken | Clean damage | Random WER | Event-row VI errors | Eligible |",
                      "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                      *(f"| {m['r']:g} | {m['alpha']:g} | {m['primary_fixed']}/{m['primary']} | {m['cs_fixed']} | "
                        f"{m['cs_broken']} | {pct(m['clean_damage'])} | {pct(m['random_wer'])} | {m['event_vi']} | "
                        f"{m['eligible']} |" for m in fold["grid"]),
                      f"| baseline | | | | | | {pct(b['random_wer'])} | {b['event_vi']} | |", ""]
        if "chosen" not in fold:
            lines += [f"Fold fails: {fold['reason']}", ""]
            continue
        b = fold["test_baseline"]
        lines += [f"Chosen on the training fold: r {fold['chosen']['r']:g}, alpha {fold['chosen']['alpha']:g}.", "",
                  "| Held-out variant | Primary fixed | Rank-2 fixed | CS fixed | CS broken | Clean damage | Random WER | Event-row VI errors |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|",
                  f"| baseline | 0/{b['primary']} | 0/{b['rank2']} | | | | {pct(b['random_wer'])} | {b['event_vi']} |",
                  *(f"| {name} | {m['primary_fixed']}/{m['primary']} | {m['rank2_fixed']}/{m['rank2']} | {m['cs_fixed']} | "
                    f"{m['cs_broken']} | {pct(m['clean_damage'])} | {pct(m['random_wer'])} | {m['event_vi']} |"
                    for name, m in fold["test"].items()),
                  "", f"P1 {fold['P1']}, P2 {fold['P2']}, P3 {fold['P3']}: fold {'passes' if fold['passed'] else 'fails'}.", ""]
    lines += ["## Result", "",
              f"Registered pass (P1, P2 and P3 on both held-out folds): {'passed' if outcome['passed'] else 'not passed'}.",
              "", "A failure closes static edits of `joiner.output_linear` for direction A; it does not close X2."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
