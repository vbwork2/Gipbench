"""Descriptive audit of Phase 5 (scripts/09_joiner_trigger.py): does the probe fire on event frames or on event rows?

Phase 5 set tau so the probe fires on 0.1% of the training negatives (blank frames of clean and random rows),
yet it fired on 4-6% of all decoded frames. Positives come only from event rows and negatives only from
non-event rows, so the probe may have learned row-level traits (speaker, density of English) instead of
frame-level ones. This script refits the probe exactly as script 09 does (same rows, folds, positives,
negatives, `trigger.fit_probe`) and, on the baseline trajectory of each held-out fold, reports the firing
rate at the 0.1% tau on: non-event-row blank frames (the negatives' own population), event-row blank frames
more than WINDOW frames from any listed event, event-row blank frames within WINDOW frames of an event (event
frames excluded), and the replicated event frames. No decoding with a penalty; no heldout is read.
Not registered: it is a post-hoc limitation check of a closed phase, and changes none of its outcomes.
"""

import argparse
from collections import defaultdict
import importlib.util
import json
from multiprocessing import get_context
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))

import numpy as np

from gipformer_mech import joiner_probe as jp
from gipformer_mech import trigger as tg
from gipformer_mech.fpt import read_rows, settings
from gipformer_mech.onnx_greedy import OnnxTransducer

spec = importlib.util.spec_from_file_location("joiner_trigger", ROOT / "scripts/09_joiner_trigger.py")
p5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p5)

RATE = 0.001
WINDOW = 10
_STATE = {}


def _init(model_dir: str, encoder_dir: str) -> None:
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    weight, bias = jp.load_output_layer(Path(model_dir), model_map)
    _STATE.update(model=OnnxTransducer(Path(model_dir), threads=1), weight=weight, bias=bias,
                  encoder_dir=Path(encoder_dir))


def _replay(task):
    utt_id, audio_idx = task
    model = _STATE["model"]
    out = jp.replay(model, np.load(_STATE["encoder_dir"] / f"{audio_idx}.npy"), _STATE["weight"], _STATE["bias"])
    return utt_id, out["hidden"], out["tokens"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--extra-dir", type=Path, required=True)
    parser.add_argument("--split-lock", type=Path, default=ROOT / "reports/phase1_extra_split_lock.json")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--findings", type=Path, required=True)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pool = get_context("fork").Pool(args.workers, _init, (str(args.model_dir), str(args.encoder_dir)))
    _, case = settings()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    blank = model_map["blank_id"]
    sets = p5.build_sets(args, case)
    rows, event_set = sets["rows"], set(sets["event"])
    all_ids = sets["event"] + sets["clean"] + sets["random"]
    listed = defaultdict(list)
    for event in read_rows(args.events):
        if event["group"] == "extra_diagnostic":
            listed[event["utt_id"]].append((event["frame"], event["target"], event["replicated"]))
    replays = {u: (h, t) for u, h, t in pool.imap_unordered(
        _replay, [(u, rows[u]["audio_idx"]) for u in all_ids], chunksize=4)}
    pool.close()
    weight, bias = jp.load_output_layer(args.model_dir, model_map)

    def parts(ids):
        out = {"negative": [], "event_row_far": [], "event_row_near": [], "event": []}
        for utt in ids:
            h, tokens = replays[utt]
            is_blank = tokens == blank
            if utt not in event_set:
                out["negative"].append(h[is_blank])
                continue
            frames = np.arange(len(h))
            near = np.zeros(len(h), dtype=bool)
            exact = np.zeros(len(h), dtype=bool)
            for frame, target, replicated in listed[utt]:
                near |= np.abs(frames - frame) <= WINDOW
                if frame < len(h):
                    exact[frame] = True
                    logits = h[frame] @ weight.T + bias
                    if replicated and is_blank[frame] and int((logits > logits[target]).sum()) + 1 <= 10:
                        out["event"].append(h[frame][None])
            out["event_row_far"].append(h[is_blank & ~near])
            out["event_row_near"].append(h[is_blank & near & ~exact])
        return {k: np.concatenate(v) for k, v in out.items()}

    result = {"rate": RATE, "window": WINDOW, "folds": []}
    for held in (0, 1):
        train = [u for u in all_ids if sets["fold"][u] != held]
        test = [u for u in all_ids if sets["fold"][u] == held]
        tr, te = parts(train), parts(test)
        w, c, mean, std = tg.fit_probe(tr["event"], tr["negative"])
        score = lambda x: ((x - mean) / std) @ w + c
        tau = tg.threshold_for_rate(score(tr["negative"]), RATE)
        fold = {"held_out": held, "tau": tau}
        for side, data in (("train", tr), ("test", te)):
            fold[side] = {k: {"frames": int(len(v)), "fire_rate": float((score(v) > tau).mean()) if len(v) else None}
                          for k, v in data.items()}
        result["folds"].append(fold)
        print(json.dumps(fold, indent=1), flush=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
