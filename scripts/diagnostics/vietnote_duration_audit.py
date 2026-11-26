"""Audit utterance duration distributions before and after Icefall's duration policy."""

import argparse
import json
from pathlib import Path

import numpy as np
import sentencepiece as spm

from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO
from gipbench.core.textnorm import plain_text
from gipbench.model.vietnote import LOCK

TARGETS = [
    ("pool/vietnote_vimedcss_cs@v1", "train_pool", "data/releases/pool/vietnote_vimedcss_cs@v1/manifest.jsonl"),
    ("pool/vimedcss_train@v4", "train_pool", "data/releases/pool/vimedcss_train@v4/manifest.jsonl"),
    ("pool/vss_cs@v5", "train_pool", "data/releases/pool/vss_cs@v5/manifest.jsonl"),
    ("pool/vivos_train@v3", "train_pool", "data/releases/pool/vivos_train@v3/manifest.jsonl"),
    ("pool/cv_other@v3", "train_pool", "data/releases/pool/cv_other@v3/manifest.jsonl"),
    ("pool/vietnote_fleurs_vi@v2", "train_pool", "data/releases/pool/vietnote_fleurs_vi@v2/manifest.jsonl"),
    ("pool/vietnote_librispeech_en@v2", "train_pool", "data/releases/pool/vietnote_librispeech_en@v2/manifest.jsonl"),
    ("pilot/vietspeech", "pilot", "data/work/vietnote/pilots/vietspeech/candidates.jsonl"),
    ("pilot/vivoice", "pilot", "data/work/vietnote/pilots/vivoice/candidates.jsonl"),
    ("pilot/lsvsc", "pilot", "data/work/vietnote/pilots/lsvsc/candidates.jsonl"),
    ("pilot/zerospeech", "pilot", "data/work/vietnote/pilots/zerospeech/candidates.jsonl"),
    ("v02/tiny_163", "v02", "data/work/vietnote/v02/tiny/train.jsonl"),
    ("dev/dev_v2_vi@1", "dev", "data/releases/dev/dev_v2_vi@1/manifest.jsonl"),
    ("dev/dev_interim_cs@1", "dev", "data/releases/dev/dev_interim_cs@1/manifest.jsonl"),
    ("quick/vi_quick", "quick_dev", "data/work/vietnote/dev/vi_quick.jsonl"),
    ("quick/interim_cs_quick", "quick_dev", "data/work/vietnote/dev/interim_cs_quick.jsonl"),
]

BANDS = [
    ("<1s", 0.0, 1.0),
    ("1-5s", 1.0, 5.0),
    ("5-10s", 5.0, 10.0),
    ("10-15s", 10.0, 15.0),
    ("15-20s", 15.0, 20.0),
    ("20-25s", 20.0, 25.0),
    ("25-30s", 25.0, 30.0),
    (">30s", 30.0, float("inf")),
]


def summarize_manifest(rows, sp, normalize_text, policies=(20.0, 30.0, 35.0, 45.0)):
    durs = np.array([float(r["duration"]) for r in rows], dtype=float)
    bands = {}
    for label, lo, hi in BANDS:
        if label == "<1s":
            mask = durs < hi
        elif label == ">30s":
            mask = durs > lo
        else:
            mask = (durs >= lo) & (durs <= hi) if label == "15-20s" else (durs >= lo) & (durs < hi)
        bands[label] = {
            "rows": int(mask.sum()),
            "hours": round(float(durs[mask].sum() / 3600.0), 6),
        }

    # Precompute frame-vs-token filter check using upstream Icefall formula:
    # T = ((int(c.duration * 100) - 7) // 2 + 1) // 2
    t_ok = []
    for r in rows:
        d = float(r["duration"])
        frames_sub = ((int(d * 100) - 7) // 2 + 1) // 2
        txt = normalize_text(plain_text(r.get("text_tagged") or r.get("text") or ""))
        toks = sp.encode(txt, out_type=str)
        t_ok.append(frames_sub >= len(toks))
    t_ok_arr = np.array(t_ok, dtype=bool)

    policy_results = {}
    for max_d in policies:
        in_range = (durs >= 1.0) & (durs <= max_d)
        kept = in_range & t_ok_arr
        policy_results[f"1.0-{max_d:g}s"] = {
            "kept_rows": int(kept.sum()),
            "kept_hours": round(float(durs[kept].sum() / 3600.0), 6),
            "dropped_below_1s": int((durs < 1.0).sum()),
            "dropped_above_max": int((durs > max_d).sum()),
            "dropped_above_max_hours": round(float(durs[durs > max_d].sum() / 3600.0), 6),
            "dropped_frames_lt_tokens": int((in_range & ~t_ok_arr).sum()),
            "retention_row_pct": round(100.0 * float(kept.sum()) / len(rows), 2) if rows else 0.0,
            "retention_hour_pct": round(100.0 * float(durs[kept].sum()) / float(durs.sum()), 2) if durs.sum() > 0 else 0.0,
        }

    return {
        "rows": len(rows),
        "hours": round(float(durs.sum() / 3600.0), 6),
        "min": round(float(durs.min()), 4) if len(durs) else 0.0,
        "p50": round(float(np.percentile(durs, 50)), 4) if len(durs) else 0.0,
        "p90": round(float(np.percentile(durs, 90)), 4) if len(durs) else 0.0,
        "p95": round(float(np.percentile(durs, 95)), 4) if len(durs) else 0.0,
        "p99": round(float(np.percentile(durs, 99)), 4) if len(durs) else 0.0,
        "max": round(float(durs.max()), 4) if len(durs) else 0.0,
        "gt_20s_rows": int((durs > 20.0).sum()),
        "gt_20s_hours": round(float(durs[durs > 20.0].sum() / 3600.0), 6),
        "gt_25s_rows": int((durs > 25.0).sum()),
        "gt_30s_rows": int((durs > 30.0).sum()),
        "bands": bands,
        "policies": policy_results,
    }


def main():
    import sys
    sys.path.insert(0, str(REPO / "models/vietnote78/upstream/src"))
    from vnasr.text import normalize_text

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/work/vietnote/duration_audit/report.json")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.load(str(REPO / "models/vietnote78/bpe.model"))
    card = json.loads(LOCK.read_text(encoding="utf-8"))

    datasets = {}
    for name, role, rel in TARGETS:
        path = REPO / rel
        if not path.is_file():
            continue
        rows = read_jsonl(path)
        if not rows:
            continue
        datasets[name] = {"role": role, "path": rel, **summarize_manifest(rows, sp, normalize_text)}

    report = {
        "model_revision": card["revision"],
        "icefall_commit": card["icefall_commit"],
        "recipe_filter_location": "data/work/vietnote/recipe/finetune.py:1302-1340",
        "upstream_filter_bounds_seconds": [1.0, 20.0],
        "v03_mixture_configured_max_duration_seconds": 45.0,
        "speed_perturb_note": (
            "vi_datamodule.py:299 applies PerturbSpeed(factors=[0.9, 1.1], p=2/3) whenever "
            "--on-the-fly-feats=1, after remove_short_and_long_utt filters raw cut durations."
        ),
        "datasets": datasets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {"rows": v["rows"], "hours": v["hours"], "gt_20s_rows": v["gt_20s_rows"],
                          "gt_20s_hours": v["gt_20s_hours"], "max": v["max"]}
                      for k, v in datasets.items()}, indent=2))


if __name__ == "__main__":
    main()
