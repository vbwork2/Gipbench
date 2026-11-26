"""Automated provenance-tiered curation, validation, and teacher transcription for VietNote."""

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sentencepiece as spm
import soundfile as sf
from scipy.signal import resample_poly

from gipbench.core import release as R
from gipbench.core.align import align
from gipbench.core.audiofp import load_cache, save_cache
from gipbench.core.decontam import find_terms_in_texts, read_lines
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.normprofiles import get_profile
from gipbench.core.paths import REPO
from gipbench.core.tagging import auto_tag
from gipbench.core.textnorm import Catalog, model_case_text, normalize_hyp, normalize_ref, plain_text
from gipbench.dataset.pseudolabel.agreement import agree, pair_wer, vi_wer
from gipbench.dataset.release.acoustic_audit import DEFAULT_CACHE
from gipbench.dataset.release.clean_pool import acoustic_hits, clean, hash_refs, load_refs, verify
from gipbench.infer.adapters import build_adapter
from gipbench.infer.adapters.base import load_audio
from gipbench.infer.cache import HashIndex, HypLookup

DATAHIVE_DIR = REPO / "data/work/vietnote/pilots/datahive_vi"
DATAHIVE_TRAIN_CALLS = ("vi_000001", "vi_000003", "vi_000005", "vi_000006", "vi_000007")
DATAHIVE_HELD_OUT_DEV = ("vi_000002", "vi_000004", "vi_000008")
RESERVED_PATH = REPO / "configs/dataset/reserved_terms_v1.txt"
DEV_REFS = (
    "dev/dev_v2_vi@1",
    "dev/dev_interim_cs@1",
    "dev/dev_interim_vi@1",
    str(REPO / "data/work/vietnote/dev/vi_quick.jsonl"),
    str(REPO / "data/work/vietnote/dev/interim_cs_quick.jsonl"),
)


def _norm_wer(ref_text: str, hyp_text: str, catalog: Catalog, profile) -> float:
    ref_toks, _ = normalize_ref(profile.pre(ref_text or ""), catalog, profile.catalog_mode)
    hyp_toks = normalize_hyp(profile.pre(hyp_text or ""), catalog, profile.catalog_mode)
    r_seq = [t.tok for t in ref_toks]
    if not r_seq:
        return 0.0 if not hyp_toks else 1.0
    errs = sum(1 for op, _, _ in align(r_seq, hyp_toks) if op != "C")
    return errs / len(r_seq)


def _norm_pair_wer(a: str, b: str, catalog: Catalog, profile) -> float:
    ta = normalize_hyp(profile.pre(a or ""), catalog, profile.catalog_mode)
    tb = normalize_hyp(profile.pre(b or ""), catalog, profile.catalog_mode)
    return pair_wer(ta, tb)


def _bpe_unk_count(sp: spm.SentencePieceProcessor, text: str, catalog: Catalog) -> int:
    cased = model_case_text(text or "", case="lower", catalog=catalog)
    if not cased:
        return 0
    ids = sp.encode(cased, out_type=int)
    unk = sp.unk_id()
    return sum(1 for i in ids if i == unk)


def _audio_stats(path: Path) -> dict:
    samples, rate = load_audio(str(path), target_sr=16000)
    if len(samples) == 0:
        return {"duration": 0.0, "rms": 0.0, "rms_dbfs": -120.0, "clipping_ratio": 0.0}
    rms = float(np.sqrt(np.mean(samples * samples)))
    dbfs = 20.0 * math.log10(max(rms, 1e-12))
    clip = float(np.mean(np.abs(samples) >= 0.999))
    return {
        "duration": round(len(samples) / float(rate), 4),
        "rms": round(rms, 6),
        "rms_dbfs": round(dbfs, 2),
        "clipping_ratio": round(clip, 6),
    }


def _transcribe_safe(adapter, audio_path: Path, max_chunk_sec: float = 28.0) -> str:
    samples, _ = load_audio(str(audio_path), target_sr=16000)
    if len(samples) == 0:
        return ""
    if len(samples) <= int(max_chunk_sec * 16000):
        s = adapter.rec.create_stream()
        s.accept_waveform(16000, samples)
        adapter.rec.decode_stream(s)
        return s.result.text.strip()
    chunk_len = int(25.0 * 16000)
    parts = []
    for start in range(0, len(samples), chunk_len):
        chunk = samples[start:start + chunk_len]
        if len(chunk) < 1600:
            continue
        s = adapter.rec.create_stream()
        s.accept_waveform(16000, chunk)
        adapter.rec.decode_stream(s)
        txt = s.result.text.strip()
        if txt:
            parts.append(txt)
    return " ".join(parts)


def _build_local_teachers(work_dir: Path):
    vn_cfg = {
        "engine": "sherpa_transducer",
        "system": "vietnote78_offline_fp32_curation",
        "model": {"id": "VietNote-Zipformer-78M", "dir": str((REPO / "models/vietnote78/offline/fp32").resolve())},
        "inference_mode": "open_vocab",
        "decode": {"streaming": False, "quantize": "fp32", "threads": 8,
                   "decoding_method": "greedy_search", "beam": 4, "case": "lower", "blank_penalty": 0.0},
        "batch": 1,
    }
    gip_cfg = {
        "engine": "sherpa_transducer",
        "system": "gip15_int8_curation",
        "model": {"id": "gipformer1.5-68M-rnnt", "dir": str((REPO / "models/gip15").resolve())},
        "inference_mode": "open_vocab",
        "decode": {"streaming": False, "quantize": "int8", "threads": 8,
                   "decoding_method": "greedy_search", "beam": 4, "case": "upper", "blank_penalty": 0.0},
        "batch": 1,
    }
    vn = build_adapter(vn_cfg, work_dir)
    gip = build_adapter(gip_cfg, work_dir)
    vn.load()
    gip.load()
    return vn, gip


def _stratified_sample(rows: list, n_total: int, seed: int = 0) -> list:
    if len(rows) <= n_total:
        return list(rows)
    b1 = [r for r in rows if float(r["duration"]) < 10.0]
    b2 = [r for r in rows if 10.0 <= float(r["duration"]) <= 20.0]
    b3 = [r for r in rows if float(r["duration"]) > 20.0]
    buckets = [b for b in (b1, b2, b3) if b]
    per = max(1, n_total // len(buckets))
    picked = []
    used = set()
    for b in buckets:
        b_sorted = sorted(b, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        for r in b_sorted[:per]:
            picked.append(r)
            used.add(r["id"])
    if len(picked) < n_total:
        rest = [r for r in sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
                if r["id"] not in used]
        picked.extend(rest[: n_total - len(picked)])
    return picked[:n_total]


def _write_review_csv(path: Path, fieldnames: list, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def audit_cv_other(out_dir: Path, vn, gip, sp, catalog: Catalog, profile, refs, reserved_terms) -> dict:
    rel = R.resolve("pool/cv_other@v3")
    rows = read_jsonl(rel.manifest)
    t0 = time.perf_counter()

    cur_p, drp_p, rev_p = (out_dir / "cv_other_curated.jsonl", out_dir / "cv_other_dropped.jsonl",
                           out_dir / "cv_other_flagged_review.csv")
    if cur_p.is_file() and drp_p.is_file() and rev_p.is_file():
        cur_prev = read_jsonl(cur_p)
        drp_prev = read_jsonl(drp_p)
        if len(cur_prev) + len(drp_prev) == len(rows):
            all_prev = cur_prev + drp_prev
            with rev_p.open("r", encoding="utf-8-sig") as f:
                n_rev = sum(1 for _ in csv.DictReader(f))
            return {
                "release": "pool/cv_other@v3",
                "total_rows": len(rows),
                "total_hours": round(sum(float(r["duration"]) for r in rows) / 3600.0, 4),
                "curated_rows": len(cur_prev),
                "curated_hours": round(sum(float(r["duration"]) for r in cur_prev) / 3600.0, 4),
                "dropped_rows": len(drp_prev),
                "dropped_hours": round(sum(float(r["duration"]) for r in drp_prev) / 3600.0, 4),
                "bpe_unk_rows": sum(1 for r in rows if _bpe_unk_count(sp, r.get("text_tagged") or r.get("text") or "", catalog) > 0),
                "reserved_term_hits": find_terms_in_texts(reserved_terms, (r.get("text_tagged") or r.get("text") or "" for r in rows)),
                "leak_drops": sum(1 for r in drp_prev if "leak:" in r.get("drop_reason", "")),
                "acoustic_fail": sum(1 for r in drp_prev if "acoustic_fail" in r.get("drop_reason", "")),
                "severe_both_disagree_gt35": sum(1 for r in drp_prev if "both_teachers_disagree_gt35" in r.get("drop_reason", "")),
                "teachers_agree_against_prompt": sum(1 for r in drp_prev if "teachers_agree_against_prompt" in r.get("drop_reason", "")),
                "median_vietnote_wer": round(float(np.median([r["curation_audit"]["vietnote_wer"] for r in all_prev])), 4),
                "mean_vietnote_wer": round(float(np.mean([r["curation_audit"]["vietnote_wer"] for r in all_prev])), 4),
                "median_gip15_wer": round(float(np.median([r["curation_audit"]["gip15_wer"] for r in all_prev])), 4),
                "median_teacher_pair_wer": round(float(np.median([r["curation_audit"]["teacher_pair_wer"] for r in all_prev])), 4),
                "human_review_queued": n_rev,
                "human_reviewed_rows": 0,
                "reused_cached_curation": True,
                "elapsed_sec": round(time.perf_counter() - t0, 2),
            }

    # Check reserved terms & leakage against refs first
    res_hits = find_terms_in_texts(reserved_terms, (r.get("text_tagged") or r.get("text") or "" for r in rows))
    keep_leak, drop_leak, leak_stats = clean(rows, refs, spread=True, near_threshold=0.6)
    leak_dropped_ids = {r["id"]: r.get("drop_reason", "leak") for r in drop_leak}

    curated, dropped, flagged = [], [], []
    wer_vn_list, wer_gip_list, pair_list = [], [], []
    unk_rows = 0
    bad_acoustic = 0
    agree_disagree = 0
    severe_both_disagree = 0

    for row in rows:
        audio_path = Path(row["_audio_path"])
        text = row.get("text_tagged") or row.get("text") or ""
        if _bpe_unk_count(sp, text, catalog) > 0:
            unk_rows += 1
        ast = _audio_stats(audio_path)
        hyp_vn = _transcribe_safe(vn, audio_path)
        hyp_gip = _transcribe_safe(gip, audio_path)
        w_vn = _norm_wer(text, hyp_vn, catalog, profile)
        w_gip = _norm_wer(text, hyp_gip, catalog, profile)
        p_wer = _norm_pair_wer(hyp_vn, hyp_gip, catalog, profile)
        wer_vn_list.append(w_vn)
        wer_gip_list.append(w_gip)
        pair_list.append(p_wer)

        reasons = []
        if row["id"] in leak_dropped_ids:
            reasons.append(f"leak:{leak_dropped_ids[row['id']]}")
        if ast["duration"] < 1.0 or ast["duration"] > 35.0 or ast["rms"] < 1e-4 or ast["clipping_ratio"] > 0.05:
            reasons.append("acoustic_fail")
            bad_acoustic += 1
        if min(w_vn, w_gip) > 0.35:
            reasons.append("both_teachers_disagree_gt35")
            severe_both_disagree += 1
        elif p_wer <= 0.15 and min(w_vn, w_gip) > 0.25:
            reasons.append("teachers_agree_against_prompt")
            agree_disagree += 1

        enriched = dict(row)
        enriched.setdefault("text_tagged", text)
        enriched.setdefault("license", "CC0-1.0")
        enriched.setdefault("speech_type", "read")
        enriched["curation_audit"] = {
            "audio_rms_dbfs": ast["rms_dbfs"],
            "clipping_ratio": ast["clipping_ratio"],
            "vietnote_wer": round(w_vn, 4),
            "gip15_wer": round(w_gip, 4),
            "teacher_pair_wer": round(p_wer, 4),
            "vietnote_hyp": hyp_vn,
            "gip15_hyp": hyp_gip,
        }
        if reasons:
            enriched["drop_reason"] = "+".join(reasons)
            dropped.append(enriched)
            flagged.append({
                "id": row["id"], "audio": row["audio"], "duration": row["duration"],
                "speaker": row.get("speaker", ""), "prompt_text": text,
                "vietnote_hyp": hyp_vn, "gip15_hyp": hyp_gip,
                "vietnote_wer": round(w_vn, 4), "gip15_wer": round(w_gip, 4),
                "teacher_pair_wer": round(p_wer, 4), "flag_reason": "+".join(reasons),
                "sample_kind": "flagged", "decision": "", "reviewer": "",
            })
        else:
            curated.append(enriched)

    # Add 25 stratified spot-audit rows from curated
    for row in _stratified_sample(curated, 25, seed=0):
        ca = row["curation_audit"]
        flagged.append({
            "id": row["id"], "audio": row["audio"], "duration": row["duration"],
            "speaker": row.get("speaker", ""), "prompt_text": row.get("text_tagged", ""),
            "vietnote_hyp": ca["vietnote_hyp"], "gip15_hyp": ca["gip15_hyp"],
            "vietnote_wer": ca["vietnote_wer"], "gip15_wer": ca["gip15_wer"],
            "teacher_pair_wer": ca["teacher_pair_wer"], "flag_reason": "spot_audit",
            "sample_kind": "spot_audit", "decision": "", "reviewer": "",
        })

    write_jsonl(out_dir / "cv_other_curated.jsonl", curated)
    write_jsonl(out_dir / "cv_other_dropped.jsonl", dropped)
    _write_review_csv(
        out_dir / "cv_other_flagged_review.csv",
        ["id", "audio", "duration", "speaker", "prompt_text", "vietnote_hyp", "gip15_hyp",
         "vietnote_wer", "gip15_wer", "teacher_pair_wer", "flag_reason", "sample_kind", "decision", "reviewer"],
        flagged,
    )
    elapsed = round(time.perf_counter() - t0, 2)
    return {
        "release": "pool/cv_other@v3",
        "total_rows": len(rows),
        "total_hours": round(sum(float(r["duration"]) for r in rows) / 3600.0, 4),
        "curated_rows": len(curated),
        "curated_hours": round(sum(float(r["duration"]) for r in curated) / 3600.0, 4),
        "dropped_rows": len(dropped),
        "dropped_hours": round(sum(float(r["duration"]) for r in dropped) / 3600.0, 4),
        "bpe_unk_rows": unk_rows,
        "reserved_term_hits": res_hits,
        "leak_drops": len(drop_leak),
        "acoustic_fail": bad_acoustic,
        "severe_both_disagree_gt35": severe_both_disagree,
        "teachers_agree_against_prompt": agree_disagree,
        "median_vietnote_wer": round(float(np.median(wer_vn_list)), 4),
        "mean_vietnote_wer": round(float(np.mean(wer_vn_list)), 4),
        "median_gip15_wer": round(float(np.median(wer_gip_list)), 4),
        "median_teacher_pair_wer": round(float(np.median(pair_list)), 4),
        "human_review_queued": len(flagged),
        "human_reviewed_rows": 0,
        "elapsed_sec": elapsed,
    }


def audit_spot_pool(ref: str, sample_n: int, vn, gip, sp, catalog: Catalog, profile, refs, reserved_terms,
                    out_csv: Path | None = None, is_english_only: bool = False) -> dict:
    rel = R.resolve(ref)
    rows = read_jsonl(rel.manifest)
    t0 = time.perf_counter()
    unk_rows = sum(1 for r in rows if _bpe_unk_count(sp, r.get("text_tagged") or r.get("text") or "", catalog) > 0)
    res_hits = find_terms_in_texts(reserved_terms, (r.get("text_tagged") or r.get("text") or "" for r in rows))
    _, drop_leak, _ = clean(rows, refs, spread=False, near_threshold=0.0)
    sample = _stratified_sample(rows, sample_n, seed=0)

    wer_vn, wer_gip, pair_list, bad_ac = [], [], [], 0
    flagged = []
    for row in sample:
        audio_path = Path(row["_audio_path"])
        text = row.get("text_tagged") or row.get("text") or ""
        ast = _audio_stats(audio_path)
        if ast["rms"] < 1e-4 or ast["clipping_ratio"] > 0.05:
            bad_ac += 1
        hyp_vn = _transcribe_safe(vn, audio_path)
        w_vn = _norm_wer(text, hyp_vn, catalog, profile)
        wer_vn.append(w_vn)
        hyp_gip, w_gip, p_wer = "", None, None
        if not is_english_only and gip is not None:
            hyp_gip = _transcribe_safe(gip, audio_path)
            w_gip = _norm_wer(text, hyp_gip, catalog, profile)
            p_wer = _norm_pair_wer(hyp_vn, hyp_gip, catalog, profile)
            wer_gip.append(w_gip)
            pair_list.append(p_wer)
        is_outlier = (w_vn > 0.35) if is_english_only else (min(w_vn, w_gip) > 0.35)
        if is_outlier or (out_csv is not None and len(flagged) < 20):
            flagged.append({
                "id": row["id"], "duration": row["duration"], "text_tagged": text,
                "vietnote_hyp": hyp_vn, "gip15_hyp": hyp_gip,
                "vietnote_wer": round(w_vn, 4),
                "gip15_wer": round(w_gip, 4) if w_gip is not None else "",
                "teacher_pair_wer": round(p_wer, 4) if p_wer is not None else "",
                "flag_reason": "outlier_wer_gt35" if is_outlier else "spot_audit",
                "decision": "", "reviewer": "",
            })
    if out_csv is not None:
        _write_review_csv(
            out_csv,
            ["id", "duration", "text_tagged", "vietnote_hyp", "gip15_hyp",
             "vietnote_wer", "gip15_wer", "teacher_pair_wer", "flag_reason", "decision", "reviewer"],
            flagged,
        )
    return {
        "release": ref,
        "total_rows": len(rows),
        "total_hours": round(sum(float(r["duration"]) for r in rows) / 3600.0, 4),
        "bpe_unk_rows": unk_rows,
        "reserved_term_hits": res_hits,
        "exact_leak_hits": len(drop_leak),
        "sample_rows": len(sample),
        "sample_acoustic_fail": bad_ac,
        "sample_median_vietnote_wer": round(float(np.median(wer_vn)), 4),
        "sample_mean_vietnote_wer": round(float(np.mean(wer_vn)), 4),
        "sample_median_gip15_wer": round(float(np.median(wer_gip)), 4) if wer_gip else None,
        "sample_median_pair_wer": round(float(np.median(pair_list)), 4) if pair_list else None,
        "sample_outliers_min_wer_gt35": sum(1 for i in range(len(sample))
                                            if (wer_vn[i] if is_english_only else min(wer_vn[i], wer_gip[i])) > 0.35),
        "human_reviewed_rows": 0,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def audit_vss_cached(catalog: Catalog) -> dict:
    rows = read_jsonl(R.resolve("pool/vss_cs@v5").manifest)
    hyps_dirs = [str(REPO / "results/asr_hyps/qwen3_asr_0.6b"), str(REPO / "results/asr_hyps/fw_large_v3_int8")]
    look = HypLookup(hyps_dirs, HashIndex(str(REPO / "results/asr_hyps/vss_audio_sha256.tsv")),
                     str(REPO / "data/dl/VietSuperSpeech"))
    both, qwen_only, missing = 0, 0, 0
    pair_wers, qwen_vi_wers, fw_vi_wers, agreed_all_spans = [], [], [], 0
    for r in rows:
        hq, hfw = look.get(r)
        if hq is None and hfw is None:
            missing += 1
            continue
        if hq is not None and hfw is not None:
            both += 1
            pair_wers.append(pair_wer(normalize_hyp(hq, catalog, "strict"), normalize_hyp(hfw, catalog, "strict")))
        elif hq is not None:
            qwen_only += 1
        if hq is not None:
            qwen_vi_wers.append(vi_wer(r["text_tagged"], hq, catalog))
        if hfw is not None:
            fw_vi_wers.append(vi_wer(r["text_tagged"], hfw, catalog))
        res = agree(r["text_tagged"], [hq or "", hfw or ""], catalog, min_agree=1, min_span_frac=1.0)
        if res["keep"]:
            agreed_all_spans += 1
    return {
        "release": "pool/vss_cs@v5",
        "total_rows": len(rows),
        "total_hours": round(sum(float(r["duration"]) for r in rows) / 3600.0, 4),
        "qwen_cached_rows": both + qwen_only,
        "fw_cached_rows": both,
        "both_teachers_cached": both,
        "missing_cached": missing,
        "median_qwen_fw_pair_wer": round(float(np.median(pair_wers)), 4) if pair_wers else None,
        "median_qwen_vi_wer": round(float(np.median(qwen_vi_wers)), 4) if qwen_vi_wers else None,
        "median_fw_vi_wer": round(float(np.median(fw_vi_wers)), 4) if fw_vi_wers else None,
        "span_confirmed_min1_rows": agreed_all_spans,
        "cloud_transfer_allowed": False,
        "human_reviewed_rows": 0,
    }


def _vad_segments(mono_16k: np.ndarray, sr: int = 16000) -> list[tuple[int, int]]:
    hop = int(0.010 * sr)
    win = int(0.030 * sr)
    if len(mono_16k) < int(2.0 * sr):
        return []
    n_frames = 1 + (len(mono_16k) - win) // hop
    if n_frames <= 0:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(mono_16k, win)[::hop]
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    thresh = max(0.012, 3.0 * float(np.percentile(rms, 20)))
    active = rms >= thresh

    # Group contiguous active frames, pad +/-0.15s, merge gaps < 0.50s
    pad_s = int(0.15 * sr)
    merge_gap = int(0.50 * sr)
    raw_spans = []
    i = 0
    while i < len(active):
        if not active[i]:
            i += 1
            continue
        j = i
        while j < len(active) and active[j]:
            j += 1
        s = max(0, i * hop - pad_s)
        e = min(len(mono_16k), j * hop + win + pad_s)
        if raw_spans and s - raw_spans[-1][1] <= merge_gap:
            raw_spans[-1] = (raw_spans[-1][0], max(raw_spans[-1][1], e))
        else:
            raw_spans.append((s, e))
        i = j

    # Split segments > 24.0s at the minimum-energy internal pause near the midpoint
    max_samples = int(24.0 * sr)
    split_stack = list(raw_spans)
    final_spans = []
    while split_stack:
        s, e = split_stack.pop(0)
        if e - s <= max_samples:
            if e - s >= int(2.0 * sr):
                final_spans.append((s, e))
            continue
        mid = (s + e) // 2
        lo = max(s + int(4.0 * sr), mid - int(4.0 * sr))
        hi = min(e - int(4.0 * sr), mid + int(4.0 * sr))
        if hi <= lo:
            cut = mid
        else:
            sub = mono_16k[lo:hi]
            w_sub = int(0.050 * sr)
            if len(sub) > w_sub:
                sub_rms = np.convolve(sub * sub, np.ones(w_sub) / w_sub, mode="valid")
                cut = lo + int(np.argmin(sub_rms)) + w_sub // 2
            else:
                cut = mid
        split_stack.insert(0, (cut, e))
        split_stack.insert(0, (s, cut))
    return final_spans


def _try_wsl_whisper(rows_to_check: list[dict], work_dir: Path) -> dict[str, str]:
    """Optionally run WSL faster-whisper large-v3 on <=30 borderline/EN segments with a strict 120s timeout."""
    if not rows_to_check:
        return {}
    subset = rows_to_check[:30]
    manifest = work_dir / "fw_check_manifest.jsonl"
    out_hyps = work_dir / "fw_check_hyps.jsonl"
    out_hyps.unlink(missing_ok=True)
    wsl_rows = []
    for r in subset:
        p = Path(r["_audio_path"]).resolve()
        # Convert D:\... to /mnt/d/...
        drive = p.drive.rstrip(":").lower()
        wsl_path = f"/mnt/{drive}/{p.relative_to(p.anchor).as_posix()}"
        wsl_rows.append({"id": r["id"], "_audio_path": wsl_path, "duration": r["duration"]})
    write_jsonl(manifest, wsl_rows)
    m_drv = manifest.resolve().drive.rstrip(":").lower()
    m_wsl = f"/mnt/{m_drv}/{manifest.resolve().relative_to(manifest.resolve().anchor).as_posix()}"
    o_drv = out_hyps.resolve().drive.rstrip(":").lower()
    o_wsl = f"/mnt/{o_drv}/{out_hyps.resolve().relative_to(out_hyps.resolve().anchor).as_posix()}"
    py_code = (
        "import json; from faster_whisper import WhisperModel; "
        "m = WhisperModel('large-v3', device='cpu', compute_type='int8'); "
        f"rows = [json.loads(l) for l in open('{m_wsl}', encoding='utf-8') if l.strip()]; "
        f"out = open('{o_wsl}', 'w', encoding='utf-8'); "
        "for r in rows: "
        "  segs, _ = m.transcribe(r['_audio_path'], language='vi', beam_size=1, vad_filter=False, condition_on_previous_text=False); "
        "  hyp = ' '.join(s.text.strip() for s in segs).strip(); "
        "  out.write(json.dumps({'id': r['id'], 'hyp': hyp}, ensure_ascii=False) + '\\n'); "
        "out.close()"
    )
    try:
        proc = subprocess.run(
            ["wsl", "-e", "bash", "-c", "~/.venvs/gipbench-asr/bin/python -"],
            input=py_code, capture_output=True, text=True, timeout=120,
        )
        if proc.returncode != 0 or not out_hyps.is_file():
            return {}
        return {r["id"]: r["hyp"] for r in read_jsonl(out_hyps)}
    except (subprocess.TimeoutExpired, OSError):
        return {}


def curate_datahive(out_dir: Path, vn, gip, sp, catalog: Catalog, profile, refs, reserved_terms) -> dict:
    t0 = time.perf_counter()
    meta_map = {}
    for line in (DATAHIVE_DIR / "preview/metadata.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            meta_map[Path(d["file_name"]).stem] = d

    seg_dir = out_dir / "datahive_segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    # Remove stale segments from earlier runs if any
    for old in seg_dir.glob("*.wav"):
        old.unlink()

    all_segments = []
    raw_call_sec = 0.0
    for call in DATAHIVE_TRAIN_CALLS:
        if call in DATAHIVE_HELD_OUT_DEV:
            raise AssertionError(f"Refusing to process held-out dev call: {call}")
        mp3_path = DATAHIVE_DIR / "preview" / f"{call}.mp3"
        audio, sr = sf.read(str(mp3_path), dtype="float32", always_2d=True)
        if sr != 48000 or audio.shape[1] != 2:
            raise ValueError(f"Unexpected DataHive format in {mp3_path}: sr={sr}, shape={audio.shape}")
        raw_call_sec += audio.shape[0] / float(sr)
        topic = meta_map.get(call, {}).get("topic", "conversational")
        for ch in (0, 1):
            mono_16k = resample_poly(audio[:, ch], 1, 3).astype(np.float32)
            spans = _vad_segments(mono_16k, 16000)
            for idx, (s, e) in enumerate(spans):
                seg = mono_16k[s:e]
                dur = len(seg) / 16000.0
                rms = float(np.sqrt(np.mean(seg * seg)))
                clip = float(np.mean(np.abs(seg) >= 0.999))
                if not (2.0 <= dur <= 28.0 and rms >= 0.008 and clip <= 0.02):
                    continue
                seg_id = f"datahive_{call}_ch{ch}_{idx:04d}"
                wav_path = seg_dir / f"{seg_id}.wav"
                sf.write(str(wav_path), seg, 16000, subtype="PCM_16")
                digest = sha256_file(wav_path)
                all_segments.append({
                    "id": seg_id,
                    "audio": wav_path.relative_to(out_dir).as_posix(),
                    "_audio_path": str(wav_path.resolve()),
                    "audio_sha256": digest,
                    "duration": round(dur, 4),
                    "speaker": f"datahive_{call}_ch{ch}",
                    "source": call,
                    "dataset": "datahive_vi:train",
                    "split": "train",
                    "license": "CC-BY-4.0",
                    "speech_type": "conversational",
                    "domain": topic,
                    "language_category": "vi",
                    "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
                    "clipping_ratio": round(clip, 6),
                    "start_sec": round(s / 16000.0, 3),
                    "end_sec": round(e / 16000.0, 3),
                })

    # Hard assertion: zero held-out dev calls
    assert not any(c in r["source"] for r in all_segments for c in DATAHIVE_HELD_OUT_DEV)

    # Transcribe with both local ASR families + auto_tag
    for row in all_segments:
        p = Path(row["_audio_path"])
        hyp_vn = _transcribe_safe(vn, p)
        hyp_gip = _transcribe_safe(gip, p)
        tagged = auto_tag(hyp_vn, catalog)
        p_wer = _norm_pair_wer(hyp_vn, hyp_gip, catalog, profile)
        toks = normalize_hyp(profile.pre(hyp_vn), catalog, profile.catalog_mode)
        has_en = bool(tagged["spans"])
        row.update({
            "text": hyp_vn,
            "text_tagged": tagged["text_tagged"],
            "en_spans": tagged["spans"],
            "has_en_span": has_en,
            "language_category": "mixed" if has_en else "vi",
            "vietnote_hyp": hyp_vn,
            "gip15_hyp": hyp_gip,
            "teacher_pair_wer": round(p_wer, 4),
            "n_tokens": len(toks),
            "bpe_unk": _bpe_unk_count(sp, tagged["text_tagged"], catalog),
            "label_source": "pseudo:vietnote78+gip15_agreed",
            "pseudo_labeled": True,
            "human_reviewed": False,
        })

    # Acoustic + leak check against all locked benchmarks & local devs
    cache = load_cache(str(DEFAULT_CACHE)) if DEFAULT_CACHE.is_file() else {}
    n0 = len(cache)
    ac_hits = acoustic_hits(all_segments, refs, workers=2, cache=cache)
    if len(cache) > n0:
        DEFAULT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        save_cache(str(DEFAULT_CACHE), cache)
    leak_keep, leak_drop, _ = clean(all_segments, refs, spread=False, near_threshold=0.6, extra_hits=ac_hits)
    leak_drop_reasons = {r["id"]: r.get("drop_reason", "leak") for r in leak_drop}
    res_terms_set = reserved_terms

    # Optional 3rd teacher check on <= 30 EN-span / borderline segments
    check_cands = [r for r in all_segments
                   if r["id"] not in leak_drop_reasons and (r["has_en_span"] or 0.15 < r["teacher_pair_wer"] <= 0.35)]
    fw_hyps = _try_wsl_whisper(check_cands, out_dir)
    for r in all_segments:
        if r["id"] in fw_hyps:
            r["fw_large_v3_hyp"] = fw_hyps[r["id"]]

    curated, dropped, review_queue = [], [], []
    n_res_drop = 0
    for r in all_segments:
        hit_res = find_terms_in_texts(res_terms_set, [r["text_tagged"]])
        if r["id"] in leak_drop_reasons:
            dropped.append(dict(r, drop_reason=f"leak:{leak_drop_reasons[r['id']]}"))
        elif hit_res:
            n_res_drop += 1
            dropped.append(dict(r, drop_reason=f"reserved_term:{','.join(sorted(hit_res))}"))
        elif r["n_tokens"] < 4 or r["bpe_unk"] > 0:
            dropped.append(dict(r, drop_reason="short_or_bpe_unk"))
        elif r["teacher_pair_wer"] <= 0.20:
            curated.append(r)
        elif r["teacher_pair_wer"] <= 0.40:
            dropped.append(dict(r, drop_reason="borderline_teacher_disagreement"))
        else:
            dropped.append(dict(r, drop_reason="high_teacher_disagreement"))

        if (r["id"] not in leak_drop_reasons and not hit_res and r["n_tokens"] >= 4
                and (r["has_en_span"] or 0.20 < r["teacher_pair_wer"] <= 0.40)):
            review_queue.append({
                "id": r["id"], "audio": r["audio"], "duration": r["duration"],
                "speaker": r["speaker"], "domain": r["domain"],
                "text_tagged": r["text_tagged"], "vietnote_hyp": r["vietnote_hyp"],
                "gip15_hyp": r["gip15_hyp"], "fw_large_v3_hyp": r.get("fw_large_v3_hyp", ""),
                "teacher_pair_wer": r["teacher_pair_wer"],
                "reason": "en_span" if r["has_en_span"] else "borderline_disagreement",
                "decision": "", "reviewer": "", "corrected_text_tagged": "",
            })

    # Also add 20 random accepted segments for representative audit
    rng = sorted(curated, key=lambda r: hashlib.sha256(f"dh_spot:{r['id']}".encode()).hexdigest())
    queued_ids = {x["id"] for x in review_queue}
    for r in rng[:20]:
        if r["id"] not in queued_ids:
            review_queue.append({
                "id": r["id"], "audio": r["audio"], "duration": r["duration"],
                "speaker": r["speaker"], "domain": r["domain"],
                "text_tagged": r["text_tagged"], "vietnote_hyp": r["vietnote_hyp"],
                "gip15_hyp": r["gip15_hyp"], "fw_large_v3_hyp": r.get("fw_large_v3_hyp", ""),
                "teacher_pair_wer": r["teacher_pair_wer"],
                "reason": "spot_audit", "decision": "", "reviewer": "", "corrected_text_tagged": "",
            })

    write_jsonl(out_dir / "datahive_train_curated.jsonl", curated)
    write_jsonl(out_dir / "datahive_train_dropped.jsonl", dropped)
    _write_review_csv(
        out_dir / "datahive_uncertain_review.csv",
        ["id", "audio", "duration", "speaker", "domain", "text_tagged", "vietnote_hyp",
         "gip15_hyp", "fw_large_v3_hyp", "teacher_pair_wer", "reason",
         "decision", "reviewer", "corrected_text_tagged"],
        review_queue,
    )
    en_curated = [r for r in curated if r["has_en_span"]]
    return {
        "train_calls": list(DATAHIVE_TRAIN_CALLS),
        "held_out_dev_calls": list(DATAHIVE_HELD_OUT_DEV),
        "raw_call_hours": round(raw_call_sec / 3600.0, 4),
        "raw_channel_hours": round(2 * raw_call_sec / 3600.0, 4),
        "vad_segments": len(all_segments),
        "vad_hours": round(sum(r["duration"] for r in all_segments) / 3600.0, 4),
        "curated_rows": len(curated),
        "curated_hours": round(sum(r["duration"] for r in curated) / 3600.0, 4),
        "curated_en_span_rows": len(en_curated),
        "curated_en_span_hours": round(sum(r["duration"] for r in en_curated) / 3600.0, 4),
        "dropped_rows": len(dropped),
        "dropped_hours": round(sum(r["duration"] for r in dropped) / 3600.0, 4),
        "drop_reasons": dict(Counter(r["drop_reason"] for r in dropped)),
        "reserved_term_drops": n_res_drop,
        "leak_drops": len(leak_drop),
        "fw_large_v3_spot_checked": len(fw_hyps),
        "median_teacher_pair_wer_all": round(float(np.median([r["teacher_pair_wer"] for r in all_segments])), 4) if all_segments else None,
        "median_teacher_pair_wer_curated": round(float(np.median([r["teacher_pair_wer"] for r in curated])), 4) if curated else None,
        "human_review_queued": len(review_queue),
        "human_reviewed_rows": 0,
        "curated_manifest_sha256": manifest_sha256(curated),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "data/work/vietnote/curation")
    args = parser.parse_args()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = Catalog.load(str(R.resolve("catalog/train-v1").catalog))
    profile = get_profile("gipbench_strict_v1")
    sp = spm.SentencePieceProcessor(model_file=str(REPO / "models/vietnote78/bpe.model"))
    reserved_terms = read_lines(str(RESERVED_PATH))
    refs = load_refs(R.locked_bench_refs(), DEV_REFS)
    hash_refs(refs)

    vn, gip = _build_local_teachers(out_dir)
    print("Loaded VietNote-78M FP32 and GipFormer-1.5 INT8 local teachers.", flush=True)

    cv_summary = audit_cv_other(out_dir, vn, gip, sp, catalog, profile, refs, reserved_terms)
    print("CV other audit:", json.dumps(cv_summary), flush=True)

    fleurs_summary = audit_spot_pool(
        "pool/vietnote_fleurs_vi@v2", 150, vn, gip, sp, catalog, profile, refs, reserved_terms,
        out_csv=out_dir / "fleurs_audit_review.csv", is_english_only=False,
    )
    print("FLEURS spot audit:", json.dumps(fleurs_summary), flush=True)

    libri_summary = audit_spot_pool(
        "pool/vietnote_librispeech_en@v2", 100, vn, None, sp, catalog, profile, refs, reserved_terms,
        is_english_only=True,
    )
    print("LibriSpeech spot audit:", json.dumps(libri_summary), flush=True)

    vivos_summary = audit_spot_pool(
        "pool/vivos_train@v3", 150, vn, gip, sp, catalog, profile, refs, reserved_terms,
        is_english_only=False,
    )
    print("VIVOS spot audit:", json.dumps(vivos_summary), flush=True)

    vimed_summary = audit_spot_pool(
        "pool/vietnote_vimedcss_cs@v1", 150, vn, gip, sp, catalog, profile, refs, reserved_terms,
        is_english_only=False,
    )
    print("ViMedCSS spot audit:", json.dumps(vimed_summary), flush=True)

    vss_summary = audit_vss_cached(catalog)
    print("VSS cached audit:", json.dumps(vss_summary), flush=True)

    datahive_summary = curate_datahive(out_dir, vn, gip, sp, catalog, profile, refs, reserved_terms)
    print("DataHive curation:", json.dumps(datahive_summary), flush=True)

    summary = {
        "catalog_sha256": R.sha256(R.resolve("catalog/train-v1").catalog),
        "reserved_terms_sha256": R.sha256(RESERVED_PATH),
        "cv_other": cv_summary,
        "fleurs_vi": fleurs_summary,
        "librispeech_en": libri_summary,
        "vivos_vi": vivos_summary,
        "vimedcss_cs": vimed_summary,
        "vss_cs": vss_summary,
        "datahive_vi": datahive_summary,
    }
    (out_dir / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    md = [
        "# VietNote Automated Curation Summary",
        "",
        "## Existing Pool Audits",
        "",
        "| Pool | Rows | Hours | Cloud OK | BPE UNK | Leak/Res | Sample/Full Median VietNote WER | Flagged / Human Reviewed |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
        f"| `pool/cv_other@v3` | {cv_summary['curated_rows']}/{cv_summary['total_rows']} | {cv_summary['curated_hours']:.3f}/{cv_summary['total_hours']:.3f} | Yes (CC0) | {cv_summary['bpe_unk_rows']} | {cv_summary['leak_drops']}/{len(cv_summary['reserved_term_hits'])} | {cv_summary['median_vietnote_wer']*100:.2f}% | {cv_summary['human_review_queued']} queued / 0 reviewed |",
        f"| `pool/vietnote_fleurs_vi@v2` | {fleurs_summary['total_rows']} | {fleurs_summary['total_hours']:.3f} | Yes (CC-BY-4.0) | {fleurs_summary['bpe_unk_rows']} | {fleurs_summary['exact_leak_hits']}/{len(fleurs_summary['reserved_term_hits'])} | {fleurs_summary['sample_median_vietnote_wer']*100:.2f}% (n={fleurs_summary['sample_rows']}) | {fleurs_summary['sample_outliers_min_wer_gt35']} outliers / 0 reviewed |",
        f"| `pool/vietnote_librispeech_en@v2` | {libri_summary['total_rows']} | {libri_summary['total_hours']:.3f} | Yes (CC-BY-4.0) | {libri_summary['bpe_unk_rows']} | {libri_summary['exact_leak_hits']}/{len(libri_summary['reserved_term_hits'])} | {libri_summary['sample_median_vietnote_wer']*100:.2f}% (n={libri_summary['sample_rows']}) | {libri_summary['sample_outliers_min_wer_gt35']} outliers / 0 reviewed |",
        f"| `pool/vivos_train@v3` | {vivos_summary['total_rows']} | {vivos_summary['total_hours']:.3f} | Local (CC-BY-NC-SA-4.0) | {vivos_summary['bpe_unk_rows']} | {vivos_summary['exact_leak_hits']}/{len(vivos_summary['reserved_term_hits'])} | {vivos_summary['sample_median_vietnote_wer']*100:.2f}% (n={vivos_summary['sample_rows']}) | {vivos_summary['sample_outliers_min_wer_gt35']} outliers / 0 reviewed |",
        f"| `pool/vietnote_vimedcss_cs@v1` | {vimed_summary['total_rows']} | {vimed_summary['total_hours']:.3f} | Local only | {vimed_summary['bpe_unk_rows']} | {vimed_summary['exact_leak_hits']}/{len(vimed_summary['reserved_term_hits'])} | {vimed_summary['sample_median_vietnote_wer']*100:.2f}% (n={vimed_summary['sample_rows']}) | {vimed_summary['sample_outliers_min_wer_gt35']} outliers / 0 reviewed |",
        "",
        "## DataHive VI Conversational Curation (5 train calls only; 3 dev calls held out)",
        "",
        f"- Raw train calls: {datahive_summary['raw_call_hours']:.3f}h ({datahive_summary['raw_channel_hours']:.3f}h across 2 channels)",
        f"- VAD segments kept: {datahive_summary['vad_segments']} ({datahive_summary['vad_hours']:.3f}h)",
        f"- Teacher-agreed curated (`pair_wer <= 0.20`): **{datahive_summary['curated_rows']} segments / {datahive_summary['curated_hours']:.3f}h** "
        f"(with `<en>` spans: {datahive_summary['curated_en_span_rows']} / {datahive_summary['curated_en_span_hours']:.3f}h)",
        f"- Selective human-review queue: {datahive_summary['human_review_queued']} rows (0 reviewed)",
        f"- Drop reasons: `{json.dumps(datahive_summary['drop_reasons'])}`",
    ]
    (out_dir / "audit_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
