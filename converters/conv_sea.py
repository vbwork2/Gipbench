#!/usr/bin/env python3
"""SEA Code-Switching (gijs/sea_speech), cặp en-vi  ->  train CS + một phần ứng viên B1 để sửa nhãn tay.

  huggingface-cli login      # dataset gated: phải đồng ý điều kiện trên trang dataset trước
  python converters/conv_sea.py --out data/raw/sea --catalog configs/catalog.tsv --bench-hours 2

Trường gốc: audio (FLAC 16 kHz), text, language_pair, switch_level, n_switches, spans, duration, youtube_id,
start_sec, end_sec... Transcript do máy tạo (MOSS-Transcribe-Diarize), CHƯA được người kiểm tra.
Chia theo VIDEO: các video được chọn ngẫu nhiên cho tới khi đủ --bench-hours đi vào ứng viên B1
(mọi câu phải duyệt tay); phần còn lại là train. Một video không bao giờ nằm ở cả hai phía.
Ghi: b1_candidates_sea.jsonl (+_review.csv), train_sea.jsonl.
"""
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converters.common import Writer, auto_tag, base_parser, load_hf, safe, save_audio  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def english_spans(spans):
    """Lấy text các span tiếng Anh nếu cấu trúc `spans` nhận ra được (list dict có lang/language + text)."""
    out = []
    if isinstance(spans, dict):  # dạng cột: {"lang": [...], "text": [...]}
        keys = list(spans)
        lk = next((k for k in keys if k in ("lang", "language")), None)
        if lk and "text" in spans:
            spans = [{"lang": a, "text": b} for a, b in zip(spans[lk], spans["text"])]
    if isinstance(spans, (list, tuple)):
        for s in spans:
            if isinstance(s, dict) and str(s.get("lang") or s.get("language") or "").lower().startswith("en") and s.get("text"):
                out.append(str(s["text"]))
    return out


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--pair", default="en-vi")
    ap.add_argument("--bench-hours", type=float, default=2.0)
    ap.add_argument("--min-dur", type=float, default=2.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--no-streaming", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    ds = load_hf("gijs/sea_speech", "train", args.parquet, streaming=not args.no_streaming and not args.parquet)
    by_video = defaultdict(list)
    n = 0
    for ex in ds:
        if ex.get("language_pair") != args.pair:
            continue
        n += 1
        if args.limit and n > args.limit:
            break
        vid = safe(ex.get("youtube_id") or "unk")
        uid = f"sea_{vid}_{int(float(ex.get('start_sec') or 0) * 100):07d}"
        path, dur, _ = save_audio(ex["audio"], Path(args.out) / "audio" / vid[:2], uid)
        if not (args.min_dur <= dur <= args.max_dur):
            continue
        tag = auto_tag(ex["text"], cat, english_spans(ex.get("spans")))
        if not tag["spans"]:
            continue
        tag["flags"].append("machine_transcript")
        by_video[vid].append(({"id": uid, "audio": path, "duration": dur, "speaker": f"yt_{vid}",
                               "source": f"youtube:{vid}", "region": "unk", "gender": "u", "channel": "wideband",
                               "domain": "general", "switch_level": ex.get("switch_level"), "dataset": "sea_speech:en-vi"}, tag))
    vids = sorted(by_video)
    random.Random(args.seed).shuffle(vids)
    w = Writer(args.out, "sea")
    acc = 0.0
    for vid in vids:
        to_bench = acc < args.bench_hours * 3600
        for row, tag in by_video[vid]:
            if to_bench:
                acc += row["duration"]
                w.add("b1_candidates_sea", dict(row, split="B1"), tag)
            else:
                row = dict(row, split="train")
                w.add("train_sea", row, tag)
    w.close(review=["b1_candidates_sea"])


if __name__ == "__main__":
    main()
