#!/usr/bin/env python3
"""Chuyển manifest JSONL -> Lhotse CutSet để fine-tune bằng icefall, CÓ CHẶN RÒ RỈ bench.

  python ft/to_lhotse.py --manifest data/train_cs.jsonl --name cs_train \
      --bench data/bench.jsonl --catalog configs/catalog.tsv --out data/lhotse \
      [--speed-perturb] [--phone-sim 0.3] [--case upper]

Bỏ mọi utterance train có: speaker hoặc source trùng bench, thuật ngữ reserved_b3, hoặc text trùng hệt một câu trong bench.
Text train = text đã chuẩn hoá (số -> chữ, chính tả Anh theo catalog), không tag, đúng kiểu chữ tokenizer
(Gipformer: IN HOA — kiểm tra bằng check_gipformer_tokenizer.py).
Kết quả: <out>/<name>_cuts_raw.jsonl.gz (chưa có fbank) -> chạy tiếp ft/compute_fbank.py.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.manifest import read_jsonl  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, model_case_text, plain_text  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--bench", help="manifest bench — để chặn rò rỉ (nên luôn truyền)")
    ap.add_argument("--catalog")
    ap.add_argument("--out", default="data/lhotse")
    ap.add_argument("--case", choices=["upper", "lower"], default="upper")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--speed-perturb", action="store_true", help="thêm bản 0.9x và 1.1x")
    ap.add_argument("--phone-sim", type=float, default=0.0, help="tỉ lệ cut được hạ 8 kHz rồi nâng lại 16 kHz")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lhotse import CutSet, Recording, RecordingSet, SupervisionSegment, SupervisionSet

    cat = Catalog.load(args.catalog)
    reserved = {t for t, v in cat.terms.items() if v.reserved_b3}
    bench_spk, bench_src, bench_txt = set(), set(), set()
    if args.bench:
        for r in read_jsonl(args.bench):
            bench_spk.add(r["speaker"])
            bench_src.add(r["source"])
            bench_txt.add(" ".join(basic_tokens(plain_text(r["text_tagged"]))))
    elif not args.bench:
        print("CẢNH BÁO: không truyền --bench, không chặn được rò rỉ.")

    drop = Counter()
    recs, sups = [], []
    for mp in args.manifest:
        for r in read_jsonl(mp):
            text_src = r.get("text_tagged") or r.get("text") or ""
            toks = cat.apply(basic_tokens(plain_text(text_src)))
            d = float(r["duration"])
            if r.get("speaker") in bench_spk:
                drop["speaker trùng bench"] += 1
                continue
            if r.get("source") in bench_src:
                drop["source trùng bench"] += 1
                continue
            if reserved and any(t in reserved for t in cat.find_terms(toks)):
                drop["có thuật ngữ reserved_b3"] += 1
                continue
            if " ".join(toks) in bench_txt:
                drop["text trùng câu bench"] += 1
                continue
            if not (args.min_dur <= d <= args.max_dur):
                drop[f"thời lượng ngoài {args.min_dur}–{args.max_dur}s"] += 1
                continue
            if not toks:
                drop["text rỗng"] += 1
                continue
            rid = f"{args.name}-{r['id']}"
            try:
                rec = Recording.from_file(r["_audio_path"], recording_id=rid)
            except Exception as e:  # noqa: BLE001
                drop[f"audio lỗi ({e.__class__.__name__})"] += 1
                continue
            recs.append(rec)
            sups.append(SupervisionSegment(id=rid, recording_id=rid, start=0.0, duration=rec.duration, channel=0,
                                           text=model_case_text(text_src, args.case, cat), speaker=r.get("speaker"),
                                           language="vi"))
    if not recs:
        sys.exit("Không còn utterance nào sau khi lọc.")
    cuts = CutSet.from_manifests(recordings=RecordingSet.from_recordings(recs),
                                 supervisions=SupervisionSet.from_segments(sups))
    if any(r.sampling_rate != 16000 for r in recs):
        cuts = cuts.resample(16000)
    if args.phone_sim > 0:
        rnd = random.Random(args.seed)
        cuts = CutSet.from_cuts(c.resample(8000).resample(16000) if rnd.random() < args.phone_sim else c for c in cuts)
    if args.speed_perturb:
        cuts = cuts + cuts.perturb_speed(0.9) + cuts.perturb_speed(1.1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.name}_cuts_raw.jsonl.gz"
    cuts.to_file(path)
    hrs = sum(c.duration for c in cuts) / 3600
    print(f"Giữ {len(recs)} utterance gốc -> {len(cuts)} cut ({hrs:.2f} h sau augment) -> {path}")
    for k, v in drop.most_common():
        print(f"  bỏ {v}: {k}")


if __name__ == "__main__":
    main()
