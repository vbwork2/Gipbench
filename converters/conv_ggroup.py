#!/usr/bin/env python3
"""G-Group vi-asr-{tech,edu,finance,pubadmin}-test  ->  ứng viên B1 (câu có từ Anh) và B5 (pubadmin không có từ Anh).

  python converters/conv_ggroup.py --names tech edu finance pubadmin --out data/raw/ggroup --catalog configs/catalog.tsv

Trường gốc: id, audio (MP3 16 kHz, ghi nguyên), text, duration (split test). License CC-BY-NC-4.0. Không có thông tin người nói:
speaker và source được đặt riêng cho từng câu, nên bootstrap trên phần này thực chất là theo câu.
Ghi: b1_candidates.jsonl + b1_candidates_review.csv, b5_ggroup.jsonl, term_candidates.tsv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converters.common import Writer, auto_tag, base_parser, load_hf, load_vi_words, safe, save_audio  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--names", nargs="+", default=["tech", "edu", "finance", "pubadmin"])
    ap.add_argument("--b5-from", nargs="*", default=["pubadmin"], help="bộ nào cho câu không có từ Anh vào B5")
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    vw = load_vi_words(args.vi_words)
    w = Writer(args.out, "ggroup")
    for name in args.names:
        ds = load_hf(f"g-group-ai-lab/vi-asr-{name}-test", "test", args.parquet.replace("{name}", name) if args.parquet else None,
                     limit=args.limit)
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            uid = f"gg_{name}_{safe(ex['id'])}"
            path, dur, _ = save_audio(ex["audio"], Path(args.out) / "audio" / name, uid)
            tag = auto_tag(ex["text"], cat, vi_words=vw)
            row = {"id": uid, "audio": path, "duration": dur, "speaker": f"gg_{name}_{safe(ex['id'])}",
                   "source": f"ggroup:{name}:{safe(ex['id'])}", "region": "unk", "gender": "u",
                   "channel": "wideband", "domain": name, "dataset": f"vi-asr-{name}-test"}
            if tag["spans"]:
                w.add("b1_candidates", dict(row, split="B1"), tag)
            elif name in args.b5_from:
                w.add("b5_ggroup", dict(row, split="B5"), tag)
    w.close(review=["b1_candidates"])


if __name__ == "__main__":
    main()
