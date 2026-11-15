#!/usr/bin/env python3
"""ViMedCSS (tensorxt/ViMedCSS, CC-BY-4.0)  ->  B3 (split hard), bench ngoài (test), train, dev + catalog y tế.

  python converters/conv_vimedcss.py --out data/raw/vimedcss --catalog configs/catalog.tsv

Trường gốc: segment_id, audio, duration_seconds, segment_text, cs_terms_list ("a; b"), cs_terms_count, topic,
original_video_link, start_time, end_time. Split: train / validation / test / hard.
Thuật ngữ trong cs_terms_list được dò lại trong câu (khớp gần đúng) để gắn <en>. Nhãn gốc do Gemini tạo, nên phần
vào bench cần phiếu duyệt.
Mỗi câu có HAI nhãn:
  text_tagged       theo Spec: từ mượn có trong từ điển tiếng Việt không gắn tag (configs/vi_loanwords.txt:
                    axit, canxi = chính tả Việt; vitamin, protein = trùng mặt chữ, chờ duyệt), cộng catalog + heuristic.
  text_tagged_orig  theo dataset: chỉ gắn đúng các thuật ngữ trong cs_terms_list, không heuristic, không catalog.
Câu không dò ra thuật ngữ nào của dataset bị gắn cờ `term_not_found`. Chấm B3 với score.py --label-field
text_tagged (số chính) và --label-field text_tagged_orig (số phụ, so được với paper ViMedCSS).
Thuật ngữ CHỈ có ở split hard (theo nhãn Spec) được ghi reserved_b3=1 trong catalog_vimedcss.tsv.
Audio gốc: WAV 44.1 kHz, cột duration_seconds làm tròn theo giây (lệch audio thật tới ~1 s) -> audio được resample về
16 kHz mono (--target-sr) và duration lấy từ audio thật.
Người nói không có trong dataset: speaker = video (yt_<id>), source = youtube:<id>.
Ghi: b3_candidates.jsonl (+_review.csv), ext_vimedcss_test.jsonl (+_review.csv, split ghi B1 để chấm độc lập),
     train_vimedcss.jsonl, dev_vimedcss.jsonl, catalog_vimedcss.tsv.
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from converters.common import (Writer, auto_tag, base_parser, load_hf, load_vi_words, safe, save_audio,  # noqa: E402
                               youtube_id)
from lib.textnorm import Catalog  # noqa: E402

PLAN = {"hard": ("b3_candidates", "B3"), "test": ("ext_vimedcss_test", "B1"),
        "train": ("train_vimedcss", "train"), "validation": ("dev_vimedcss", "dev")}


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--splits", nargs="+", default=list(PLAN))
    ap.add_argument("--target-sr", type=int, default=16000,
                    help="audio gốc là WAV 44.1 kHz: resample về tần số này khi ghi (0 = giữ nguyên)")
    ap.add_argument("--streaming", action="store_true",
                    help="đọc HF dạng streaming (không cache ~14 GB parquet train; dùng khi ổ đĩa không đủ chỗ)")
    args = ap.parse_args()
    cat = Catalog.load(args.catalog)
    vw = load_vi_words(args.vi_words)  # gồm cả same_surface (vitamin...), như mọi bộ (decisions §12)
    w = Writer(args.out, "vimedcss")
    surface = defaultdict(Counter)  # split -> dạng viết thực tế của thuật ngữ (theo nhãn Spec)
    untagged = Counter()            # thuật ngữ dataset bị bỏ tag vì là từ mượn
    for split in args.splits:
        name, bsplit = PLAN[split]
        # có --limit thì chỉ đọc N dòng đầu, không tải cả split (train ~24 h audio, ~14 GB)
        ds = load_hf("tensorxt/ViMedCSS", split, args.parquet.replace("{split}", split) if args.parquet else None,
                     streaming=args.streaming, limit=args.limit)
        for i, ex in enumerate(ds):
            if args.limit and i >= args.limit:
                break
            uid = f"vmc_{safe(ex['segment_id'])}"
            terms = [t.strip() for t in str(ex.get("cs_terms_list") or "").split(";") if t.strip()]
            orig = auto_tag(ex["segment_text"], None, terms, vi_words=set(), heuristic=False, term_dec={})  # nhãn gốc
            tag = auto_tag(ex["segment_text"], cat, terms, vi_words=vw)
            if not orig["spans"]:
                tag["flags"].append("term_not_found")
            dropped = set(orig["spans"]) - set(tag["spans"])
            if dropped:
                tag["flags"].append("loanword_untagged")
                untagged.update(dropped)
            for s, k in zip(tag["spans"], tag["span_kinds"]):
                if k == "C":
                    surface[split][s] += 1
            vid = youtube_id(ex.get("original_video_link") or "")
            path, dur, _ = save_audio(ex["audio"], Path(args.out) / "audio" / split, uid, args.target_sr)
            w.add(name, {"id": uid, "split": bsplit, "audio": path, "duration": dur, "speaker": f"yt_{vid}",
                         "source": f"youtube:{vid}", "region": "unk", "gender": "u", "channel": "wideband",
                         "domain": "medical", "topic": ex.get("topic"), "dataset": f"ViMedCSS:{split}",
                         "text_tagged_orig": orig["text_tagged"]}, tag)
    w.close(review=["b3_candidates", "ext_vimedcss_test"])
    if untagged:
        print(f"  bỏ tag {sum(untagged.values())} lần thuật ngữ dataset là từ mượn (vi_loanwords): "
              f"{', '.join(f'{t} ×{c}' for t, c in untagged.most_common(12))}")
    other = set()
    for sp, c in surface.items():
        if sp != "hard":
            other |= set(c)
    allc = Counter()
    for c in surface.values():
        allc.update(c)
    p = Path(args.out) / "catalog_vimedcss.tsv"
    with open(p, "w", encoding="utf-8") as f:
        f.write("term\tcategory\tvariants\tphonetic\treserved_b3\n")
        for t, _ in allc.most_common():
            reserved = int(t in surface.get("hard", {}) and t not in other)
            f.write(f"{t}\tmedical\t\t\t{reserved}\n")
    print(f"  {p}: {len(allc)} thuật ngữ y tế ({sum(1 for t in allc if t in surface.get('hard', {}) and t not in other)} reserved_b3)")


if __name__ == "__main__":
    main()
