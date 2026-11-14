#!/usr/bin/env python3
"""Kiểm tra manifest bench trước khi đóng băng. Thoát mã 1 nếu có LỖI.

  python validate.py --bench data/bench.jsonl --catalog configs/catalog.tsv \
      [--train data/train_cs.jsonl data/train_vi.jsonl] [--check-audio]

LỖI (phải sửa): thiếu trường, id trùng, tag hỏng, B1–B4 không có span tiếng Anh,
  speaker/source của bench xuất hiện trong train, thuật ngữ dành riêng B3 có trong train,
  audio không đọc được / sai thời lượng, audio trùng hash giữa bench và train.
CẢNH BÁO: span không có trong catalog, B5 có span tiếng Anh, thời lượng ngoài 1–30 s,
  text còn chữ số, thuật ngữ B3 xuất hiện > 2 lần, cùng speaker/source ở nhiều split bench.
"""
import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.manifest import BENCH_SPLITS, CS_SPLITS, REQUIRED, read_jsonl  # noqa: E402
from lib.textnorm import Catalog, basic_tokens, normalize_ref, plain_text, tagged_well_formed  # noqa: E402


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--train", nargs="*", default=[], help="manifest train (JSONL cùng định dạng, split tuỳ ý)")
    ap.add_argument("--check-audio", action="store_true", help="mở từng file audio (chậm hơn)")
    ap.add_argument("--max-show", type=int, default=15)
    args = ap.parse_args()

    cat = Catalog.load(args.catalog)
    bench = read_jsonl(args.bench)
    errors, warns = defaultdict(list), defaultdict(list)

    ids = Counter(r.get("id") for r in bench)
    for i, c in ids.items():
        if c > 1:
            errors["id trùng"].append(f"{i} ×{c}")

    reserved = {t for t, v in cat.terms.items() if v.reserved_b3}
    b3_term_counts = Counter()
    split_of_speaker, split_of_source = defaultdict(set), defaultdict(set)

    for r in bench:
        rid = r.get("id", f"dòng {r['_line']}")
        miss = [k for k in REQUIRED if k not in r or r[k] in (None, "")]
        if miss:
            errors["thiếu trường"].append(f"{rid}: {', '.join(miss)}")
            continue
        if r["split"] not in BENCH_SPLITS:
            errors["split lạ"].append(f"{rid}: {r['split']}")
        bad = tagged_well_formed(r["text_tagged"])
        if bad:
            errors["tag hỏng"].append(f"{rid}: {bad}")
            continue
        toks, spans = normalize_ref(r["text_tagged"], cat)
        if r["split"] in CS_SPLITS and not spans:
            errors["B1–B4 không có span <en>"].append(rid)
        if r["split"] == "B5" and spans:
            warns["B5 có span <en> (nên là tiếng Việt thuần)"].append(f"{rid}: {[' '.join(s) for s in spans]}")
        if cat.terms:
            for s in spans:
                if s not in cat.terms:
                    warns["span không có trong catalog"].append(f"{rid}: {' '.join(s)}")
        if r["split"] == "B3":
            for s in spans:
                b3_term_counts[s] += 1
                if reserved and s not in reserved:
                    warns["span B3 không thuộc nhóm reserved_b3"].append(f"{rid}: {' '.join(s)}")
        if any(ch.isdigit() for ch in plain_text(r["text_tagged"])):
            warns["text còn chữ số (nên viết dạng đọc)"].append(f"{rid}: {r['text_tagged'][:60]}")
        d = float(r["duration"])
        if d < 1 or d > 30:
            warns["thời lượng ngoài 1–30 s"].append(f"{rid}: {d:.1f}s")
        split_of_speaker[r["speaker"]].add(r["split"])
        split_of_source[r["source"]].add(r["split"])
        if args.check_audio:
            p = r.get("_audio_path")
            try:
                import soundfile as sf
                info = sf.info(p)
                real = info.frames / info.samplerate
                if abs(real - d) > 0.1:
                    errors["duration khác thời lượng audio thật"].append(f"{rid}: ghi {d:.2f}s, thật {real:.2f}s")
                if info.samplerate < 8000:
                    errors["sample rate quá thấp"].append(f"{rid}: {info.samplerate}")
            except Exception as e:  # noqa: BLE001
                errors["audio không đọc được"].append(f"{rid}: {p} ({e.__class__.__name__})")

    for t, c in b3_term_counts.items():
        if c > 2:
            warns["thuật ngữ B3 xuất hiện > 2 lần"].append(f"{' '.join(t)} ×{c}")
    for spk, ss in split_of_speaker.items():
        if len(ss) > 1 and not ss <= {"B4"}:
            warns["speaker có mặt ở nhiều split bench"].append(f"{spk}: {sorted(ss)}")
    for src, ss in split_of_source.items():
        if len(ss) > 1:
            warns["source có mặt ở nhiều split bench"].append(f"{src}: {sorted(ss)}")

    # ---- rò rỉ sang train
    if args.train:
        bench_spk = {r["speaker"] for r in bench if "speaker" in r}
        bench_src = {r["source"] for r in bench if "source" in r}
        bench_hash = {}
        if args.check_audio:
            for r in bench:
                try:
                    bench_hash[md5(r["_audio_path"])] = r["id"]
                except OSError:
                    pass
        for tp in args.train:
            for r in read_jsonl(tp):
                rid = f"{Path(tp).name}:{r.get('id')}"
                if r.get("speaker") in bench_spk:
                    errors["RÒ RỈ: speaker của bench có trong train"].append(f"{rid} ({r['speaker']})")
                if r.get("source") in bench_src:
                    errors["RÒ RỈ: source/video của bench có trong train"].append(f"{rid} ({r['source']})")
                text = r.get("text_tagged") or r.get("text") or ""
                toks = basic_tokens(plain_text(text))
                for t in cat.find_terms(cat.apply(toks)):
                    if t in reserved:
                        errors["RÒ RỈ: thuật ngữ reserved_b3 có trong train"].append(f"{rid}: {' '.join(t)}")
                if args.check_audio and r.get("_audio_path"):
                    try:
                        h = md5(r["_audio_path"])
                        if h in bench_hash:
                            errors["RÒ RỈ: audio trùng hash với bench"].append(f"{rid} = {bench_hash[h]}")
                    except OSError:
                        pass

    def show(title, d):
        print(f"\n== {title}: {sum(len(v) for v in d.values())}")
        for k, v in d.items():
            print(f"  [{len(v)}] {k}")
            for x in v[: args.max_show]:
                print(f"      {x}")
            if len(v) > args.max_show:
                print(f"      … và {len(v) - args.max_show} dòng nữa")

    print(f"Manifest: {args.bench} — {len(bench)} utterance")
    show("LỖI", errors)
    show("CẢNH BÁO", warns)
    print("\nKẾT QUẢ:", "KHÔNG ĐẠT — sửa lỗi trước khi đóng băng" if errors else "ĐẠT")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
