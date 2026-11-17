#!/usr/bin/env python3
"""Các bộ tiếng Việt thuần đọc từ thư mục đã tải về: VIVOS, Common Voice, VietSuperSpeech.

  # VIVOS (bản gốc giải nén: vivos/{train,test}/prompts.txt + waves/<SPK>/<ID>.wav) -> B5 (test) + train
  python -m gipbench.dataset.ingest.local vivos --root data/dl/vivos --out data/raw/vivos
  # Common Voice vi (thư mục có test.tsv, train.tsv, clips/) -> B5 (test) + train
  python -m gipbench.dataset.ingest.local commonvoice --root data/dl/cv-corpus-vi --out data/raw/cv
  # pool replay nhiều người nói (validated + other, trừ người nói test / dev, trần 30 câu / người nói) -> pool_cv.jsonl
  python -m gipbench.dataset.ingest.local commonvoice_pool --root data/dl/cv-corpus-27.0-2026-09-11/vi --out data/raw/cv
  # VietSuperSpeech (thư mục có train.json / dev.json hoặc *.parquet, cột audio = đường dẫn tương đối) -> train replay
  python -m gipbench.dataset.ingest.local vietsuperspeech --root data/dl/VietSuperSpeech --out data/raw/vss       --bench data/raw/ggroup/b1_candidates.jsonl data/raw/vimedcss/b3_candidates.jsonl --catalog configs/catalog_v0.tsv

Audio KHÔNG được copy: manifest trỏ thẳng vào file gốc.
B5 chỉ nhận câu không có từ nào bị nghi là tiếng Anh. VietSuperSpeech là pseudo-label (Zipformer-30M): mặc định
train_vss / dev_vss (replay, chưng cất) chỉ giữ câu không có từ Anh (bỏ lọc bằng --keep-english).
Câu có từ Anh của phần train VSS được ghi riêng ra vss_cs_candidates.jsonl:
  dataset = vss_cs, train_only = true; CHỈ dùng cho train, không bao giờ vào bench. Trong experiment YAML (configs/model/experiments/*.yaml, mục pools),
  vss_cs thuộc nhóm riêng cs_pseudo (audio thật, nhãn giả), không thuộc cs_real.
  Câu rò rỉ với bench (--bench, mặc định chỉ xét split B1 B2 B3: speaker, source, text trùng; thuật ngữ reserved_b3
  trong --catalog) bị loại. Câu có từ Anh của phần dev bị bỏ (cùng video với dev_vss).
  Cần lọc thêm bằng src/gipbench/dataset/pseudolabel/agreement_filter.py (đồng thuận với faster-whisper / Qwen3-ASR) trước khi dùng.
  Ghi kèm vss_cs_terms.tsv: thuật ngữ (span tự gắn) và số lần.
--trust-duration: lấy duration trong train.json, không mở audio (để thống kê toàn bộ khi chưa tải audio).
"""
import csv
import json
import sys
from pathlib import Path

from collections import Counter

from gipbench.dataset.ingest.common import Writer, add_vi_words_arg, auto_tag, load_vi_words, safe, save_audio
from gipbench.core.leak import LeakGuard
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import Catalog


def emit(w, name, split, uid, audio, text, speaker, source, domain, dataset, cat, keep_english, stats,
         dur=None, cs_sink=None):
    """cs_sink(row, tag): nhận câu có từ Anh bị loại khỏi manifest chính (VSS -> vss_cs_candidates)."""
    if dur is None:
        try:
            path, dur, _ = save_audio(audio, w.out, uid)
        except Exception:  # noqa: BLE001
            stats["audio lỗi"] = stats.get("audio lỗi", 0) + 1
            return
    else:
        path = str(audio)
    tag = auto_tag(text, cat, vi_words=w.vi_words)
    row = {"id": uid, "split": split, "audio": path, "duration": dur, "speaker": speaker, "source": source,
           "region": "unk", "gender": "u", "channel": "wideband", "domain": domain, "dataset": dataset}
    if tag["spans"] and not keep_english:
        stats["bỏ vì có từ Anh"] = stats.get("bỏ vì có từ Anh", 0) + 1
        if cs_sink:
            cs_sink(dict(row, text=text), tag)
        return
    if split == "B5":
        w.add(name, row, dict(tag, text_tagged=text))
    else:
        row["text"] = text
        w.add(name, row)


def vivos(args, w, cat, stats):
    root = Path(args.root)
    for part, name, split in (("test", "b5_vivos", "B5"), ("train", "train_vivos", "train")):
        pf = root / part / "prompts.txt"
        if not pf.exists():
            print(f"  thiếu {pf}")
            continue
        for i, line in enumerate(open(pf, encoding="utf-8")):
            if args.limit and i >= args.limit:
                break
            uid, _, text = line.strip().partition(" ")
            spk = uid.split("_")[0]
            emit(w, name, split, f"vivos_{uid}", root / part / "waves" / spk / f"{uid}.wav", text.lower(),
                 f"vivos_{spk}", f"vivos:{spk}", "read", f"vivos:{part}", cat, False, stats)


def commonvoice(args, w, cat, stats):
    root = Path(args.root)
    for part, name, split in (("test", "b5_cv", "B5"), ("train", "train_cv", "train")):
        tsv = root / f"{part}.tsv"
        if not tsv.exists():
            print(f"  thiếu {tsv}")
            continue
        with open(tsv, encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)):
                if args.limit and i >= args.limit:
                    break
                spk = safe(r["client_id"])[:16]
                emit(w, name, split, f"cv_{safe(Path(r['path']).stem)}", root / "clips" / r["path"], r["sentence"],
                     f"cv_{spk}", f"cv:{spk}", "read", f"commonvoice:{part}", cat, False, stats)


def commonvoice_pool(args, w, cat, stats):
    """Pool replay nhiều người nói từ Common Voice (TODO 2c, IP-06) -> pool_cv.jsonl.

    validated.tsv trừ người nói test / dev chỉ còn đúng 1 người nói (người của train.tsv), nên lấy thêm phần chưa đủ phiếu
    (--cv-parts, mặc định validated + other) với luật phiếu --cv-min-up / --cv-max-down. Loại MỌI client_id có trong
    test.tsv / dev.tsv; --cv-max-per-speaker giới hạn số câu mỗi người nói (người nói train cũ có > 1 600 câu).
    Nhãn = câu nhắc (prompt) người nói được yêu cầu đọc: label_source ghi rõ phần / số phiếu để lọc lại sau.
    Duration lấy từ clip_durations.tsv (không mở mp3). Trùng với bench khoá (CV27 test trong G0, CV17 test trong E0;
    người nói, clip, câu nhắc) lọc tiếp bằng dataset.release.clean_pool."""
    import random
    root = Path(args.root)
    excl = set()
    for part in ("test", "dev"):
        with open(root / f"{part}.tsv", encoding="utf-8") as f:
            excl |= {r["client_id"] for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)}
    durs = {}
    with open(root / "clip_durations.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            durs[r["clip"]] = float(r["duration[ms]"]) / 1000
    cand = []
    for part in args.cv_parts:
        with open(root / f"{part}.tsv", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                up, down = int(r.get("up_votes") or 0), int(r.get("down_votes") or 0)
                why = ("người nói test/dev" if r["client_id"] in excl else
                       "phiếu" if part != "validated" and (up < args.cv_min_up or down > args.cv_max_down) else
                       "không có duration" if r["path"] not in durs else "")
                if why:
                    stats[f"bỏ ({why})"] = stats.get(f"bỏ ({why})", 0) + 1
                    continue
                cand.append((part, up, down, r))
    by_spk = {}
    for c in cand:
        by_spk.setdefault(c[3]["client_id"], []).append(c)
    rnd = random.Random(args.seed)
    for spk, cs in sorted(by_spk.items()):
        if args.cv_max_per_speaker and len(cs) > args.cv_max_per_speaker:
            stats["bỏ (trần người nói)"] = stats.get("bỏ (trần người nói)", 0) + len(cs) - args.cv_max_per_speaker
            cs = rnd.sample(cs, args.cv_max_per_speaker)
        for part, up, down, r in cs:
            s16 = safe(spk)[:16]
            before = len(w.rows.get("pool_cv", []))
            emit(w, "pool_cv", "train", f"cv_{safe(Path(r['path']).stem)}", root / "clips" / r["path"], r["sentence"],
                 f"cv_{s16}", f"cv:{s16}", "read", f"commonvoice:{part}", cat, False, stats, dur=durs[r["path"]])
            if len(w.rows.get("pool_cv", [])) > before:
                w.rows["pool_cv"][-1].update(
                    upstream_id=Path(r["path"]).stem, label_source=f"human_prompt:cv27_{part}(up={up},down={down})")
    rows = w.rows.get("pool_cv", [])
    stats["pool_cv: câu"] = len(rows)
    stats["pool_cv: người nói"] = len({r["speaker"] for r in rows})
    stats["pool_cv: giờ"] = round(sum(r["duration"] for r in rows) / 3600, 3)


def vietsuperspeech(args, w, cat, stats):
    root = Path(args.root)
    items = []
    for jf, part in (("train.json", "train"), ("dev.json", "dev")):
        p = root / jf
        if p.exists():
            raw = p.read_text(encoding="utf-8").strip()
            data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
            items += [(part, d) for d in data]
    if not items:
        import glob
        from datasets import load_dataset
        files = sorted(glob.glob(str(root / "**" / "*.parquet"), recursive=True))
        if not files:
            sys.exit(f"Không thấy train.json/dev.json hay *.parquet trong {root}")
        for f in files:
            part = "dev" if "valid" in Path(f).name or "dev" in Path(f).name else "train"
            items += [(part, d) for d in load_dataset("parquet", data_files=f, split="train")]
    guard = None
    if args.bench:
        guard = LeakGuard((r for b in args.bench for r in read_jsonl(b)), cat, args.bench_splits)
        print(f"  chặn rò rỉ vss_cs: {guard.n_bench} câu bench ({' '.join(args.bench_splits)}), "
              f"{len(guard.reserved)} thuật ngữ reserved_b3")
    else:
        print("  CẢNH BÁO: không truyền --bench, vss_cs_candidates CHƯA được kiểm tra rò rỉ với bench.")
    terms, cs_hours = Counter(), [0.0]
    srcs = {"train": set(), "dev": set()}

    def make_sink(part):
        def sink(row, tag):
            if part != "train":
                stats["vss_cs bỏ: câu có từ Anh ở phần dev"] = stats.get("vss_cs bỏ: câu có từ Anh ở phần dev", 0) + 1
                return
            why = guard.reason(row, row["text"]) if guard else ""
            if why:
                stats[f"vss_cs bỏ: {why}"] = stats.get(f"vss_cs bỏ: {why}", 0) + 1
                return
            row.update(split="train", dataset="vss_cs", train_only=True, label_source="pseudo:Zipformer-30M-RNNT-6000h")
            w.add("vss_cs_candidates", row, tag)
            terms.update(tag["spans"])
            cs_hours[0] += float(row["duration"]) / 3600
        return sink

    seen = {}
    for i, (part, d) in enumerate(items):
        seen[part] = seen.get(part, 0) + 1
        if args.limit and seen[part] > args.limit:  # --limit áp cho từng phần (train, dev), như vivos/commonvoice
            continue
        src = safe(d.get("source") or "unk")
        srcs.setdefault(part, set()).add(src)
        emit(w, f"{part}_vss", part, f"vss_{i:07d}", root / d["audio"], str(d["text"]).lower(), f"vss_{src}",
             f"vss:{src}", "conversation", "VietSuperSpeech", cat, args.keep_english, stats,
             dur=float(d["duration"]) if args.trust_duration else None,
             cs_sink=None if args.keep_english else make_sink(part))
    both = srcs["train"] & srcs["dev"]
    if both:
        print(f"  LƯU Ý: {len(both)} video (source) có ở cả train và dev của VSS, vd {sorted(both)[:2]}")
    if terms:
        tp = w.out / "vss_cs_terms.tsv"
        with open(tp, "w", encoding="utf-8") as f:
            f.write("term\tcount\n" + "".join(f"{t}\t{c}\n" for t, c in terms.most_common()))
        print(f"  vss_cs: {cs_hours[0]:.2f} h, {len(terms)} thuật ngữ khác nhau -> {tp}")
        print("  top 30: " + ", ".join(f"{t} ×{c}" for t, c in terms.most_common(30)))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["vivos", "commonvoice", "commonvoice_pool", "vietsuperspeech"])
    ap.add_argument("--cv-parts", nargs="+", default=["validated", "other"], help="(commonvoice_pool) các file .tsv lấy câu")
    ap.add_argument("--cv-min-up", type=int, default=0, help="(commonvoice_pool) phiếu up tối thiểu (ngoài validated)")
    ap.add_argument("--cv-max-down", type=int, default=0, help="(commonvoice_pool) phiếu down tối đa (ngoài validated)")
    ap.add_argument("--cv-max-per-speaker", type=int, default=30, help="(commonvoice_pool) 0 = không trần")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--keep-english", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bench", nargs="*", default=[], help="(vietsuperspeech) manifest bench để chặn rò rỉ vss_cs")
    ap.add_argument("--bench-splits", nargs="+", default=["B1", "B2", "B3"])
    ap.add_argument("--trust-duration", action="store_true",
                    help="(vietsuperspeech) lấy duration trong json, không mở audio")
    add_vi_words_arg(ap)
    args = ap.parse_args()
    w, stats = Writer(args.out, args.dataset), {}
    w.vi_words = load_vi_words(args.vi_words)
    {"vivos": vivos, "commonvoice": commonvoice, "commonvoice_pool": commonvoice_pool,
     "vietsuperspeech": vietsuperspeech}[args.dataset](args, w, Catalog.load(args.catalog), stats)
    w.terms.clear()
    w.close()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
