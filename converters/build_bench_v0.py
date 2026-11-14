#!/usr/bin/env python3
"""Ghép bench từ các manifest ứng viên + phiếu duyệt, gộp catalog.

  python converters/build_bench_v0.py \
      --candidates data/raw/ggroup/b1_candidates.jsonl data/raw/ggroup/b5_ggroup.jsonl \
                   data/raw/vimedcss/b3_candidates.jsonl data/raw/sea/b1_candidates_sea.jsonl \
                   data/raw/vivos/b5_vivos.jsonl data/raw/cv/b5_cv.jsonl \
      --reviews data/raw/ggroup/b1_candidates_review.csv data/raw/vimedcss/b3_candidates_review.csv \
      --catalogs configs/catalog.tsv data/raw/vimedcss/catalog_vimedcss.tsv \
      --accept reviewed --out data/bench_v0.jsonl --catalog-out configs/catalog_v0.tsv

--accept:
  reviewed (mặc định)  chỉ nhận câu người duyệt ghi decision = ok hoặc fix; B5 không có cờ thì nhận thẳng
  catalog              thêm câu chưa duyệt mà mọi span đều khớp catalog và không có cờ
  all                  nhận hết — CHỈ để chạy thử pipeline (bench nháp, không dùng để kết luận)
Phiếu duyệt: decision = ok | fix (dùng text_tagged_fixed) | drop; viet_levels = "strong,mid" theo thứ tự span;
region / gender điền nếu nghe ra.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.manifest import read_jsonl  # noqa: E402
from lib.textnorm import basic_tokens, tagged_well_formed  # noqa: E402


def merge_catalogs(paths, out):
    rows, order = {}, []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t"):
                key = " ".join(basic_tokens(r["term"]))
                if not key:
                    continue
                if key not in rows:
                    rows[key] = {"term": key, "category": r.get("category") or "", "variants": set(), "phonetic": set(),
                                 "reserved_b3": set()}
                    order.append(key)
                e = rows[key]
                e["category"] = e["category"] or (r.get("category") or "")
                e["variants"] |= {v.strip() for v in (r.get("variants") or "").split("|") if v.strip()}
                e["phonetic"] |= {v.strip() for v in (r.get("phonetic") or "").split("|") if v.strip()}
                e["reserved_b3"].add((r.get("reserved_b3") or "0").strip() in ("1", "true", "yes"))
    conflicts = [k for k in order if len(rows[k]["reserved_b3"]) > 1]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("term\tcategory\tvariants\tphonetic\treserved_b3\n")
        for k in order:
            e = rows[k]
            f.write(f"{k}\t{e['category']}\t{'|'.join(sorted(e['variants']))}\t{'|'.join(sorted(e['phonetic']))}\t{int(True in e['reserved_b3'])}\n")
    print(f"Catalog gộp: {len(order)} thuật ngữ -> {out}")
    if conflicts:
        print(f"  CẢNH BÁO: {len(conflicts)} thuật ngữ vừa reserved_b3 vừa thường (giữ reserved): {conflicts[:8]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--reviews", nargs="*", default=[])
    ap.add_argument("--catalogs", nargs="*", default=[])
    ap.add_argument("--catalog-out", default="configs/catalog_v0.tsv")
    ap.add_argument("--accept", choices=["reviewed", "catalog", "all"], default="reviewed")
    ap.add_argument("--max-per-source", type=int, default=0, help="trần số câu mỗi source (0 = không giới hạn)")
    ap.add_argument("--out", default="data/bench_v0.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    review = {}
    for rp in args.reviews:
        with open(rp, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                review[r["id"]] = r
    out_dir = Path(args.out).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    kept, why = [], Counter()
    seen = set()
    for cp in args.candidates:
        for r in read_jsonl(cp):
            if r["id"] in seen:
                why["bỏ: id trùng"] += 1
                continue
            at = r.get("auto_tag", {})
            rv = review.get(r["id"], {})
            dec = (rv.get("decision") or "").strip().lower()
            if dec == "drop":
                why["bỏ: người duyệt loại"] += 1
                continue
            if dec == "fix":
                fixed = (rv.get("text_tagged_fixed") or "").strip()
                if not fixed or tagged_well_formed(fixed):
                    why["bỏ: fix nhưng text_tagged_fixed trống / tag hỏng"] += 1
                    continue
                r["text_tagged"] = fixed
            clean_b5 = r["split"] == "B5" and at.get("confidence", "none") == "none" and not at.get("flags")
            auto_ok = at.get("confidence") == "catalog" and not at.get("flags")
            if dec in ("ok", "fix"):
                why[f"nhận: đã duyệt ({r['split']})"] += 1
            elif clean_b5:
                why["nhận: B5 không có từ Anh"] += 1
            elif args.accept == "all" or (args.accept == "catalog" and auto_ok):
                why[f"nhận: CHƯA duyệt ({r['split']}, --accept {args.accept})"] += 1
            else:
                why[f"chờ duyệt ({r['split']})"] += 1
                continue
            lv = [x.strip() for x in (rv.get("viet_levels") or "").split(",") if x.strip()]
            if lv:
                r["en_meta"] = [{"viet_level": x} for x in lv]
            for k in ("region", "gender"):
                if (rv.get(k) or "").strip():
                    r[k] = rv[k].strip()
            r["review"] = dec or "none"
            r["audio"] = os.path.relpath(r["_audio_path"], out_dir)
            seen.add(r["id"])
            kept.append(r)
    if args.max_per_source:
        rnd = random.Random(args.seed)
        by = defaultdict(list)
        for r in kept:
            by[(r["split"], r["source"])].append(r)
        kept = []
        for rows in by.values():
            if len(rows) > args.max_per_source:
                why["bỏ: vượt trần mỗi source"] += len(rows) - args.max_per_source
                rows = rnd.sample(rows, args.max_per_source)
            kept += rows
    kept.sort(key=lambda r: (r["split"], r["id"]))
    with open(args.out, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    print(f"Bench: {len(kept)} câu -> {args.out}")
    by_split = defaultdict(lambda: [0, 0.0])
    for r in kept:
        by_split[r["split"]][0] += 1
        by_split[r["split"]][1] += float(r["duration"])
    for s in sorted(by_split):
        print(f"  {s}: {by_split[s][0]} câu, {by_split[s][1] / 3600:.2f} h")
    for k, v in sorted(why.items()):
        print(f"  {v:6d}  {k}")
    if args.catalogs:
        merge_catalogs(args.catalogs, args.catalog_out)
    unrev = sum(1 for r in kept if r["review"] == "none" and r["split"] != "B5")
    if unrev:
        print(f"LƯU Ý: {unrev} câu B1–B4 chưa được người duyệt — bench này chỉ là bản nháp.")


if __name__ == "__main__":
    main()
