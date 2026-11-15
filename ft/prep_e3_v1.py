#!/usr/bin/env python3
"""Chuẩn bị dữ liệu E3 v1: tách dev theo người nói, chọn nguồn distill (VSS không có từ Anh).

  python ft/prep_e3_v1.py --out data/e3_v1 --distill-hours 10

Ghi vào --out:
  train_vss_cs.jsonl / dev_cs_vss.jsonl   vss_cs (đã gắn domain) tách theo video, dev ~--dev-frac số giờ
  train_vivos.jsonl / dev_vi.jsonl        VIVOS train tách theo người nói, dev ~--dev-frac số giờ
  train_confusable.jsonl                  confusable train, bỏ câu của người nói thuộc dev_vi
  distill_pool.jsonl                      VSS train không thuộc vss_cs_candidates (prefilter: không có từ Anh), không
                                          trùng video của vss_cs (train lẫn dev), 1–20 s, lấy ngẫu nhiên tới --distill-hours
Chỉ lấy mẫu theo seed; không đụng tới audio. Audio ghi đường dẫn tuyệt đối.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402


def hours(rows):
    return sum(float(r["duration"]) for r in rows) / 3600


def split_by_speaker(rows, frac, rnd):
    by = defaultdict(list)
    for r in rows:
        by[r["speaker"]].append(r)
    spk = sorted(by)
    rnd.shuffle(spk)
    target, dev, tr = frac * hours(rows), [], []
    for s in spk:
        (dev if hours(dev) < target else tr).extend(by[s])
    return tr, dev


def absolute(rows):
    for r in rows:
        r["audio"] = r.get("_audio_path", r["audio"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/e3_v1")
    ap.add_argument("--vss-cs", default="data/raw/vss_full/vss_cs_clean_dom.jsonl")
    ap.add_argument("--vivos", default="data/raw/vivos/train_vivos.jsonl")
    ap.add_argument("--confusable", default="data/raw/confusable/train_confusable.jsonl")
    ap.add_argument("--vss", default="data/raw/vss_full/train_vss_dom.jsonl")
    ap.add_argument("--vss-cs-candidates", default="data/raw/vss_full/vss_cs_candidates.jsonl")
    ap.add_argument("--dev-frac", type=float, default=0.05)
    ap.add_argument("--distill-hours", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(args.seed)

    def dump(rows, name):
        p = out / name
        write_jsonl(str(p), rows)
        print(f"{p}: {len(rows)} câu, {hours(rows):.2f} h, {len({r['speaker'] for r in rows})} người nói")

    cs = absolute(read_jsonl(args.vss_cs))
    cs_tr, cs_dev = split_by_speaker(cs, args.dev_frac, rnd)
    dump(cs_tr, "train_vss_cs.jsonl")
    dump(cs_dev, "dev_cs_vss.jsonl")

    vi = absolute(read_jsonl(args.vivos))
    vi_tr, vi_dev = split_by_speaker(vi, args.dev_frac, rnd)
    dump(vi_tr, "train_vivos.jsonl")
    dump(vi_dev, "dev_vi.jsonl")
    dev_spk = {r["speaker"] for r in vi_dev}
    conf = [r for r in absolute(read_jsonl(args.confusable)) if r["speaker"] not in dev_spk]
    dump(conf, "train_confusable.jsonl")

    cand = {json.loads(l)["id"] for l in open(args.vss_cs_candidates, encoding="utf-8") if l.strip()}
    cs_spk = {r["speaker"] for r in cs}
    pool = [r for r in absolute(read_jsonl(args.vss)) if r["id"] not in cand and r["speaker"] not in cs_spk
            and 1.0 <= float(r["duration"]) <= 20.0]
    print(f"VSS không có từ Anh, khác video vss_cs, 1–20 s: {len(pool)} câu, {hours(pool):.1f} h")
    rnd.shuffle(pool)
    pick, h = [], 0.0
    for r in pool:
        if h >= args.distill_hours * 3600:
            break
        pick.append(r)
        h += float(r["duration"])
    dump(pick, "distill_pool.jsonl")


if __name__ == "__main__":
    main()
