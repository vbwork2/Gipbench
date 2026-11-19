#!/usr/bin/env python3
"""Phiếu nghe kiểm chất lượng nhãn giả vss_cs (TODO 2d, audit E.3 / R11 R24) và chấm phiếu đã nghe.

  # 1. lấy mẫu (240 câu giữ, phân tầng 12 ô × 20 + 60 câu bị loại sát ranh giới làm đối chứng; 60 câu chung cho người nghe 2)
  python -m gipbench.dataset.pseudolabel.listen_sample sample --kept pool/vss_cs@e3_v1 \
      --candidates data/raw/vss_full/vss_cs_pre_rebuild.jsonl --out data/listen/vss_cs_v1
  # 2. mở src/gipbench/dataset/label/review_bench.html, "Mở CSV phiếu duyệt" = <out>/listen_A.csv, "Chọn thư mục audio" =
  #    <out>/audio. Mỗi câu: ok (nhãn đúng hết) / fix (sửa text_tagged_fixed: sửa chữ VÀ tag <en>…</en> cho đúng cách nói) /
  #    drop (audio hỏng, không nghe được). Bấm "Xuất CSV", lưu thành <out>/listen_A_reviewed.csv. Người nghe thứ hai làm
  #    listen_B.csv (60 câu chung) -> listen_B_reviewed.csv.
  # 3. chấm
  python -m gipbench.dataset.pseudolabel.listen_sample score --dir data/listen/vss_cs_v1 \
      --review listen_A_reviewed.csv [--review-b listen_B_reviewed.csv]

Phân tầng câu GIỮ (đúng đề xuất audit E.3): số span <en> (1 / 2+) × WER phần Việt của nhãn so với ASR khác
(`agreement.vi_wer`: 0–0.05 / 0.05–0.10 / 0.10–0.15) × tần suất thuật ngữ hiếm nhất của câu trong pool (cao: >= --freq-high
câu / thấp). Ô thiếu câu thì lấy hết và ghi vào báo cáo (không bù từ ô khác, để tỉ lệ từng ô không lệch).
Đối chứng (câu BỊ LOẠI, sát ranh giới, tính lại bằng đúng luật agreement_filter): `vi_wer_boundary` = mọi span được xác nhận
nhưng 0.15 < vi_wer <= 0.25; `span_unconfirmed` = vi_wer <= 0.15 nhưng còn span chưa được ASR nào xác nhận. Ưu tiên câu
có audio local; không có thì cột audio là URL HF (trang duyệt phát được nếu trình duyệt truy cập được HF).
Phiếu nghe KHÔNG có cột tầng / giữ-loại / hyp ASR (nghe mù); khoá ở <out>/sample_key.csv. Thứ tự câu xáo trộn (--seed).

Chấm (score):
  (a) lỗi span: span <en> của nhãn không còn (cùng chuỗi token) trong text_tagged_fixed -> span sai. Báo tỉ lệ span sai
      và tỉ lệ câu có >= 1 span sai, kèm CI95 Wilson, toàn cục và theo ô;
  (b) WER phần tiếng Việt của nhãn so với bản sửa (token ngoài <en>);
  (c) tỉ lệ câu có từ Anh bị viết kiểu Việt / không tag: bản sửa có span <en> mới không có trong nhãn;
  đối chứng: tỉ lệ câu bị loại mà thật ra đúng (ok) = tỉ lệ loại nhầm; kappa giữa 2 người nghe (ok / không ok) trên câu chung.
Luật quyết định (audit E.3, đăng ký trước): CẬN DƯỚI CI95 của tỉ lệ span sai (câu giữ) > 10% -> không dùng vss_cs làm
cs_pseudo (X4 bỏ). Ghi <dir>/score.json + score.md.
"""
import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from gipbench.core import release as R
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import Catalog, basic_tokens, parse_tagged

HF_URL = "https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech/resolve/main/"
REVIEW_COLS = ["id", "split", "audio", "duration", "text_tagged", "text_tagged_orig", "confidence", "flags", "decision",
               "text_tagged_fixed", "viet_levels", "reading_note", "region", "gender", "note"]
VI_BINS = ((0.0, 0.05), (0.05, 0.10), (0.10, 0.15))


# ------------------------------------------------------------------ tiện ích
def spans_of(text_tagged: str) -> List[str]:
    return [" ".join(basic_tokens(seg)) for seg, en in parse_tagged(text_tagged or "") if en and basic_tokens(seg)]


def vi_tokens(text_tagged: str) -> List[str]:
    return [t for seg, en in parse_tagged(text_tagged or "") if not en for t in basic_tokens(seg)]


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """(tỉ lệ, cận dưới, cận trên) CI95 Wilson; n = 0 -> (nan, 0, 1)."""
    if n == 0:
        return float("nan"), 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def kappa(a: Sequence[str], b: Sequence[str]) -> float:
    """Cohen's kappa cho hai dãy nhãn cùng độ dài."""
    n = len(a)
    if not n:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def vi_bin(w: float) -> Optional[str]:
    for lo, hi in VI_BINS:
        if (w >= lo if lo == 0 else w > lo) and w <= hi:
            return f"{lo:.2f}-{hi:.2f}"
    return None


def load_rows(ref_or_path: str) -> List[Dict]:
    p = Path(ref_or_path)
    if not p.exists() and "@" in ref_or_path:
        return read_jsonl(str(R.resolve(ref_or_path).manifest))
    return read_jsonl(str(p))


def local_audio(r: Dict) -> Optional[Path]:
    for p in (r.get("_audio_path"), r.get("audio")):
        if p and Path(p).exists():
            return Path(p)
    return None


# ------------------------------------------------------------------ lấy mẫu
def stratify(kept: List[Dict], freq_high: int) -> Dict[str, List[Dict]]:
    """Ô (n_span, vi_wer, tần suất) -> các câu giữ."""
    freq = Counter(s for r in kept for s in set(spans_of(r.get("text_tagged"))))
    cells = defaultdict(list)
    for r in kept:
        sp = spans_of(r.get("text_tagged"))
        vw = (r.get("agreement") or {}).get("vi_wer")
        if not sp or vw is None or vi_bin(vw) is None:
            continue
        fr = "cao" if min(freq[s] for s in set(sp)) >= freq_high else "thap"
        cells[f"span{'1' if len(sp) == 1 else '2+'}|vi{vi_bin(vw)}|freq_{fr}"].append(r)
    return cells


def controls(cands: List[Dict], kept_ids: set, hyps: Sequence[str], hash_index: str, audio_root: str, cat: Catalog,
             n_each: int, rnd: random.Random) -> Dict[str, List[Dict]]:
    """Câu bị loại sát ranh giới, tính lại theo luật agreement_filter (min_agree 1, mọi span)."""
    from gipbench.dataset.pseudolabel.agreement import agree, vi_wer
    from gipbench.infer.cache import HashIndex, HypLookup
    look = HypLookup(hyps, HashIndex(hash_index) if hash_index else None, audio_root)
    pool = {"vi_wer_boundary": [], "span_unconfirmed": []}
    for r in cands:
        if r["id"] in kept_ids or not spans_of(r.get("text_tagged")):
            continue
        hs = look.get(r)
        if not all(h is not None for h in hs):
            continue
        res = agree(r["text_tagged"], hs, cat, 1, 1.0)
        avail = [h for h in hs if (h or "").strip()]
        vw = min((vi_wer(r["text_tagged"], h, cat) for h in avail), default=float("inf"))
        if res["keep"] and 0.15 < vw <= 0.25:
            pool["vi_wer_boundary"].append(dict(r, _vw=vw))
        elif not res["keep"] and res["spans"] and vw <= 0.15:
            pool["span_unconfirmed"].append(dict(r, _vw=vw))
    out = {}
    for k, rs in pool.items():
        rs.sort(key=lambda r: (local_audio(r) is None, rnd.random()))  # ưu tiên có audio local
        out[k] = rs[:n_each]
    return out


def cmd_sample(args):
    rnd = random.Random(args.seed)
    kept = load_rows(args.kept)
    cells = stratify(kept, args.freq_high)
    pick: List[Tuple[Dict, str, str]] = []  # (row, nhóm, ô)
    short = {}
    for c in sorted(cells):
        rs = cells[c][:]
        rnd.shuffle(rs)
        if len(rs) < args.per_cell:
            short[c] = len(rs)
        pick += [(r, "kept", c) for r in rs[:args.per_cell]]
    ctrl = {}
    if args.candidates:
        ctrl = controls(load_rows(args.candidates), {r["id"] for r in kept}, args.hyps, args.hash_index, args.audio_root,
                        Catalog.load(args.catalog), args.controls // 2, rnd)
        for k, rs in ctrl.items():
            pick += [(r, "rejected", k) for r in rs]
    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rnd.shuffle(pick)
    key, rows_a, n_local = [], [], 0
    for r, grp, cell in pick:
        src = local_audio(r)
        if src is not None:
            dst = out / "audio" / f"{r['id']}{src.suffix}"
            if not dst.exists():
                shutil.copyfile(src, dst)
            audio = f"audio/{dst.name}"
            n_local += 1
        else:
            from gipbench.infer.cache import audio_rel
            audio = HF_URL + audio_rel(r, args.audio_root)
        rows_a.append({"id": r["id"], "split": "listen", "audio": audio, "duration": r.get("duration", ""),
                       "text_tagged": r.get("text_tagged", ""), "text_tagged_fixed": r.get("text_tagged", ""),
                       "region": r.get("region", "unk"), "gender": r.get("gender", "u")})
        vw = r.get("_vw", (r.get("agreement") or {}).get("vi_wer"))
        key.append({"id": r["id"], "group": grp, "cell": cell, "vi_wer": "" if vw is None else round(vw, 4),
                    "n_spans": len(spans_of(r.get("text_tagged"))), "source": r.get("source", ""),
                    "label_source": r.get("label_source", ""), "audio_local": int(src is not None)})
    kept_rows = [x for x, k in zip(rows_a, key) if k["group"] == "kept"]
    rows_b = rnd.sample(kept_rows, min(args.overlap, len(kept_rows)))
    for name, rs in (("listen_A.csv", rows_a), ("listen_B.csv", rows_b)):
        with open(out / name, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_COLS)
            w.writeheader()
            for x in rs:
                w.writerow({c: x.get(c, "") for c in REVIEW_COLS})
    with open(out / "sample_key.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0]) if key else ["id"])
        w.writeheader()
        w.writerows(key)
    meta = {"kept": args.kept, "candidates": args.candidates, "seed": args.seed, "per_cell": args.per_cell,
            "freq_high": args.freq_high, "cells": {c: len(v) for c, v in sorted(cells.items())}, "short_cells": short,
            "n_kept_sampled": sum(k["group"] == "kept" for k in key),
            "n_controls": {k: len(v) for k, v in ctrl.items()}, "n_overlap_B": len(rows_b), "audio_local": n_local,
            "audio_url": len(key) - n_local}
    (out / "sample_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"Phiếu: {out / 'listen_A.csv'} ({len(rows_a)} câu), {out / 'listen_B.csv'} ({len(rows_b)} câu chung); "
          f"audio: {out / 'audio'}; khoá (đừng mở khi nghe): {out / 'sample_key.csv'}")


# ------------------------------------------------------------------ chấm
def read_csv(p: Path) -> List[Dict]:
    with open(p, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def judge(r: Dict) -> Optional[Dict]:
    """Một câu đã nghe -> {decision, spans, spans_wrong, new_spans, vi_err, vi_n} (None nếu chưa nghe)."""
    dec = (r.get("decision") or "").strip().lower()
    if dec not in ("ok", "fix", "drop"):
        return None
    lab = r.get("text_tagged") or ""
    fixed = lab if dec == "ok" or not (r.get("text_tagged_fixed") or "").strip() else r["text_tagged_fixed"]
    sp, sf = spans_of(lab), spans_of(fixed)
    left = Counter(sf)
    wrong = 0
    for s in sp:
        if left[s] > 0:
            left[s] -= 1
        else:
            wrong += 1
    a, b = vi_tokens(lab), vi_tokens(fixed)
    return {"decision": dec, "spans": len(sp), "spans_wrong": wrong, "new_spans": sum(left.values()),
            "vi_err": edit_distance(a, b), "vi_n": len(b)}


def summarize(js: List[Dict]) -> Dict:
    js = [j for j in js if j["decision"] != "drop"]
    ns, nw = sum(j["spans"] for j in js), sum(j["spans_wrong"] for j in js)
    p, lo, hi = wilson(nw, ns)
    ps, los, his = wilson(sum(j["spans_wrong"] > 0 for j in js), len(js))
    pu, lou, hiu = wilson(sum(j["new_spans"] > 0 for j in js), len(js))
    vi_n = sum(j["vi_n"] for j in js)
    r4 = lambda t: [round(x, 4) for x in t]  # noqa: E731
    return {"n": len(js), "spans": ns, "span_error": r4((p, lo, hi)), "sent_with_span_error": r4((ps, los, his)),
            "sent_untagged_english": r4((pu, lou, hiu)),
            "vi_wer": round(sum(j["vi_err"] for j in js) / vi_n, 4) if vi_n else None,
            "ok_rate": round(sum(j["decision"] == "ok" for j in js) / len(js), 4) if js else None}


def cmd_score(args):
    d = Path(args.dir)
    key = {k["id"]: k for k in read_csv(d / "sample_key.csv")}
    ra = {r["id"]: r for r in read_csv(d / args.review)}
    judged = {i: j for i, r in ra.items() if i in key for j in [judge(r)] if j}
    drops = sum(j["decision"] == "drop" for j in judged.values())
    by = defaultdict(list)
    for i, j in judged.items():
        by[(key[i]["group"], key[i]["cell"])].append(j)
    kept = [j for i, j in judged.items() if key[i]["group"] == "kept"]
    rej = [j for i, j in judged.items() if key[i]["group"] == "rejected"]
    res = {"n_judged": len(judged), "n_total": len(key), "drop_audio": drops, "kept": summarize(kept),
           "cells": {f"{g}|{c}": summarize(v) for (g, c), v in sorted(by.items())}}
    if rej:
        nr = [j for j in rej if j["decision"] != "drop"]
        p, lo, hi = wilson(sum(j["decision"] == "ok" for j in nr), len(nr))
        res["rejected_false_reject"] = [round(p, 4), round(lo, 4), round(hi, 4), len(nr)]
    if args.review_b:
        rb = {r["id"]: judge(r) for r in read_csv(d / args.review_b)}
        common = [i for i in rb if rb[i] and i in judged]
        lab = lambda j: "ok" if j["decision"] == "ok" else "not_ok"  # noqa: E731
        res["n_common"] = len(common)
        res["kappa_ok"] = round(kappa([lab(judged[i]) for i in common], [lab(rb[i]) for i in common]), 4) if common else None
        res["kappa_span_error"] = round(kappa([str(judged[i]["spans_wrong"] > 0) for i in common],
                                              [str(rb[i]["spans_wrong"] > 0) for i in common]), 4) if common else None
    lo = res["kept"]["span_error"][1]
    res["decision"] = ("BỎ vss_cs (cận dưới CI95 lỗi span > 10%)" if lo > 0.10 else
                       "GIỮ vss_cs có điều kiện (cận dưới CI95 lỗi span <= 10%)") if res["kept"]["spans"] else "chưa đủ dữ liệu"
    if len(judged) < len(key):
        res["decision"] += f" — CHƯA NGHE HẾT ({len(judged)} / {len(key)} câu)"
    (d / "score.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    pct = lambda t: f"{100 * t[0]:.1f}% [{100 * t[1]:.1f}, {100 * t[2]:.1f}]"  # noqa: E731
    k = res["kept"]
    md = [f"# Chấm phiếu nghe nhãn giả vss_cs ({args.review})", "",
          f"Đã nghe {len(judged)} / {len(key)} câu; audio hỏng (drop) {drops}.", "",
          f"**Quyết định (luật đăng ký trước):** {res['decision']}", "",
          "| Câu giữ | Giá trị (CI95) |", "| --- | --- |",
          f"| tỉ lệ span sai ({k['spans']} span) | {pct(k['span_error'])} |",
          f"| câu có >= 1 span sai | {pct(k['sent_with_span_error'])} |",
          f"| câu có từ Anh không tag / viết kiểu Việt | {pct(k['sent_untagged_english'])} |",
          f"| WER phần Việt của nhãn | {k['vi_wer']} |"]
    if "rejected_false_reject" in res:
        fr = res["rejected_false_reject"]
        md.append(f"| (đối chứng) câu bị loại mà thật ra đúng | {pct(fr[:3])} (n = {fr[3]}) |")
    if args.review_b:
        md.append(f"| kappa ok / không ok ({res['n_common']} câu chung) | {res['kappa_ok']} |")
    md += ["", "| Ô | n | span sai | câu có span sai | WER Việt |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {c} | {v['n']} | {pct(v['span_error'])} | {pct(v['sent_with_span_error'])} | {v['vi_wer']} |"
           for c, v in res["cells"].items()]
    (d / "score.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--kept", required=True, help="release / manifest câu GIỮ (có trường agreement.vi_wer)")
    s.add_argument("--candidates", help="manifest ứng viên trước lọc (lấy câu bị loại làm đối chứng)")
    s.add_argument("--hyps", nargs="+", default=["results/asr_hyps/qwen3_asr_0.6b", "results/asr_hyps/fw_large_v3_int8"])
    s.add_argument("--hash-index", default="results/asr_hyps/vss_audio_sha256.tsv")
    s.add_argument("--audio-root", default="data/dl/VietSuperSpeech")
    s.add_argument("--catalog", default="data/raw/_proxy/catalog_proxy.tsv",
                   help="catalog chuẩn hoá giống lúc lọc đồng thuận (chỉ để tính lại luật cho đối chứng)")
    s.add_argument("--per-cell", type=int, default=20)
    s.add_argument("--freq-high", type=int, default=5, help="thuật ngữ hiếm nhất có trong >= N câu của pool = tần suất cao")
    s.add_argument("--controls", type=int, default=60)
    s.add_argument("--overlap", type=int, default=60, help="số câu chung cho người nghe thứ hai")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    c = sub.add_parser("score")
    c.add_argument("--dir", required=True)
    c.add_argument("--review", default="listen_A_reviewed.csv")
    c.add_argument("--review-b")
    args = ap.parse_args()
    (cmd_sample if args.cmd == "sample" else cmd_score)(args)


if __name__ == "__main__":
    main()
