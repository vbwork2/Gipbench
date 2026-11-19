#!/usr/bin/env python3
"""Kiểm trùng ÂM THANH train -> bench khoá bằng dấu vân tay (core.audiofp), bổ sung cho cổng rò rỉ (core.leakgate).

  python -m gipbench.dataset.release.acoustic_audit --train pool/vimedcss_train@v3 pool/vivos_train@v3 ... \
      --out reports/leak/acoustic_v3 [--workers 4] [--fail-on-match]

Bắt được thứ sha256 / source_key / text không thấy: cùng lời nói đã resample, nén lại, đổi âm lượng, hoặc cắt ranh giới
khác (video đăng lại dưới YouTube ID khác; G-Group không có id nguồn). Không bắt được: cùng người nói / cùng video nhưng
đoạn khác (đã kiểm bằng source_key), hoặc audio bị đổi tốc độ / cao độ.
Bench mặc định: mọi bench khoá (flags.locked trong releases.yaml). Dòng không đọc được audio (vd E0 v1.0 trỏ đường dẫn
cũ) được đếm và bỏ qua: cùng audio thường có ở bản v1.1.
Ghi <out>/acoustic_report.json (+ .md): mọi cặp khớp {bench, test_id, train_ref, train_id, ber, overlap_s, offset_s}
và `train_ids` (dùng cho clean_pool --exclude-ids). Dấu vân tay cache ở --cache (mặc định data/tmp/audiofp/cache.npz).
Thoát 1 nếu --fail-on-match và có cặp khớp.
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from gipbench.core import release as R
from gipbench.core.audiofp import FPIndex, fingerprint_rows, load_cache, save_cache
from gipbench.core.manifest import read_jsonl
from gipbench.core.paths import REPO

DEFAULT_CACHE = REPO / "data" / "tmp" / "audiofp" / "cache.npz"


def _rows(ref: str) -> List[Dict]:
    if Path(ref).exists():
        return read_jsonl(ref)
    rl = R.resolve(ref)
    if rl.manifest is None:
        raise SystemExit(f"LỖI {ref} không có manifest")
    return read_jsonl(str(rl.manifest))


def audit_acoustic(train: Sequence[Tuple[str, List[Dict]]], bench: Sequence[Tuple[str, List[Dict]]], *, workers: int = 1,
                   cache: Dict = None, max_ber: float = 0.30, min_overlap_s: float = 1.0, log_every: int = 0) -> Dict:
    """Trả {"matches": [...], "train_ids": {ref: [id]}, "stats": {...}}."""
    cache = {} if cache is None else cache
    t0 = time.time()
    tr_rows = [(ref, r) for ref, rows in train for r in rows]
    fingerprint_rows([r for _, r in tr_rows], log_every=log_every, workers=workers, cache=cache)
    key_of = {}
    items = []
    for ref, r in tr_rows:
        p = str(r.get("_audio_path"))
        if p in cache:
            k = f"{ref}\t{r.get('id')}"
            key_of[k] = (ref, str(r.get("id")))
            items.append((k, *cache[p]))
    idx = FPIndex(items)
    t1 = time.time()
    b_rows = [(ref, r) for ref, rows in bench for r in rows if r.get("_audio_path") and Path(r["_audio_path"]).exists()]
    unreadable = sum(len(rows) for _, rows in bench) - len(b_rows)
    fingerprint_rows([r for _, r in b_rows], log_every=log_every, workers=workers, cache=cache)
    t2 = time.time()
    by_path: Dict[str, List[Dict]] = {}
    matches = []
    for ref, r in b_rows:
        p = str(r["_audio_path"])
        if p not in cache:
            continue
        if p not in by_path:
            by_path[p] = idx.match(*cache[p], max_ber=max_ber, min_overlap_s=min_overlap_s)
        for m in by_path[p]:
            tref, tid = key_of[m["train"]]
            matches.append({"bench": ref, "test_id": str(r.get("id")), "train_ref": tref, "train_id": tid,
                            "ber": m["ber"], "overlap_s": m["overlap_s"], "offset_s": m["offset_s"]})
    train_ids = defaultdict(set)
    for m in matches:
        train_ids[m["train_ref"]].add(m["train_id"])
    st = {"n_train_rows": len(tr_rows), "n_train_fp": len(items), "n_bench_rows": sum(len(x) for _, x in bench),
          "n_bench_readable": len(b_rows), "n_bench_unreadable": unreadable, "n_bench_unique_audio": len(by_path),
          "n_matches": len(matches), "pairs_by_bench_train": {f"{b} <- {t}": c for (b, t), c in
                                                              Counter((m["bench"], m["train_ref"]) for m in matches).most_common()},
          "max_ber": max_ber, "min_overlap_s": min_overlap_s,
          "sec_train_fp": round(t1 - t0, 1), "sec_bench_fp": round(t2 - t1, 1), "sec_match": round(time.time() - t2, 1)}
    return {"matches": matches, "train_ids": {k: sorted(v) for k, v in train_ids.items()}, "stats": st}


def render_md(res: Dict, train_refs: Sequence[str], bench_refs: Sequence[str]) -> str:
    st = res["stats"]
    md = ["# Kiểm trùng âm thanh (dấu vân tay Haitsma–Kalker)", "",
          f"Train: {', '.join(f'`{r}`' for r in train_refs)}", "", f"Bench: {', '.join(f'`{r}`' for r in bench_refs)}", "",
          f"- train {st['n_train_fp']} / {st['n_train_rows']} câu có dấu vân tay; bench {st['n_bench_readable']} câu đọc được "
          f"({st['n_bench_unique_audio']} audio khác nhau), {st['n_bench_unreadable']} câu không đọc được audio (bỏ qua).",
          f"- ngưỡng: BER <= {st['max_ber']}, chồng >= {st['min_overlap_s']} s (clip ngắn: >= 90% độ dài).",
          f"- **{st['n_matches']} cặp khớp**; câu train cần loại: "
          + (", ".join(f"`{k}` {len(v)}" for k, v in res["train_ids"].items()) or "không"), ""]
    if res["matches"]:
        md += ["| Bench | Câu test | Train | Câu train | BER | Chồng (s) |", "| --- | --- | --- | --- | --- | --- |"]
        md += [f"| `{m['bench']}` | `{m['test_id']}` | `{m['train_ref']}` | `{m['train_id']}` | {m['ber']} | {m['overlap_s']} |"
               for m in sorted(res["matches"], key=lambda m: (m["bench"], m["test_id"]))[:200]]
    return "\n".join(md) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True, help="release / manifest train (pool, dev, ...)")
    ap.add_argument("--bench", nargs="*", help="bench tham chiếu (mặc định: mọi bench khoá)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="file cache dấu vân tay ('' = không cache)")
    ap.add_argument("--max-ber", type=float, default=0.30)
    ap.add_argument("--min-overlap", type=float, default=1.0)
    ap.add_argument("--fail-on-match", action="store_true")
    args = ap.parse_args()
    bench_refs = args.bench if args.bench is not None else R.locked_bench_refs()
    cache = load_cache(args.cache) if args.cache else {}
    n0 = len(cache)
    res = audit_acoustic([(r, _rows(r)) for r in args.train], [(r, _rows(r)) for r in bench_refs], workers=args.workers,
                         cache=cache, max_ber=args.max_ber, min_overlap_s=args.min_overlap, log_every=2000)
    if args.cache and len(cache) > n0:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        save_cache(args.cache, cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "acoustic_report.json").write_text(json.dumps(dict(res, train=args.train, bench=bench_refs), ensure_ascii=False,
                                                         indent=1), encoding="utf-8")
    md = render_md(res, args.train, bench_refs)
    (out / "acoustic_report.md").write_text(md, encoding="utf-8")
    print(md[:3000])
    print(json.dumps(res["stats"], ensure_ascii=False))
    if args.fail_on_match and res["matches"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
