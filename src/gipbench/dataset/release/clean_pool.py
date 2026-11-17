#!/usr/bin/env python3
"""Lọc một pool train cho sạch với MỌI bench khoá (và tập loại thêm), rồi tự kiểm lại bằng cổng (TODO 2c, IP-06).

  python -m gipbench.dataset.release.clean_pool --pool pool/vimedcss_train@full \
      --out data/pools/vimedcss_train_v2/manifest.jsonl
  python -m gipbench.dataset.release.clean_pool --manifest data/raw/cv/pool_cv.jsonl --out data/pools/cv_other_v2/manifest.jsonl
  # rồi thêm mục release (configs/dataset/releases.yaml) và: python -m gipbench.dataset.release.register pool/<tên>@v2

Loại câu train khi trùng với BẤT KỲ câu nào của bench khoá (flags.locked trong releases.yaml; --bench ghi đè) hoặc của
--exclude (dev, bộ test chưa khoá...) theo đúng các khoá mà cổng `core.leakgate` coi là critical (`core.decontam.audit`):
  id, speaker / source / session (chuỗi thô, không rỗng), source_key (video YouTube, người nói CV / VIVOS), clip_key
  (clip gốc CV / VIVOS), audio_sha256 (khi cả hai phía có), text sau chuẩn hoá (dup_key; trừ trùng trong cùng bộ đọc
  có chủ ý: cùng họ dataset và cả hai domain = read, như audit).
Loại theo NGUỒN (cả video / người nói), không theo câu: một video có trong bench thì mọi câu của video đó bị loại.
Sau khi lọc, chạy lại `decontam.audit` của pool sạch với từng tập tham chiếu (audio: chỉ audio_sha256 có sẵn, không đọc
file) và DỪNG (mã 1) nếu còn critical. Ghi <out>, <out>.dropped.jsonl, <out>.report.json, <out>.report.md (số câu / giờ /
nguồn bị loại theo khoá và theo tập tham chiếu).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from gipbench.core import release as R
from gipbench.core.decontam import audit, clip_key, dataset_family, dup_key, row_text, source_key
from gipbench.core.manifest import read_jsonl, write_jsonl

RAW_FIELDS = ("speaker", "source", "session")


def _is_read(r: Dict) -> bool:
    return str(r.get("domain") or "") == "read"


def build_index(refs: Sequence[Tuple[str, List[Dict]]]) -> Dict[str, Dict]:
    """{khoá: {giá trị: {tên tập}}}; text lưu kèm (họ dataset, có phải bộ đọc) để áp luật 'cùng bộ đọc'."""
    idx: Dict[str, Dict] = defaultdict(lambda: defaultdict(set))
    for name, rows in refs:
        for r in rows:
            if r.get("id"):
                idx["id"][str(r["id"])].add(name)
            for f in RAW_FIELDS:
                v = str(r.get(f) or "")
                if v:
                    idx[f][v].add(name)
            for k, fn in (("source_norm", source_key), ("clip", clip_key)):
                v = fn(r)
                if v:
                    idx[k][v].add(name)
            if r.get("audio_sha256"):
                idx["audio_sha256"][r["audio_sha256"]].add(name)
            t = dup_key(row_text(r))
            if t:
                idx["text"][t].add((name, dataset_family(r), _is_read(r)))
    return idx


def hits_of(r: Dict, idx) -> Dict[str, set]:
    """Các khoá của một dòng train khớp với tập tham chiếu: {khoá: {tên tập}}."""
    out = {}
    if r.get("id") and str(r["id"]) in idx["id"]:
        out["id"] = idx["id"][str(r["id"])]
    for f in RAW_FIELDS:
        v = str(r.get(f) or "")
        if v and v in idx[f]:
            out[f] = idx[f][v]
    for k, fn in (("source_norm", source_key), ("clip", clip_key)):
        v = fn(r)
        if v and v in idx[k]:
            out[k] = idx[k][v]
    if r.get("audio_sha256") and r["audio_sha256"] in idx["audio_sha256"]:
        out["audio_sha256"] = idx["audio_sha256"][r["audio_sha256"]]
    t = dup_key(row_text(r))
    if t and t in idx["text"]:
        fam, rd = dataset_family(r), _is_read(r)
        b = {n for n, f, tr in idx["text"][t] if not (rd and tr and f and f == fam)}
        if b:
            out["text"] = b
    return out


def _src(r: Dict):
    """Nguồn của một dòng pool: source_key, không có thì speaker thô (None nếu không có cả hai)."""
    return source_key(r) or (("spk:" + str(r["speaker"])) if r.get("speaker") else None)


def clean(rows: List[Dict], refs: Sequence[Tuple[str, List[Dict]]], spread: bool = True):
    """Trả (giữ, bỏ, thống kê). spread: nguồn (video / người nói) của một câu trùng NGUỒN thì mọi câu cùng nguồn trong
    pool cũng bị bỏ. Trùng chỉ do text (câu đọc trùng kịch bản) không lan ra cả nguồn."""
    idx = build_index(refs)
    hit = {}
    for i, r in enumerate(rows):
        h = hits_of(r, idx)
        if h:
            hit[i] = h
    bad_src = ({_src(rows[i]) for i, h in hit.items() if set(h) - {"text"}} - {None}) if spread else set()
    by_key, by_ref = Counter(), Counter()
    for h in hit.values():
        for k, b in h.items():
            by_key[k] += 1
        for n in set().union(*h.values()):
            by_ref[n] += 1
    keep, drop, n_spread = [], [], 0
    for i, r in enumerate(rows):
        if i in hit:
            drop.append(r)
        elif _src(r) in bad_src:
            drop.append(r)
            n_spread += 1
        else:
            keep.append(r)
    hours = lambda rs: round(sum(float(x.get("duration") or 0) for x in rs) / 3600, 3)  # noqa: E731
    st = {"n_in": len(rows), "h_in": hours(rows), "n_keep": len(keep), "h_keep": hours(keep), "n_drop": len(drop),
          "h_drop": hours(drop), "n_direct_hits": len(hit), "n_same_source_as_hit": n_spread,
          "sources_dropped": len(bad_src), "sources_kept": len({_src(r) for r in keep} - {None}),
          "speakers_kept": len({r.get("speaker") for r in keep} - {None, ""}),
          "rows_by_key": dict(by_key.most_common()), "rows_by_ref": dict(by_ref.most_common())}
    return keep, drop, st


def _read_ref(p: str) -> List[Dict]:
    if not Path(p).exists() and "/" in p and "@" in p:
        return read_jsonl(str(R.resolve(p).manifest))
    return read_jsonl(p)


def load_refs(bench: Sequence[str], exclude: Sequence[str]) -> List[Tuple[str, List[Dict]]]:
    refs = []
    for ref in bench:
        rl = R.resolve(ref)
        if rl.manifest is None:
            raise SystemExit(f"LỖI {ref} không có manifest")
        refs.append((ref, read_jsonl(str(rl.manifest))))
    return refs + [(p, _read_ref(p)) for p in exclude]


def verify(keep: List[Dict], refs) -> Dict[str, Dict]:
    """Kiểm lại pool sạch bằng chính hàm của cổng: số finding critical theo từng tập tham chiếu."""
    out = {}
    for name, rows in refs:
        rep = audit(rows, keep, check_audio_sha=True, hash_files=False, near=False, require_audio_sha=False)
        crit = [f for f in rep["findings"] if f["severity"] == "critical"]
        out[name] = {"critical": len(crit), "kinds": {f["kind"]: f["count"] for f in crit}}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pool", help="release đầu vào, vd pool/vimedcss_train@full")
    g.add_argument("--manifest", help="manifest đầu vào (chưa đăng ký)")
    ap.add_argument("--bench", nargs="*", help="bench tham chiếu (mặc định: mọi bench khoá trong releases.yaml)")
    ap.add_argument("--exclude", nargs="*", default=[], help="manifest / release loại thêm (dev, test chưa khoá...)")
    ap.add_argument("--no-spread", action="store_true", help="chỉ bỏ câu trùng, không bỏ cả nguồn (KHÔNG khuyến nghị)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bench = args.bench if args.bench is not None else R.locked_bench_refs()
    rows = read_jsonl(str(R.resolve(args.pool).manifest)) if args.pool else read_jsonl(args.manifest)
    refs = load_refs(bench, args.exclude)
    keep, drop, st = clean(rows, refs, spread=not args.no_spread)
    ver = verify(keep, refs)
    st.update({"input": args.pool or args.manifest, "against": [n for n, _ in refs], "spread": not args.no_spread,
               "verify": ver})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(str(out), keep)
    write_jsonl(str(out) + ".dropped.jsonl", drop)
    Path(str(out) + ".report.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# clean_pool: `{st['input']}` -> `{out}`", "",
          f"So với: {', '.join(f'`{n}`' for n, _ in refs)}", "",
          "| | Câu | Giờ |", "| --- | --- | --- |",
          f"| vào | {st['n_in']} | {st['h_in']} |", f"| giữ | {st['n_keep']} | {st['h_keep']} |",
          f"| bỏ | {st['n_drop']} | {st['h_drop']} |", "",
          f"- trùng trực tiếp {st['n_direct_hits']} câu; bỏ thêm {st['n_same_source_as_hit']} câu cùng nguồn; "
          f"{st['sources_dropped']} nguồn bị loại, còn {st['sources_kept']} nguồn / {st['speakers_kept']} speaker.",
          "- câu trùng theo khoá: " + (", ".join(f"{k} {v}" for k, v in st["rows_by_key"].items()) or "không"),
          "- câu trùng theo tập tham chiếu: " + (", ".join(f"`{k}` {v}" for k, v in st["rows_by_ref"].items()) or "không"),
          "", "## Kiểm lại bằng decontam.audit (audio: chỉ audio_sha256 có sẵn)", "",
          "| Tập | Critical | Loại |", "| --- | --- | --- |"]
    md += [f"| `{n}` | {v['critical']} | {', '.join(f'{k} {c}' for k, c in v['kinds'].items()) or '–'} |" for n, v in ver.items()]
    Path(str(out) + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[4:13]))
    bad = {n: v for n, v in ver.items() if v["critical"]}
    if bad:
        print(f"LỖI còn critical sau khi lọc: {bad}")
        sys.exit(1)
    print(f"0 critical với {len(refs)} tập tham chiếu -> {out}")


if __name__ == "__main__":
    main()
