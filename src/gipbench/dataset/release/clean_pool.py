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
Từ v3 (18/11/2026), mặc định thêm:
  * audio: đọc mọi file pool một lần (core.decontam.audio_check), ghi `audio_sha256` vào manifest ra (cổng chế độ
    metadata kiểm được trùng audio, khỏi băm lại), bỏ câu audio RỖNG / IM LẶNG SỐ (dữ liệu hỏng, không lan nguồn). Thiếu
    file thì dừng (mã 1). Bench thiếu audio_sha256 được băm từ file nếu đọc được; audio im lặng của bench không đưa vào
    chỉ mục (im lặng cùng độ dài trùng sha dù khác video). Trùng sha audio (critical) lan ra cả nguồn: bắt video đăng lại
    dưới YouTube ID khác.
  * text gần trùng (Jaccard shingle >= --near-threshold, mặc định 0.6, khác luật bộ đọc như trùng text): bỏ câu; nguồn
    có >= --near-spread-min câu gần trùng (mặc định 2) thì bỏ cả nguồn (video đăng lại, vd ViMedCSS 2337_2 / 2338_2).
  * dấu vân tay âm thanh (core.audiofp qua acoustic_audit; tắt bằng --no-acoustic): khớp BER <= 0.10 (cùng audio đã
    resample / cắt khác, vd Bud500 test và VietSuperSpeech cắt từ cùng một audio) -> bỏ cả nguồn; BER cao hơn (chỉ chung
    một đoạn, vd câu mở đầu theo mẫu đọc bằng cùng giọng của kênh) -> bỏ câu. Cache ở data/tmp/audiofp/cache.npz.
Sau khi lọc, chạy lại `decontam.audit` của pool sạch với từng tập tham chiếu (audio: audio_sha256 vừa tính, không đọc
lại file) + đếm lại text gần trùng, và DỪNG (mã 1) nếu còn. Ghi <out>, <out>.dropped.jsonl (kèm `drop_reason`),
<out>.report.json, <out>.report.md (số câu / giờ / nguồn bị loại theo khoá và theo tập tham chiếu).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from gipbench.core import release as R
from gipbench.core.decontam import (READ_SCRIPT_FAMILIES, audio_check, audit, clip_key, dataset_family, dup_key,
                                    near_duplicates, row_text, source_key)
from gipbench.core.manifest import read_jsonl, write_jsonl

RAW_FIELDS = ("speaker", "source", "session")
BAD_AUDIO = ("empty", "digital_silence")
NON_SPREAD_KEYS = {"text", "text_near", "acoustic_partial"}
ACOUSTIC_SPREAD_BER = 0.10   # khớp dấu vân tay BER <= ngưỡng này: cùng audio (video đăng lại / cắt từ cùng nguồn) -> lan nguồn


def acoustic_hits(rows: List[Dict], refs: Sequence[Tuple[str, List[Dict]]], workers: int = 1, cache=None) -> Dict[int, Dict]:
    """{chỉ số dòng pool: {"acoustic" | "acoustic_partial": {tên tập}}} theo dấu vân tay âm thanh (core.audiofp).
    BER <= ACOUSTIC_SPREAD_BER -> "acoustic" (critical, lan nguồn); cao hơn (chỉ chung một đoạn, vd câu mở đầu theo mẫu
    đọc bằng cùng giọng của kênh) -> "acoustic_partial" (chỉ bỏ câu)."""
    from gipbench.dataset.release.acoustic_audit import audit_acoustic
    res = audit_acoustic([("pool", rows)], refs, workers=workers, cache=cache)
    pos: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        pos[str(r.get("id"))].append(i)
    out: Dict[int, Dict] = defaultdict(lambda: defaultdict(set))
    for m in res["matches"]:
        k = "acoustic" if m["ber"] <= ACOUSTIC_SPREAD_BER else "acoustic_partial"
        for i in pos.get(m["train_id"], ()):
            out[i][k].add(m["bench"])
    return {i: dict(v) for i, v in out.items()}


def screen_audio(rows: List[Dict], on_missing: str = "error") -> Tuple[List[Dict], List[Dict], Dict]:
    """Ghi audio_sha256 cho mọi dòng pool; trả (dòng tốt, dòng audio rỗng / im lặng, thống kê). Thiếu file: on_missing =
    "error" -> SystemExit; "keep" -> giữ dòng (không sha), đếm vào thống kê `missing`."""
    ok, bad, defects, missing = [], [], Counter(), []
    for r in rows:
        a = audio_check(r.get("_audio_path"))
        if a["defect"] == "missing":
            missing.append(r.get("id"))
            if on_missing == "keep":
                ok.append(r)
            continue
        r["audio_sha256"] = a["sha256"]
        defects[a["defect"] or "ok"] += 1
        if a["defect"] in BAD_AUDIO:
            bad.append(dict(r, drop_reason=f"audio_{a['defect']}"))
        else:
            ok.append(r)
    if missing and on_missing != "keep":
        raise SystemExit(f"LỖI {len(missing)} câu pool không có file audio (vd {missing[:3]}): sửa đường dẫn trước khi lọc")
    return ok, bad, {"audio_checked": len(rows), "audio": dict(defects.most_common()), "n_bad_audio": len(bad),
                     "missing": len(missing)}


def hash_refs(refs: Sequence[Tuple[str, List[Dict]]]) -> Dict:
    """Thêm audio_sha256 cho dòng bench thiếu sha mà đọc được file; audio rỗng / im lặng đánh dấu `_silent` (không vào chỉ
    mục sha). Trả thống kê số dòng đã băm / không đọc được / im lặng."""
    st = Counter()
    for _, rows in refs:
        for r in rows:
            if r.get("audio_sha256"):
                continue
            p = r.get("_audio_path")
            if not p or not Path(p).exists():
                st["ref_audio_unreadable"] += 1
                continue
            a = audio_check(p)
            st["ref_audio_hashed"] += 1
            if a["defect"] in BAD_AUDIO:
                r["_silent"] = True
                st["ref_audio_silent"] += 1
            else:
                r["audio_sha256"] = a["sha256"]
    return dict(st)


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
            if r.get("audio_sha256") and not r.get("_silent"):
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
        b = {n for n, f, tr in idx["text"][t] if not (rd and tr and f == fam and fam in READ_SCRIPT_FAMILIES)}
        if b:
            out["text"] = b
    return out


def _src(r: Dict):
    """Nguồn của một dòng pool: source_key, không có thì speaker thô (None nếu không có cả hai)."""
    return source_key(r) or (("spk:" + str(r["speaker"])) if r.get("speaker") else None)


def near_hits(rows: List[Dict], refs: Sequence[Tuple[str, List[Dict]]], threshold: float) -> Dict[int, set]:
    """{chỉ số dòng pool: {tên tập}} cho dòng pool có text gần trùng (không trùng hẳn) với một câu tham chiếu. Miễn như
    trùng text: cả hai cùng họ dataset và cùng là bộ đọc (kịch bản đọc dùng lại có chủ ý)."""
    pos: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        pos[str(r.get("id"))].append(i)
    out: Dict[int, set] = defaultdict(set)
    for name, ref_rows in refs:
        by_id = {str(r.get("id")): r for r in ref_rows}
        for d in near_duplicates(ref_rows, rows, threshold, best_only=False):
            t = by_id.get(str(d["test_id"]))
            for i in pos.get(str(d["train_id"]), ()):
                tr = rows[i]
                if t is not None and _is_read(t) and _is_read(tr) and dataset_family(t) in READ_SCRIPT_FAMILIES and \
                        dataset_family(t) == dataset_family(tr):
                    continue
                out[i].add(name)
    return dict(out)


def clean(rows: List[Dict], refs: Sequence[Tuple[str, List[Dict]]], spread: bool = True, near_threshold: float = 0.0,
          near_spread_min: int = 2, extra_hits: Dict[int, Dict] = None):
    """Trả (giữ, bỏ, thống kê). spread: nguồn (video / người nói) của một câu trùng NGUỒN / audio / id thì mọi câu cùng
    nguồn trong pool cũng bị bỏ. Trùng chỉ do text (câu đọc trùng kịch bản) không lan ra cả nguồn. near_threshold > 0:
    bỏ thêm câu gần trùng text; nguồn có >= near_spread_min câu gần trùng thì bỏ cả nguồn (video đăng lại).
    Câu bị bỏ có `drop_reason` (khoá trùng, hoặc `same_source`)."""
    idx = build_index(refs)
    hit = {}
    for i, r in enumerate(rows):
        h = hits_of(r, idx)
        if h:
            hit[i] = h
    if near_threshold > 0:
        for i, b in near_hits(rows, refs, near_threshold).items():
            hit.setdefault(i, {})["text_near"] = b
    for i, h in (extra_hits or {}).items():   # vd acoustic_hits
        hit.setdefault(i, {}).update(h)
    bad_src, near_src = set(), set()
    if spread:
        bad_src = {_src(rows[i]) for i, h in hit.items() if set(h) - NON_SPREAD_KEYS} - {None}
        nc = Counter(_src(rows[i]) for i, h in hit.items() if "text_near" in h)
        near_src = {s for s, c in nc.items() if s is not None and c >= near_spread_min} - bad_src
        bad_src |= near_src
    by_key, by_ref = Counter(), Counter()
    for h in hit.values():
        for k, b in h.items():
            by_key[k] += 1
        for n in set().union(*h.values()):
            by_ref[n] += 1
    keep, drop, n_spread = [], [], 0
    for i, r in enumerate(rows):
        if i in hit:
            drop.append(dict(r, drop_reason="+".join(sorted(hit[i]))))
        elif _src(r) in bad_src:
            drop.append(dict(r, drop_reason="same_source"))
            n_spread += 1
        else:
            keep.append(r)
    hours = lambda rs: round(sum(float(x.get("duration") or 0) for x in rs) / 3600, 3)  # noqa: E731
    st = {"n_in": len(rows), "h_in": hours(rows), "n_keep": len(keep), "h_keep": hours(keep), "n_drop": len(drop),
          "h_drop": hours(drop), "n_direct_hits": len(hit), "n_same_source_as_hit": n_spread,
          "sources_dropped": len(bad_src), "sources_dropped_near": len(near_src),
          "sources_kept": len({_src(r) for r in keep} - {None}),
          "speakers_kept": len({r.get("speaker") for r in keep} - {None, ""}),
          "near_threshold": near_threshold, "near_spread_min": near_spread_min,
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


def verify(keep: List[Dict], refs, near_threshold: float = 0.0) -> Dict[str, Dict]:
    """Kiểm lại pool sạch bằng chính hàm của cổng: số finding critical theo từng tập tham chiếu; near_threshold > 0 thì
    đếm thêm câu pool còn gần trùng (`text_near`, tính như critical ở đây vì clean đã phải bỏ hết)."""
    out = {}
    near = near_hits(keep, refs, near_threshold) if near_threshold > 0 else {}
    for name, rows in refs:
        rep = audit(rows, keep, check_audio_sha=True, hash_files=False, near=False, require_audio_sha=False)
        crit = [f for f in rep["findings"] if f["severity"] == "critical"]
        kinds = {f["kind"]: f["count"] for f in crit}
        n_near = sum(name in b for b in near.values())
        if n_near:
            kinds["text_near"] = n_near
        out[name] = {"critical": len(crit) + bool(n_near), "kinds": kinds}
    return out


def write_pool(out: Path, keep: List[Dict], drop: List[Dict]):
    """Ghi pool sạch + câu bị bỏ. `audio` viết lại tương đối theo thư mục file ra (từ `_audio_path`): giữ nguyên chuỗi
    tương đối của manifest nguồn thì lệch cấp khi file ra nằm ở độ sâu khác (lỗi 17/11 ở cả 3 pool v2)."""
    from gipbench.dataset.release.make_dev_split import rel_audio
    out_dir = str(Path(out).resolve().parent)
    fix = lambda rs: [dict(r, audio=rel_audio(r, out_dir)) if r.get("audio") else r for r in rs]  # noqa: E731
    write_jsonl(str(out), fix(keep))
    write_jsonl(str(out) + ".dropped.jsonl", fix(drop))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pool", help="release đầu vào, vd pool/vimedcss_train@full")
    g.add_argument("--manifest", help="manifest đầu vào (chưa đăng ký)")
    ap.add_argument("--bench", nargs="*", help="bench tham chiếu (mặc định: mọi bench khoá trong releases.yaml)")
    ap.add_argument("--exclude", nargs="*", default=[], help="manifest / release loại thêm (dev, test chưa khoá...)")
    ap.add_argument("--no-spread", action="store_true", help="chỉ bỏ câu trùng, không bỏ cả nguồn (KHÔNG khuyến nghị)")
    ap.add_argument("--near-threshold", type=float, default=0.6,
                    help="Jaccard shingle để bỏ câu gần trùng text với tham chiếu (0 = tắt; mặc định 0.6)")
    ap.add_argument("--near-spread-min", type=int, default=2,
                    help="số câu gần trùng trong một nguồn để bỏ cả nguồn (video đăng lại; mặc định 2)")
    ap.add_argument("--no-audio", action="store_true",
                    help="không đọc file audio (không sha, không lọc audio rỗng / im lặng; KHÔNG khuyến nghị)")
    ap.add_argument("--no-acoustic", action="store_true",
                    help="bỏ qua kiểm dấu vân tay âm thanh (acoustic_audit; KHÔNG khuyến nghị)")
    ap.add_argument("--workers", type=int, default=4, help="số tiến trình tính dấu vân tay")
    ap.add_argument("--fp-cache", default=None, help="cache dấu vân tay (mặc định như acoustic_audit; '' = không cache)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bench = args.bench if args.bench is not None else R.locked_bench_refs()
    rows = read_jsonl(str(R.resolve(args.pool).manifest)) if args.pool else read_jsonl(args.manifest)
    bad_audio, ast = [], {"audio_checked": 0}
    if not args.no_audio:
        rows, bad_audio, ast = screen_audio(rows)
    refs = load_refs(bench, args.exclude)
    if not args.no_audio:
        ast.update(hash_refs(refs))
    ac = {}
    if not args.no_acoustic:
        from gipbench.core.audiofp import load_cache, save_cache
        from gipbench.dataset.release.acoustic_audit import DEFAULT_CACHE
        cpath = str(DEFAULT_CACHE) if args.fp_cache is None else args.fp_cache
        cache = load_cache(cpath) if cpath else {}
        n0 = len(cache)
        ac = acoustic_hits(rows, refs, workers=args.workers, cache=cache)
        if cpath and len(cache) > n0:
            Path(cpath).parent.mkdir(parents=True, exist_ok=True)
            save_cache(cpath, cache)
    keep, drop, st = clean(rows, refs, spread=not args.no_spread, near_threshold=args.near_threshold,
                           near_spread_min=args.near_spread_min, extra_hits=ac)
    st["acoustic_checked"] = not args.no_acoustic
    ver = verify(keep, refs, args.near_threshold)
    if not args.no_acoustic:   # kiểm lại bằng dấu vân tay (cache: không tính lại)
        left = acoustic_hits(keep, refs, workers=args.workers, cache=cache)
        for h in left.values():
            for k, names in h.items():
                for n in names:
                    ver[n]["kinds"][k] = ver[n]["kinds"].get(k, 0) + 1
                    ver[n]["critical"] += 1
    hours = lambda rs: round(sum(float(x.get("duration") or 0) for x in rs) / 3600, 3)  # noqa: E731
    st.update({"input": args.pool or args.manifest, "against": [n for n, _ in refs], "spread": not args.no_spread,
               "audio_screen": ast, "n_in_total": st["n_in"] + len(bad_audio), "n_bad_audio": len(bad_audio),
               "h_bad_audio": hours(bad_audio), "verify": ver})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_pool(out, keep, bad_audio + drop)
    Path(str(out) + ".report.json").write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    md = [f"# clean_pool: `{st['input']}` -> `{out}`", "",
          f"So với: {', '.join(f'`{n}`' for n, _ in refs)}", "",
          "| | Câu | Giờ |", "| --- | --- | --- |",
          f"| vào | {st['n_in_total']} | {round(st['h_in'] + st['h_bad_audio'], 3)} |",
          f"| bỏ: audio rỗng / im lặng | {st['n_bad_audio']} | {st['h_bad_audio']} |",
          f"| bỏ: trùng / gần trùng tham chiếu | {st['n_drop']} | {st['h_drop']} |",
          f"| giữ | {st['n_keep']} | {st['h_keep']} |", "",
          f"- trùng trực tiếp {st['n_direct_hits']} câu; bỏ thêm {st['n_same_source_as_hit']} câu cùng nguồn; "
          f"{st['sources_dropped']} nguồn bị loại ({st['sources_dropped_near']} do >= {args.near_spread_min} câu gần trùng), "
          f"còn {st['sources_kept']} nguồn / {st['speakers_kept']} speaker.",
          "- câu trùng theo khoá: " + (", ".join(f"{k} {v}" for k, v in st["rows_by_key"].items()) or "không"),
          "- câu trùng theo tập tham chiếu: " + (", ".join(f"`{k}` {v}" for k, v in st["rows_by_ref"].items()) or "không"),
          "- audio: " + (", ".join(f"{k} {v}" for k, v in ast.items() if k != "audio") or "không đọc")
          + (f"; pool: {', '.join(f'{k} {v}' for k, v in ast.get('audio', {}).items())}" if ast.get("audio") else ""),
          "", f"## Kiểm lại bằng decontam.audit (audio: audio_sha256 đã tính; text gần trùng >= {args.near_threshold})", "",
          "| Tập | Critical | Loại |", "| --- | --- | --- |"]
    md += [f"| `{n}` | {v['critical']} | {', '.join(f'{k} {c}' for k, c in v['kinds'].items()) or '–'} |" for n, v in ver.items()]
    Path(str(out) + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[4:15]))
    bad = {n: v for n, v in ver.items() if v["critical"]}
    if bad:
        print(f"LỖI còn critical sau khi lọc: {bad}")
        sys.exit(1)
    print(f"0 critical với {len(refs)} tập tham chiếu -> {out}")


if __name__ == "__main__":
    main()
