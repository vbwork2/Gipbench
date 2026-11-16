#!/usr/bin/env python3
"""Xếp hạng câu bench cần người duyệt trước: câu mà nhiều hệ ASR khác nhau đều KHÔNG ra thuật ngữ <en> của ref
(ref có thể sai, vd ghi "postcard" trong khi mọi hệ nghe "podcast") hoặc có cờ nghi ngờ của auto_tag.

  python tools/review_priority.py --bench data/bench_v0_draft/bench_v0_draft.jsonl \
      --catalog data/bench_v0_draft/catalog_v0_draft.tsv --splits B1 B2 --top 300 \
      --hyps gipformer=results/v0_draft/e0_baseline/hyps.jsonl whisper=results/asr_hyps/<hệ>/<bench>.jsonl \
             qwen3=results/asr_hyps/<hệ>/<bench>.jsonl \
      --out data/review/b1b2_priority.csv
  # ra thêm <out>.report.md; mở CSV bằng tools/review_bench.html như phiếu duyệt thường

Mỗi hyps = JSONL {"id","hyp"} của MỘT hệ ASR (NAME=PATH, lặp lại nhiều lần). Hệ nào thiếu hyp của một câu thì bị bỏ khỏi phép
đếm của câu đó. Cần >= 2 hệ ĐỘC LẬP (khác họ model); nhiều biến thể của Gipformer (hotwords, FT) có chung lỗi nên không thay được.

Điểm ưu tiên của câu = 10 * S_miss + 4 * S_dongthuan + điểm cờ
  S_miss       = max theo span của (số hệ không ra thuật ngữ / số hệ có hyp) * trọng số theo số hệ (1 hệ: 0.5, 2 hệ: 0.85, >= 3: 1),
                 cộng 0.25 cho mỗi thuật ngữ khác thứ hai trở đi mà MỌI hệ đều bỏ (tối đa +0.5)
  S_dongthuan  = max theo span của (số hệ cùng ghi ra MỘT dạng khác ref / số hệ), chỉ tính khi >= 2 hệ cùng dạng (dấu hiệu ref sai)
  điểm cờ      = suspect 2, ambiguous_joined 2, abbr_reading 1.5, vi_word 1.5 cho mỗi cờ (trần 4), từ auto_tag.flags
Mức: P1 >= 12, P2 >= 8, P3 >= 4, P4 >= 1, P5 còn lại.
"Có thuật ngữ" = n-gram token của hyp (sau normalize_hyp strict) trùng thuật ngữ, hoặc ghép các token liền nhau bỏ dấu cách thì trùng
(vd hyp "pod cast" ~ ref "podcast").

CSV dùng cùng cột với phiếu duyệt hiện có (id, split, audio, duration, text_tagged, text_tagged_orig, confidence, flags, decision,
text_tagged_fixed, viet_levels, reading_note, region, gender, note) nên build_bench_v0.py --reviews đọc được sau khi duyệt; thêm cột
priority, level, reason và hint (dòng đầu = lý do ưu tiên, các dòng sau = hyp của từng hệ, ngăn bằng " || " như review_bench.html đọc).
"""
import argparse
import csv
import difflib
import sys
from collections import Counter
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.manifest import filter_splits, read_jsonl  # noqa: E402
from lib.textnorm import Catalog, normalize_hyp, normalize_ref  # noqa: E402

FLAG_POINTS = {"suspect": 2.0, "ambiguous_joined": 2.0, "abbr_reading": 1.5, "vi_word": 1.5}
FLAG_CAP = 4.0
SYS_WEIGHT = {1: 0.5, 2: 0.85}
LEVELS = [(12.0, "P1"), (8.0, "P2"), (4.0, "P3"), (1.0, "P4"), (-1e9, "P5")]
HEADER = ["id", "split", "audio", "duration", "text_tagged", "text_tagged_orig", "confidence", "flags", "priority", "level",
          "reason", "hint", "decision", "text_tagged_fixed", "viet_levels", "reading_note", "region", "gender", "note"]


def level_of(score):
    return next(name for th, name in LEVELS if score >= th)


def has_term(htoks, term):
    """Hyp có thuật ngữ nếu một n-gram token của hyp bằng nó, hoặc ghép liền các token kề nhau (bỏ dấu cách) thì trùng."""
    target = "".join(term)
    if not target:
        return True
    n = len(htoks)
    for i in range(n):
        acc = ""
        for j in range(i, min(n, i + len(term) + 3)):
            acc += htoks[j]
            if acc == target:
                return True
            if len(acc) >= len(target):
                break
    return False


def span_form(ref, a, b, htoks):
    """Đoạn hyp ứng với ref[a:b] (căn bằng difflib). Trả về danh sách token (rỗng = hyp bỏ mất đoạn đó)."""
    lo, hi = None, None

    def upd(x, y):
        nonlocal lo, hi
        lo = x if lo is None else min(lo, x)
        hi = y if hi is None else max(hi, y)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ref, htoks, autojunk=False).get_opcodes():
        if tag == "equal":
            for p in range(max(i1, a), min(i2, b)):
                upd(j1 + p - i1, j1 + p - i1 + 1)
        elif i2 > a and i1 < b and i1 != i2:   # replace / delete chồng lên span
            upd(j1, j2)
        elif i1 == i2 and a < i1 < b:          # insert nằm giữa span
            upd(j1, j2)
    return htoks[lo:hi] if lo is not None else []


def flag_points(flags):
    pts, names = 0.0, []
    for f in flags or []:
        k = str(f).split(":")[0]
        if k in FLAG_POINTS:
            pts += FLAG_POINTS[k]
            names.append(str(f))
    return min(pts, FLAG_CAP), names


def score_row(r, hyps, cat):
    """hyps = {tên hệ: hyp (str)} chỉ gồm hệ CÓ hyp của câu này. Trả về dict điểm + chi tiết từng span."""
    rt, spans = normalize_ref(r["text_tagged"], cat)
    ref = [t.tok for t in rt]
    pos = {}
    for i, t in enumerate(rt):
        if t.span >= 0:
            lo, hi = pos.get(t.span, (i, i + 1))
            pos[t.span] = (min(lo, i), max(hi, i + 1))
    htok = {n: normalize_hyp(h, cat) for n, h in hyps.items()}
    n_sys = len(htok)
    w = SYS_WEIGHT.get(n_sys, 1.0) if n_sys else 0.0
    det, s_miss, s_cons, all_miss = [], 0.0, 0.0, set()
    for si, term in enumerate(spans):
        miss = [n for n, ht in htok.items() if not has_term(ht, term)]
        forms = {}
        for n in miss:
            f = span_form(ref, *pos[si], htok[n])
            if f:
                forms.setdefault("".join(f), []).append((n, " ".join(f)))
        best = max(forms.values(), key=len) if forms else []
        d = {"term": " ".join(term), "n_sys": n_sys, "miss": miss, "form": best[0][1] if len(best) >= 2 else "",
             "form_n": len(best) if len(best) >= 2 else 0}
        det.append(d)
        if n_sys:
            m = len(miss) / n_sys
            s_miss = max(s_miss, m * w)
            if len(miss) == n_sys:
                all_miss.add(d["term"])
            if d["form_n"]:
                s_cons = max(s_cons, d["form_n"] / n_sys * w)
    if len(all_miss) > 1:  # nhiều thuật ngữ khác nhau cùng bị mọi hệ bỏ
        s_miss += 0.25 * min(2, len(all_miss) - 1)
    fp, fnames = flag_points((r.get("auto_tag") or {}).get("flags"))
    score = 10 * s_miss + 4 * s_cons + fp
    return {"score": round(score, 2), "level": level_of(score), "spans": det, "n_sys": n_sys, "flags": fnames}


def reason_of(res):
    parts = []
    if not res["n_sys"]:
        parts.append("không có hyp của hệ nào")
    seen = set()
    for d in res["spans"]:
        if d["miss"] and d["term"] not in seen:
            seen.add(d["term"])
            s = f"{len(d['miss'])}/{d['n_sys']} hệ không ra '{d['term']}'"
            if d["form_n"]:
                s += f" ({d['form_n']} hệ cùng ghi '{d['form']}')"
            parts.append(s)
    if res["flags"]:
        parts.append("cờ " + " ".join(res["flags"]))
    return "; ".join(parts) or "mọi hệ đều ra đủ thuật ngữ"


def rank(rows, hyps_by_sys, cat):
    """-> danh sách (dòng, kết quả, {tên hệ: hyp}) xếp theo điểm giảm dần (hoà: id)."""
    out = []
    for r in rows:
        if "<en" not in (r.get("text_tagged") or ""):
            continue
        hy = {n: h[r["id"]] for n, h in hyps_by_sys.items() if r["id"] in h}
        out.append((r, score_row(r, hy, cat), hy))
    out.sort(key=lambda x: (-x[1]["score"], x[0]["id"]))
    return out


def load_hyps(specs):
    out = {}
    for sp in specs:
        name, sep, path = sp.partition("=")
        if not sep or not name or not path:
            sys.exit(f"--hyps cần dạng NAME=PATH, nhận được: {sp!r}")
        out[name] = {r["id"]: (r.get("hyp") or "") for r in read_jsonl(path)}
    return out


def write_csv(path, ranked):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=HEADER, lineterminator="\r\n")
        w.writeheader()
        for r, res, hy in ranked:
            at = r.get("auto_tag") or {}
            reason = reason_of(res)
            hint = " || ".join([f"ưu tiên {res['score']:.1f} ({res['level']}): {reason}"]
                               + [f"{n}: {h}" for n, h in hy.items()])
            w.writerow({"id": r["id"], "split": r.get("split", ""), "audio": r.get("audio", ""), "duration": r.get("duration", ""),
                        "text_tagged": r["text_tagged"], "text_tagged_orig": r.get("text_tagged_orig", ""),
                        "confidence": at.get("confidence", ""), "flags": " ".join(at.get("flags") or []),
                        "priority": f"{res['score']:.2f}", "level": res["level"], "reason": reason, "hint": hint,
                        "region": r.get("region", ""), "gender": r.get("gender", "")})


def write_report(path, ranked, hyps_by_sys, args, shown):
    lv = Counter(res["level"] for _, res, _ in ranked)
    min_n = min(2, len(hyps_by_sys))
    allmiss, occ = Counter(), Counter()
    for _, res, _ in ranked:
        for d in res["spans"]:
            occ[d["term"]] += 1
            if d["n_sys"] >= max(min_n, 1) and len(d["miss"]) == d["n_sys"]:
                allmiss[d["term"]] += 1
    L = ["# Ưu tiên duyệt bench", "", f"- bench: `{args.bench}`, split: {', '.join(args.splits)}; {len(ranked)} câu có span `<en>`"
         + (f", xuất {shown} câu đầu (--top {args.top})" if shown < len(ranked) else ""), "", "## Hệ ASR", ""]
    for n, h in hyps_by_sys.items():
        L.append(f"- {n}: {sum(1 for r, _, _ in ranked if r['id'] in h)}/{len(ranked)} câu có hyp")
    if len(hyps_by_sys) < 2:
        L += ["", "CẢNH BÁO: dưới 2 hệ ASR nên điểm 'mọi hệ đều bỏ' rất yếu; chạy thêm ASR khác họ trên bench trước khi duyệt thật."]
    L += ["", "## Số câu theo mức ưu tiên", "", "| mức | câu |", "| --- | ---: |"]
    L += [f"| {n} | {lv.get(n, 0)} |" for _, n in LEVELS]
    L += ["", f"## 30 thuật ngữ bị mọi hệ bỏ nhiều nhất (span có >= {max(min_n, 1)} hệ có hyp)", "",
          "| thuật ngữ | số span bị mọi hệ bỏ | số span |", "| --- | ---: | ---: |"]
    L += [f"| {t} | {c} | {occ[t]} |" for t, c in sorted(allmiss.items(), key=lambda x: (-x[1], x[0]))[:30]]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def run(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--hyps", nargs="+", required=True, metavar="NAME=PATH", help="hyps JSONL {id, hyp} của từng hệ ASR")
    ap.add_argument("--splits", nargs="+", default=["B1", "B2"])
    ap.add_argument("--top", type=int, default=0, help="chỉ xuất N câu điểm cao nhất (0 = tất cả)")
    ap.add_argument("--out", required=True, help="CSV ra; report ghi cạnh: <out>.report.md")
    args = ap.parse_args(argv)
    cat = Catalog.load(args.catalog)
    rows = filter_splits(read_jsonl(args.bench), args.splits)
    hyps = load_hyps(args.hyps)
    ranked = rank(rows, hyps, cat)
    shown = min(args.top, len(ranked)) if args.top > 0 else len(ranked)
    write_csv(args.out, ranked[:shown])
    write_report(str(args.out) + ".report.md", ranked, hyps, args, shown)
    print(f"{len(ranked)} câu có <en> trong {args.splits}; xuất {shown} câu -> {args.out}")
    print(", ".join(f"{n}: {c}" for n, c in sorted(Counter(res['level'] for _, res, _ in ranked).items())))
    return 0


if __name__ == "__main__":
    sys.exit(run())
