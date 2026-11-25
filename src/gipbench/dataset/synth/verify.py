#!/usr/bin/env python3
"""Bước 4 của chuỗi TTS: lọc audio TTS hỏng (đọc sai, nuốt từ, kéo dài) -> train_tts.jsonl.

  # 1) hyps của một ASR mạnh trên audio TTS (công cụ này KHÔNG tự chạy ASR):
  python -m gipbench.infer.asr_hyps --engine faster-whisper --model large-v3 --manifest data/tts/raw/train_tts_raw.jsonl \
      --out results/fw_large_v3/tts/hyps.jsonl
  # 2) lọc
  python -m gipbench.dataset.synth.verify --manifest data/tts/raw/train_tts_raw.jsonl --hyps fw=results/fw_large_v3/tts/hyps.jsonl \
      --catalog data/bench_v0_draft/catalog_v0_draft.tsv --lexicon configs/dataset/synth/tts_lexicon.tsv --out data/tts/train_tts.jsonl

Giữ một câu khi, ở ít nhất --min-agree hệ ASR (--hyps NAME=PATH, lặp được; PATH là file hyps.jsonl {id, hyp} hoặc thư mục chứa nó):
  (a) WER phần TIẾNG VIỆT giữa hyp và nhãn <= --max-vi-wer (mặc định 0.15; span <en> không tính vào WER này);
  (b) ở mỗi vị trí span, hyp ra đúng chính tả Anh, HOẶC biến thể / phiên âm catalog của thuật ngữ, HOẶC một cách đọc trong
      lexicon của thuật ngữ, HOẶC đúng `reading` đã dùng để tổng hợp (ASR nghe ra phiên âm vẫn là audio đọc đúng). So cách đọc
      không phân biệt dấu thanh (ASR hay đổi dấu); chính tả Anh so nguyên văn;
  và (cho cả câu, không cần hyp):
  (c) tốc độ nói (âm tiết của tts_text / giây có tiếng) trong [--min-syl-per-sec, --max-syl-per-sec];
  (d) không clipping (--max-clip-frac), lặng đầu / cuối <= --max-edge-sil, lặng giữa câu liên tục <= --max-inner-sil,
      duration trong [--min-dur, --max-dur] (1-20 s).
Không có --hyps: chỉ chạy (c)(d) và in CẢNH BÁO rõ (không biết audio có đọc đúng chữ hay không).
Căn chỉnh hyp - nhãn là Levenshtein theo từ, trong đó mỗi span <en> là một đơn vị "ăn" 0..K từ hyp: khớp một cách đọc được chấp nhận
thì giá 0, không thì tính một lỗi span (không cộng vào WER tiếng Việt).
Ra: --out (manifest đã lọc, thêm tts.verify), <out>.rejected.jsonl (lý do loại), <out>.report.md: tỉ lệ giữ theo backend / giọng /
mức đọc / category, lý do loại, thuật ngữ còn < --min-voices-per-term giọng hoặc < --min-contexts-per-term câu sau lọc, và 30 thuật
ngữ bị loại nhiều nhất kèm cách đọc hay trượt nhất (dấu hiệu cách đọc trong lexicon SAI: sửa configs/dataset/synth/tts_lexicon.tsv).
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.textnorm import Catalog, basic_tokens, parse_tagged
from gipbench.dataset.synth.backends import analyze, count_syllables, fold, load_lexicon, read_audio, term_key

VI_COST, SPAN_MISS = 1000, 1


def load_hyps(spec):
    name, _, path = spec.partition("=")
    if not path:
        sys.exit(f"--hyps '{spec}' phải có dạng NAME=PATH")
    p = Path(path)
    if p.is_dir():
        p = p / "hyps.jsonl"
    if not p.exists():
        sys.exit(f"không có file hyps: {p}")
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            d[r["id"]] = r.get("hyp") or ""
    return name, d


class Alts:
    """Các cách ASR được phép ra cho một thuật ngữ: chính tả (exact) và cách đọc (so không dấu)."""

    def __init__(self, cat, lex):
        self.cat, self.lex, self.cache = cat, lex, {}
        self.rev = defaultdict(lambda: ([], []))  # canon -> (strict keys, phonetic keys)
        for k, canon in cat.lenient_map.items():
            (self.rev[canon][0] if k in cat.strict_map else self.rev[canon][1]).append(k)

    def get(self, key, used_reading):
        ck = (key, used_reading)
        if ck not in self.cache:
            canon = tuple(key.split())
            exact = {canon} | {tuple(k) for k in self.rev[canon][0]}
            fold_ = set()
            for k in self.rev[canon][1]:
                fold_.add(tuple(fold(t) for t in k))
            for r in self.lex.get(key, []):
                fold_.add(tuple(fold(t) for t in basic_tokens(r.reading)))
            if used_reading:
                fold_.add(tuple(fold(t) for t in basic_tokens(used_reading)))
            self.cache[ck] = (exact, fold_)
        return self.cache[ck]


def align_units(units, hyp, alts):
    """units: [("vi", tok) | ("span", key, used_reading)]; hyp: [token].
    Trả về (số lỗi VI, [{"term","ok","kind","got"}] theo thứ tự span)."""
    R, J = len(units), len(hyp)
    spans_K = {}
    INF = 10 ** 9
    dp = [[INF] * (J + 1) for _ in range(R + 1)]
    bk = [[None] * (J + 1) for _ in range(R + 1)]
    dp[0][0] = 0
    for j in range(1, J + 1):
        dp[0][j] = j * VI_COST
        bk[0][j] = ("i", 0)
    for i in range(1, R + 1):
        u = units[i - 1]
        if u[0] == "span":
            ex, fo = alts.get(u[1], u[2])
            K = max([len(a) for a in ex | fo] + [1]) + 2
            spans_K[i] = (ex, fo, K)
        for j in range(J + 1):
            best, how = INF, None
            if j > 0 and dp[i][j - 1] + VI_COST < best:  # chèn từ hyp
                best, how = dp[i][j - 1] + VI_COST, ("i", 0)
            if u[0] == "vi":
                if dp[i - 1][j] + VI_COST < best:
                    best, how = dp[i - 1][j] + VI_COST, ("d", 0)
                if j > 0:
                    c = dp[i - 1][j - 1] + (0 if u[1] == hyp[j - 1] else VI_COST)
                    if c < best:
                        best, how = c, ("m", 0)
            else:
                ex, fo, K = spans_K[i]
                for k in range(0, min(K, j) + 1):
                    seg = tuple(hyp[j - k:j])
                    ok = bool(seg) and (seg in ex or tuple(fold(t) for t in seg) in fo)
                    c = dp[i - 1][j - k] + (0 if ok else SPAN_MISS)
                    if c < best or (c == best and how and how[0] == "i"):
                        best, how = c, ("s", k)
            dp[i][j], bk[i][j] = best, how
    total = dp[R][J]
    res, i, j = [], R, J
    while i > 0 or j > 0:
        how = bk[i][j]
        if how is None:
            break
        op, k = how
        if op == "i":
            j -= 1
        elif op == "d":
            i -= 1
        elif op == "m":
            i, j = i - 1, j - 1
        else:
            seg = tuple(hyp[j - k:j])
            u = units[i - 1]
            ex, fo, _ = spans_K[i]
            kind = "spelling" if seg in ex else "reading" if seg and tuple(fold(t) for t in seg) in fo else ""
            res.append({"term": u[1], "ok": bool(kind), "kind": kind, "got": " ".join(seg)})
            i, j = i - 1, j - k
    res.reverse()
    return total // VI_COST, res


def units_of(text_tagged, readings):
    used = {term_key(r["term"]): r["reading"] for r in readings}
    units = []
    for seg, is_en in parse_tagged(text_tagged):
        if is_en:
            k = term_key(seg)
            units.append(("span", k, used.get(k, "")))
        else:
            units += [("vi", t) for t in basic_tokens(seg)]
    return units


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="train_tts_raw.jsonl")
    ap.add_argument("--hyps", nargs="*", default=[], help="NAME=PATH (hyps.jsonl hoặc thư mục); lặp được")
    ap.add_argument("--catalog")
    ap.add_argument("--lexicon", default=str(G / "configs/dataset/synth/tts_lexicon.tsv"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-agree", type=int, default=1, help="số hệ ASR phải đồng ý giữ câu")
    ap.add_argument("--max-vi-wer", type=float, default=0.15)
    ap.add_argument("--min-syl-per-sec", type=float, default=2.5)
    ap.add_argument("--max-syl-per-sec", type=float, default=8.0)
    ap.add_argument("--max-clip-frac", type=float, default=0.001, help="tỉ lệ mẫu |x| >= 0.999 tối đa (đo cả trên audio gốc của backend)")
    ap.add_argument("--max-edge-sil", type=float, default=0.6, help="lặng đầu / cuối tối đa (giây)")
    ap.add_argument("--max-inner-sil", type=float, default=1.2, help="lặng liên tục giữa câu tối đa (giây)")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--min-voices-per-term", type=int, default=3)
    ap.add_argument("--min-contexts-per-term", type=int, default=5)
    args = ap.parse_args(argv)

    rows = read_jsonl(args.manifest)
    cat = Catalog.load(args.catalog)
    lex, _ = load_lexicon(args.lexicon) if args.lexicon and Path(args.lexicon).exists() else ({}, [])
    alts = Alts(cat, lex)
    systems = [load_hyps(s) for s in args.hyps]
    if systems and args.min_agree > len(systems):
        sys.exit(f"--min-agree {args.min_agree} > số hệ ASR {len(systems)}")
    acoustic_only = not systems
    if acoustic_only:
        print("CẢNH BÁO: không có --hyps -> CHỈ kiểm âm học (tốc độ, clipping, lặng, thời lượng). Không biết audio có đọc đúng chữ "
              "không; đừng đưa vào train trước khi chạy src/gipbench/infer/asr_hyps.py + tts_verify --hyps.")

    kept, rejected = [], []
    reasons = Counter()
    span_kinds = Counter()
    term_stat = defaultdict(lambda: [0, 0])  # term -> [số câu, số câu bị loại]
    pair_stat = defaultdict(lambda: [0, 0, ""])  # (term, reading) -> [số span, số span trượt (mọi hệ), hyp ví dụ]
    for r in rows:
        t = r.get("tts") or {}
        rds = t.get("readings") or []
        why = []
        # ---- âm học
        try:
            x, sr = read_audio(r["_audio_path"])
            a = analyze(x, sr)
        except Exception as e:  # noqa: BLE001
            a, why = None, [f"audio không đọc được ({e.__class__.__name__})"]
        sps = None
        if a:
            st = t.get("stats") or {}
            dur = a["duration"]
            if not (args.min_dur <= dur <= args.max_dur):
                why.append("thời lượng ngoài khoảng")
            if max(a["clip_frac"], float(st.get("clip_frac", 0) or 0)) > args.max_clip_frac:
                why.append("clipping")
            if a["voiced"] <= 0:
                why.append("không có tiếng")
            else:
                if max(a["lead"], a["trail"]) > args.max_edge_sil:
                    why.append("lặng đầu/cuối dài")
                if a["inner"] > args.max_inner_sil:
                    why.append("lặng giữa câu dài")
                sps = count_syllables(t.get("tts_text") or r["text_tagged"]) / max(0.1, dur - a["lead"] - a["trail"])
                if sps < args.min_syl_per_sec:
                    why.append("nói quá chậm / kéo dài")
                elif sps > args.max_syl_per_sec:
                    why.append("nói quá nhanh / nuốt từ")
        # ---- đối chiếu hyp
        ver = {"systems_pass": 0, "n_systems": len(systems), "vi_wer": [], "span_ok": [], "syl_per_sec": round(sps, 2) if sps else None}
        span_fail_all = None
        if systems:
            units = units_of(r["text_tagged"], rds)
            n_vi = sum(1 for u in units if u[0] == "vi")
            first_got = {}
            for name, hy in systems:
                if r["id"] not in hy:
                    ver["vi_wer"].append(None)
                    ver["span_ok"].append(None)
                    continue
                err, res = align_units(units, basic_tokens(hy[r["id"]]), alts)
                wer = err / max(1, n_vi)
                ok_sp = all(s["ok"] for s in res)
                ver["vi_wer"].append(round(wer, 3))
                ver["span_ok"].append(ok_sp)
                if wer <= args.max_vi_wer and ok_sp:
                    ver["systems_pass"] += 1
                bad = {s["term"] for s in res if not s["ok"]}
                span_fail_all = bad if span_fail_all is None else span_fail_all & bad
                for s in res:
                    if s["ok"]:
                        span_kinds[s["kind"]] += 1
                    first_got.setdefault(s["term"], s["got"])
            n_with = sum(1 for v in ver["vi_wer"] if v is not None)
            if n_with == 0:
                why.append("không có hyp")
            elif ver["systems_pass"] < args.min_agree:
                vi_bad = any(v is not None and v > args.max_vi_wer for v in ver["vi_wer"])
                sp_bad = any(s is False for s in ver["span_ok"])
                if vi_bad:
                    why.append("WER tiếng Việt cao")
                if sp_bad:
                    why.append("span tiếng Anh sai")
                if not (vi_bad or sp_bad):
                    why.append("thiếu hyp ở các hệ còn lại (không đủ --min-agree)")
            for rd in rds:
                p = pair_stat[(rd["term"], rd["reading"])]
                p[0] += 1
                if span_fail_all and rd["term"] in span_fail_all and ver["systems_pass"] < args.min_agree:
                    p[1] += 1
                    p[2] = p[2] or first_got.get(rd["term"], "")
        for rd in rds:
            term_stat[rd["term"]][0] += 1
            if why:
                term_stat[rd["term"]][1] += 1
        for w in set(why):
            reasons[w] += 1
        r.setdefault("tts", {})["verify"] = ver
        (rejected if why else kept).append((r, why))

    out_rows = [r for r, _ in kept]
    write_jsonl(args.out, out_rows)
    write_jsonl(args.out + ".rejected.jsonl", [{"id": r["id"], "voice": (r.get("tts") or {}).get("voice"), "reasons": w,
                                                "terms": [x["term"] for x in (r.get("tts") or {}).get("readings", [])],
                                                "verify": (r.get("tts") or {}).get("verify")} for r, w in rejected])

    # ---- report
    def rate_table(title, keyf):
        tot, ok = Counter(), Counter()
        for r, w in kept + rejected:
            for k in set(keyf(r)):
                tot[k] += 1
                ok[k] += 0 if w else 1
        md = ["", f"### Tỉ lệ giữ theo {title}", "", f"| {title} | Giữ | Tổng | Tỉ lệ |", "| --- | --- | --- | --- |"]
        return md + [f"| {k} | {ok[k]} | {tot[k]} | {ok[k] / tot[k]:.1%} |" for k in sorted(tot, key=lambda k: -tot[k])[:40]]

    tv, tc = defaultdict(set), Counter()
    for r in out_rows:
        for rd in r["tts"]["readings"]:
            tv[rd["term"]].add(r["tts"]["voice"])
            tc[rd["term"]] += 1
    all_terms = set(term_stat)
    low_v = sorted(t for t in all_terms if len(tv[t]) < args.min_voices_per_term)
    low_c = sorted(t for t in all_terms if tc[t] < args.min_contexts_per_term)
    hrs = lambda rs: sum(float(r["duration"]) for r in rs) / 3600  # noqa: E731
    md = ["# Lọc TTS", "",
          f"`{args.manifest}` -> `{args.out}`: giữ **{len(kept)} / {len(rows)}** câu ({len(kept) / max(1, len(rows)):.1%}; "
          f"{hrs(out_rows):.2f} h / {hrs(rows):.2f} h).", ""]
    if acoustic_only:
        md += ["> **CẢNH BÁO: không có --hyps. Chỉ kiểm âm học; chưa kiểm audio có đọc đúng chữ. Đừng train trước khi lọc bằng hyp ASR.**", ""]
    else:
        md += [f"Hệ ASR: {', '.join(n for n, _ in systems)}; --min-agree {args.min_agree}, --max-vi-wer {args.max_vi_wer:g}."]
    md += [f"Ngưỡng: {args.min_syl_per_sec:g}–{args.max_syl_per_sec:g} âm tiết/giây, clipping <= {args.max_clip_frac:g}, lặng đầu/cuối <= "
           f"{args.max_edge_sil:g} s, lặng giữa <= {args.max_inner_sil:g} s, thời lượng {args.min_dur:g}–{args.max_dur:g} s.", "",
           "### Lý do loại (một câu có thể nhiều lý do)", "", "| Lý do | Câu |", "| --- | --- |"]
    md += [f"| {k} | {v} |" for k, v in reasons.most_common()] or ["| (không câu nào bị loại) | 0 |"]
    if span_kinds:
        md += ["", "### Span được chấp nhận theo kiểu (mọi hệ)", "", "| Kiểu | Span |", "| --- | --- |"]
        md += [f"| {k} | {v} |" for k, v in span_kinds.most_common()]
    md += rate_table("backend", lambda r: [(r.get("tts") or {}).get("backend", "?")])
    md += rate_table("giọng", lambda r: [(r.get("tts") or {}).get("voice", "?")])
    md += rate_table("mức đọc (viet_level)", lambda r: [m.get("viet_level", "?") for m in r.get("en_meta") or []] or ["?"])
    md += rate_table("category", lambda r: [(r.get("tts") or {}).get("category") or "(trống)"])
    md += ["", f"### Thuật ngữ còn < {args.min_voices_per_term} giọng sau lọc: {len(low_v)}", "",
           ", ".join(f"{t} ({len(tv[t])})" for t in low_v[:200]) or "không có"]
    md += ["", f"### Thuật ngữ còn < {args.min_contexts_per_term} câu sau lọc: {len(low_c)}", "",
           ", ".join(f"{t} ({tc[t]})" for t in low_c[:200]) or "không có"]
    top = sorted(((t, v[1], v[0]) for t, v in term_stat.items() if v[1]), key=lambda x: (-x[1], x[0]))[:30]
    md += ["", "### 30 thuật ngữ bị loại nhiều nhất (cách đọc trong lexicon nghi sai nếu tỉ lệ loại cao)", "",
           "| Thuật ngữ | Bị loại | Tổng | Tỉ lệ | Cách đọc hay trượt nhất (hyp ví dụ) |", "| --- | --- | --- | --- | --- |"]
    for t, bad, tot in top:
        worst = max(((p[1], rd, p[2]) for (tt, rd), p in pair_stat.items() if tt == t), default=(0, "", ""))
        md.append(f"| {t} | {bad} | {tot} | {bad / tot:.0%} | " + (f"`{worst[1]}` ({worst[0]} lần; hyp: {worst[2] or '-'})" if worst[0] else "-") + " |")
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Giữ {len(kept)}/{len(rows)} câu ({hrs(out_rows):.2f} h); loại: "
          + (", ".join(f"{k} {v}" for k, v in reasons.most_common()) or "không"))
    print(f"Thuật ngữ còn < {args.min_voices_per_term} giọng: {len(low_v)}; < {args.min_contexts_per_term} câu: {len(low_c)}")
    print(f"Đã ghi {args.out}, {args.out}.rejected.jsonl, {args.out}.report.md")


if __name__ == "__main__":
    main()
