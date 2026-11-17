"""Công thức metric dùng chung cho bench (chấm điểm, leaderboard) và model (chọn checkpoint trên dev).

Chuyển nguyên từ score.py; định nghĩa metric xem docstring của gipbench.bench.score. Mọi chỗ tính WER / CS-WER / span metric
phải gọi các hàm ở đây để bench và chọn checkpoint không lệch nhau.
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core.align import align, attribute, char_edit_distance, span_hyps
from gipbench.core.manifest import CS_SPLITS, read_jsonl, span_meta
from gipbench.core.textnorm import Catalog, basic_tokens, normalize_hyp, normalize_ref, plain_text

MAIN = ["B1", "B2"]


def count_occ(toks, term):
    n, c, i = len(term), 0, 0
    while i <= len(toks) - n:
        if tuple(toks[i:i + n]) == term:
            c += 1
            i += n
        else:
            i += 1
    return c


def train_term_freq(paths, cat, terms):
    """Đếm số lần mỗi thuật ngữ (tuple token) xuất hiện trong text train, tách dòng thật / dòng synthetic.
    Trả về {"t1 t2": (real, synth)} cho mọi thuật ngữ trong `terms`."""
    first = defaultdict(list)
    for t in terms:
        first[t[0]].append(t)
    cnt = {t: [0, 0] for t in terms}
    for p in paths:
        for r in read_jsonl(p):
            toks = cat.apply(basic_tokens(plain_text(r.get("text_tagged") or r.get("text") or "")), "strict")
            col = 1 if r.get("synthetic") else 0
            for t in {t for tok in set(toks) for t in first.get(tok, ())}:
                cnt[t][col] += count_occ(toks, t)
    return {" ".join(t): tuple(v) for t, v in cnt.items()}


def score_utt(r, hyp_text, cat, label_field="text_tagged", freq=None):
    rec = {"id": r["id"], "split": r["split"], "speaker": r["speaker"],
           "boot_unit": r.get("speaker_cluster") or r["speaker"]}
    for k in ("region", "gender", "channel", "domain"):
        rec[k] = r.get(k, "unk")
    ref_text = r.get(label_field) or r["text_tagged"]
    tcat = r.get("term_category") or {}
    for mode in ("strict", "lenient"):
        rt, spans = normalize_ref(ref_text, cat, mode)
        ht = normalize_hyp(hyp_text, cat, mode)
        ref = [t.tok for t in rt]
        sp = [t.span for t in rt]
        ops = align(ref, ht)
        errs, S, D, I = attribute(ops, sp)
        n_en = sum(s >= 0 for s in sp)
        sfx = "" if mode == "strict" else "_len"
        rec["err_en" + sfx] = sum(v for k, v in errs.items() if k >= 0)
        if mode == "strict":
            rec.update(n=len(ref), n_en=n_en, n_vi=len(ref) - n_en, S=S, D=D, I=I,
                       err=S + D + I, err_vi=errs.get(-1, 0))
            rec["spans"] = []
            sh = span_hyps(ops, sp, ht, len(spans))
            for si, term in enumerate(spans):
                t = cat.terms.get(term)
                ts = " ".join(term)
                rj, hj, e = "".join(term), "".join(sh[si]), errs.get(si, 0)
                sd = {"term": ts, "n": sum(1 for s in sp if s == si), "err": e,
                      "category": tcat.get(ts) or (t.category if t and t.category else "(ngoài catalog)"),
                      "viet_level": span_meta(r, si).get("viet_level", "chưa gán"),
                      "hyp": " ".join(sh[si]), "exact": e == 0 or rj == hj,
                      "cer": char_edit_distance(rj, hj) / len(rj)}
                if freq is not None:
                    sd["train_freq_real"], sd["train_freq_synth"] = freq.get(ts, (0, 0))
                rec["spans"].append(sd)
        # thuật ngữ
        ref_terms = Counter(spans)
        matched = sum(min(c, count_occ(ht, t)) for t, c in ref_terms.items())
        hyp_terms = Counter(cat.find_terms(ht))
        for t in ref_terms:  # span ngoài catalog vẫn tính
            if t not in cat.terms:
                hyp_terms[t] += count_occ(ht, t)
        hyp_hits = sum(hyp_terms.values())
        rec["ref_terms" + sfx] = sum(ref_terms.values())
        rec["matched_terms" + sfx] = matched
        rec["hyp_terms" + sfx] = hyp_hits
        rec["false_terms" + sfx] = sum(max(0, c - ref_terms.get(t, 0)) for t, c in hyp_terms.items())
    return rec


# --- phân rã CS-WER và giữ tiếng Việt (IP-04; chỉ THÊM trường, không đổi score_utt / agg) ---
# Âm tiết tiếng Việt viết không dấu (vd 'anh', 'cho', 'con', 'ban'): phụ âm đầu + vần + âm cuối theo luật chính tả.
# Token ASCII không khớp mẫu này (vd 'ok', 'meeting', 'team', 'code') được coi là "giống từ Anh".
_VI_INIT = "ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx]"
_VI_NUC = ("oai|oao|oay|oeo|uay|uoi|uou|uya|uye|uyu|ieu|yeu|ai|ao|au|ay|eo|eu|ia|ie|iu|oa|oe|oi|oo|ua|ue|ui|uo|uu|uy|"
           "ya|ye|[aeiouy]")
_VI_SYL = re.compile(rf"(?:{_VI_INIT})?(?:{_VI_NUC})(?:ch|ng|nh|[cmnpt])?")
_ASCII_WORD = re.compile(r"[a-z]{2,}")


def en_like(tok: str) -> bool:
    """Token chỉ gồm chữ Latin không dấu, dài >= 2, và không phải một âm tiết tiếng Việt hợp lệ khi viết không dấu."""
    return bool(_ASCII_WORD.fullmatch(tok)) and not _VI_SYL.fullmatch(tok)


def cs_breakdown(ops, ref_span, ref_toks, hyp_toks):
    """Đếm theo câu (cùng ops / span với score_utt nên Σ = err_en):
    S_en, D_en              : thay / xoá token span Anh;
    I_en_left               : chèn có token ref bên trái thuộc span Anh;
    I_en_right_only         : chèn có token ref bên trái là tiếng Việt (hoặc đầu câu), bên phải thuộc span Anh
                              (luật hiện tại tính cho tiếng Anh; audit 2.2: ~10-16% lỗi EN của C1 là loại này);
    boundary_ins            : chèn mà token ref trái và phải khác ngôn ngữ (ranh giới chuyển ngữ; thiếu một bên thì không);
    err_en_sd               : S_en + D_en (CS-WER không tính chèn);
    utt_vi_only, n_vi_only, err_vi_only: câu thuần Việt = ref không rỗng, không có span Anh VÀ không có token ref nào
                              `en_like` (bộ không gắn tag như G0 có từ Anh trong ref không tag: câu đó không tính);
    hyp_latin_extra         : (chỉ câu thuần Việt) số token hyp `en_like` không có trong ref (từ Anh chèn vào câu Việt).
    """
    next_ref = [None] * len(ops)
    nxt = None
    for k in range(len(ops) - 1, -1, -1):
        next_ref[k] = nxt
        if ops[k][1] is not None:
            nxt = ops[k][1]
    c = Counter()
    last_ref = None
    for k, (t, ri, _) in enumerate(ops):
        if t != "I":
            if t in ("S", "D") and ref_span[ri] >= 0:
                c["S_en" if t == "S" else "D_en"] += 1
            last_ref = ri
            continue
        left = ref_span[last_ref] if last_ref is not None else None
        right = ref_span[next_ref[k]] if next_ref[k] is not None else None
        if left is not None and left >= 0:
            c["I_en_left"] += 1
        elif right is not None and right >= 0:
            c["I_en_right_only"] += 1
        if left is not None and right is not None and (left >= 0) != (right >= 0):
            c["boundary_ins"] += 1
    n_en = sum(s >= 0 for s in ref_span)
    vi_only = n_en == 0 and len(ref_toks) > 0 and not any(en_like(t) for t in ref_toks)
    rset = set(ref_toks)
    out = {k: c[k] for k in ("S_en", "D_en", "I_en_left", "I_en_right_only", "boundary_ins")}
    out.update(err_en_sd=c["S_en"] + c["D_en"], utt_vi_only=int(vi_only), n_vi_only=len(ref_toks) if vi_only else 0,
               err_vi_only=sum(o[0] != "C" for o in ops) if vi_only else 0,
               hyp_latin_extra=sum(1 for h in hyp_toks if h not in rset and en_like(h)) if vi_only else 0)
    return out


def term_weights(recs):
    """Trọng số cố định theo thuật ngữ cho bootstrap: w = 1 / (số span của thuật ngữ trong TOÀN BỘ tập `recs`).
    Trên tập đầy đủ, Σ w·exact / Σ w = term_acc_macro (Σ w = số thuật ngữ)."""
    n = Counter(x["term"] for r in recs for x in r.get("spans", []) if "exact" in x)
    return {t: 1.0 / c for t, c in n.items()}


def span_stats(spans, term_w=None):
    """Chỉ số theo span trên một danh sách span (thang %). Span thiếu `exact` (utt_scores cũ) bị bỏ qua.
    term_w (từ term_weights của tập đầy đủ): tính term_acc_macro trên MẪU bootstrap bằng trọng số cố định thay vì
    trung bình đều qua các thuật ngữ có mặt trong mẫu (cách đó lệch lên vì thuật ngữ hiếm rơi khỏi mẫu)."""
    sp = [x for x in spans if "exact" in x]
    if not sp:
        return {"n_span": 0, "n_term": 0, "span_acc": None, "term_acc_macro": None, "span_cer": None, "cs_wer": None}
    per = defaultdict(lambda: [0, 0])  # thuật ngữ -> [số span exact, số span]
    for x in sp:
        per[x["term"]][0] += bool(x["exact"])
        per[x["term"]][1] += 1
    n_tok = sum(x["n"] for x in sp)
    if term_w:
        den = sum(term_w.get(x["term"], 0.0) for x in sp)
        macro = 100.0 * sum(term_w.get(x["term"], 0.0) * bool(x["exact"]) for x in sp) / den if den else None
    else:
        macro = 100.0 * sum(a / b for a, b in per.values()) / len(per)
    return {"n_span": len(sp), "n_term": len(per),
            "span_acc": 100.0 * sum(bool(x["exact"]) for x in sp) / len(sp),
            "term_acc_macro": macro,
            "span_cer": 100.0 * sum(min(x["cer"], 1.0) for x in sp) / len(sp),
            "cs_wer": 100.0 * sum(x["err"] for x in sp) / n_tok if n_tok else None}


FREQ_SLICES = ["unseen", "synth_only", "real_1-4", "real_5+"]


def freq_slice(x):
    """Lát tần suất train của một span (None nếu span không có train_freq_*)."""
    if "train_freq_real" not in x:
        return None
    real, synth = x["train_freq_real"], x.get("train_freq_synth", 0)
    if real == 0:
        return "synth_only" if synth > 0 else "unseen"
    return "real_1-4" if real < 5 else "real_5+"


def freq_slices(recs):
    """{lát: span_stats} theo tần suất thuật ngữ trong train; {} nếu bench chưa gắn train_freq."""
    g = defaultdict(list)
    for r in recs:
        for x in r.get("spans", []):
            k = freq_slice(x)
            if k:
                g[k].append(x)
    return {k: span_stats(g[k]) for k in FREQ_SLICES} if g else {}


SPAN_METRICS = ("span_acc", "term_acc_macro", "span_cer")


def agg(recs, with_spans=True, term_w=None):
    """Gộp chỉ số trên danh sách utt. with_spans=False bỏ qua chỉ số theo span (nhanh hơn trong bootstrap khi
    không cần chúng; khi đó các khoá span là None). term_w: xem span_stats (chỉ dùng trong bootstrap)."""
    s = Counter()
    for r in recs:
        for k in ("n", "err", "n_en", "err_en", "err_en_len", "n_vi", "err_vi", "ref_terms", "matched_terms",
                  "hyp_terms", "false_terms", "matched_terms_len", "S", "D", "I"):
            s[k] += r.get(k, 0)
    s["utt"] = len(recs)
    ss = span_stats([x for r in recs for x in r.get("spans", [])] if with_spans else [], term_w)

    def rate(a, b):
        return 100.0 * s[a] / s[b] if s[b] else None
    return {"utt": len(recs), "n": s["n"], "n_en": s["n_en"],
            "wer": rate("err", "n"), "cs_wer": rate("err_en", "n_en"), "cs_wer_lenient": rate("err_en_len", "n_en"),
            "n_wer": rate("err_vi", "n_vi"), "term_recall": rate("matched_terms", "ref_terms"),
            "term_recall_lenient": rate("matched_terms_len", "ref_terms"),
            "term_precision": rate("matched_terms", "hyp_terms"),
            "false_terms_per_100utt": 100.0 * s["false_terms"] / len(recs) if recs else None,
            "S": s["S"], "D": s["D"], "I": s["I"],
            "n_span": ss["n_span"], "n_term": ss["n_term"], "span_acc": ss["span_acc"],
            "term_acc_macro": ss["term_acc_macro"], "span_cer": ss["span_cer"]}


def bootstrap_many(recs, metrics, iters=1000, seed=0):
    """CI 95% nhiều metric một lượt bằng cách lấy mẫu lại NGƯỜI NÓI (boot_unit: speaker_cluster nếu có, không thì
    speaker; có hoàn lại). Mỗi mẫu tính lại agg từ đầu nên metric macro cũng được tính lại đúng trong mẫu."""
    by = defaultdict(list)
    for r in recs:
        by[r.get("boot_unit") or r["speaker"]].append(r)
    spk = list(by)
    if len(spk) < 2:
        return {m: None for m in metrics}
    rnd = random.Random(seed)
    vals = {m: [] for m in metrics}
    ws = any(m in SPAN_METRICS for m in metrics)
    tw = term_weights(recs) if "term_acc_macro" in metrics else None  # cố định theo tập đầy đủ, không tính lại theo mẫu
    for _ in range(iters):
        sample = [x for s in (rnd.choice(spk) for _ in spk) for x in by[s]]
        a = agg(sample, ws, tw)
        for m in metrics:
            if a[m] is not None:
                vals[m].append(a[m])
    out = {}
    for m, v in vals.items():
        v.sort()
        out[m] = [v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]] if v else None
    return out


def bootstrap(recs, metric, iters=1000, seed=0):
    return bootstrap_many(recs, [metric], iters, seed)[metric]


GROUPS = {"main_B1+B2": MAIN, "B1": ["B1"], "B2": ["B2"], "B3": ["B3"], "B4": ["B4"], "B5": ["B5"], "cs_B1-B4": CS_SPLITS}
