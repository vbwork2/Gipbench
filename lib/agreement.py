"""Lọc nhãn máy theo đồng thuận: giữ câu CS khi span tiếng Anh của nhãn gốc xuất hiện trong hyp của các model khác.

Dùng cho vss_cs (nhãn Zipformer-30M) và ft/curate_train.py (--agreement-hyps); SEA (tùy chọn, không dùng) cũng đi
được đường này. Model xác nhận là model đa ngôn ngữ (faster-whisper large-v3, Qwen3-ASR), không dùng Gipformer vì nó
hiếm khi ra chính tả Anh. Reference và hyp đều đi qua lib.textnorm (normalize_ref / normalize_hyp, strict, cùng catalog).

relabel_spans(): gán lại TỪNG span (không gán lại cả câu) khi mọi hyp cho cùng một chữ ở vị trí span đó và phần
tiếng Việt của nhãn khớp từng hyp (WER tiếng Việt <= ngưỡng).
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from lib.align import align, attribute
from lib.textnorm import Catalog, basic_tokens, normalize_hyp, normalize_ref, plain_text


def label_spans(text_tagged: str, cat: Catalog) -> List[Tuple[str, ...]]:
    """Các span <en> của nhãn, dạng token đã chuẩn hoá (strict)."""
    toks, _ = normalize_ref(text_tagged, cat, "strict")
    spans: Dict[int, List[str]] = {}
    for t in toks:
        if t.span >= 0:
            spans.setdefault(t.span, []).append(t.tok)
    return [tuple(spans[k]) for k in sorted(spans)]


def contains(seq: Sequence[str], sub: Sequence[str]) -> bool:
    n = len(sub)
    return n > 0 and any(tuple(seq[i:i + n]) == tuple(sub) for i in range(len(seq) - n + 1))


def agree(text_tagged: str, hyps: Sequence[str], cat: Catalog, min_agree: int = 1,
          min_span_frac: float = 1.0) -> Dict:
    """Một span "được xác nhận" khi nó có mặt (chuỗi token liền) trong ít nhất `min_agree` hyp.

    Giữ câu khi tỉ lệ span được xác nhận >= min_span_frac (mặc định 1.0 = mọi span). Câu không có span: không giữ
    (không có gì để xác nhận). Trả về {"keep", "spans", "confirmed": [số hyp xác nhận từng span]}.
    """
    spans = label_spans(text_tagged, cat)
    hyp_toks = [normalize_hyp(h or "", cat, "strict") for h in hyps]
    votes = [sum(contains(h, s) for h in hyp_toks) for s in spans]
    ok = sum(v >= min_agree for v in votes)
    keep = bool(spans) and ok >= min_span_frac * len(spans)
    return {"keep": keep, "spans": [" ".join(s) for s in spans], "confirmed": votes}


def span_segments(ref_tok: List[str], ref_span: List[int], hyp: List[str]
                  ) -> Tuple[Dict[int, List[str]], int, int]:
    """Căn nhãn (đã chuẩn hoá) với một hyp. Trả về (đoạn hyp ứng với từng span, số lỗi phần tiếng Việt,
    số token tiếng Việt của nhãn).

    `ref_span[i]` = chỉ số span của token nhãn thứ i (-1 = tiếng Việt). Token hyp thuộc span khi được căn (C/S)
    với một token của span, hoặc là token chèn mà lib.align.attribute quy về span đó (kề token tiếng Anh, ưu tiên
    bên trái). Lỗi tiếng Việt = S/D trên token tiếng Việt + chèn quy về tiếng Việt (cùng quy ước với score.py).
    """
    ops = align(ref_tok, hyp)
    errs, _, _, _ = attribute(ops, ref_span)
    seg: Dict[int, List[str]] = {}
    last_ref, nxt, next_ref = None, None, [None] * len(ops)
    for k in range(len(ops) - 1, -1, -1):
        next_ref[k] = nxt
        if ops[k][1] is not None:
            nxt = ops[k][1]
    for k, (t, ri, hj) in enumerate(ops):
        if t in ("C", "S"):
            last_ref = ri
            if ref_span[ri] >= 0:
                seg.setdefault(ref_span[ri], []).append(hyp[hj])
        elif t == "D":
            last_ref = ri
        else:  # chèn: cùng quy tắc với attribute()
            left = ref_span[last_ref] if last_ref is not None else -1
            right = ref_span[next_ref[k]] if next_ref[k] is not None else -1
            sp = left if left >= 0 else right
            if sp >= 0:
                seg.setdefault(sp, []).append(hyp[hj])
    n_vi = sum(1 for s in ref_span if s < 0)
    return seg, errs.get(-1, 0), n_vi


_EN = re.compile(r"(<en(?:\s[^>]*)?>)(.*?)(</en>)", re.S)


def relabel_spans(text_tagged: str, hyps: Sequence[Optional[str]], cat: Catalog, min_agree: int = 1,
                  max_vi_wer: float = 0.10) -> Dict:
    """Gán lại các span CHƯA được xác nhận (ít hơn min_agree hyp chứa nó).

    Một span được thay khi: có ít nhất 2 hyp và hyp nào cũng có mặt (không rỗng); đoạn hyp ứng với span
    giống hệt nhau ở MỌI hyp sau chuẩn hoá (strict, catalog) và không rỗng; WER phần tiếng Việt của nhãn so với
    TỪNG hyp <= max_vi_wer. Phần tiếng Việt và các span khác giữ nguyên.
    Trả về {"text_tagged", "changed": [(span_idx, cũ, mới)], "vi_wer": [...], "reason"}.
    """
    out = {"text_tagged": text_tagged, "changed": [], "vi_wer": [], "reason": ""}
    if len(hyps) < 2 or any(not (h or "").strip() for h in hyps):
        out["reason"] = "thiếu hyp"
        return out
    toks, spans = normalize_ref(text_tagged, cat, "strict")
    ref_tok, ref_span = [t.tok for t in toks], [t.span for t in toks]
    hyp_toks = [normalize_hyp(h, cat, "strict") for h in hyps]
    votes = [sum(contains(h, s) for h in hyp_toks) for s in spans]
    todo = [i for i, v in enumerate(votes) if v < min_agree]
    if not todo:
        out["reason"] = "không có span cần gán lại"
        return out
    segs, wers = [], []
    for h in hyp_toks:
        seg, e_vi, n_vi = span_segments(ref_tok, ref_span, h)
        segs.append(seg)
        wers.append(e_vi / n_vi if n_vi else float("inf"))
    out["vi_wer"] = [round(w, 4) for w in wers]
    if any(w > max_vi_wer for w in wers):
        out["reason"] = "WER tiếng Việt vượt ngưỡng"
        return out
    new = {}
    for i in todo:
        cand = [tuple(s.get(i, [])) for s in segs]
        if cand[0] and all(c == cand[0] for c in cand) and cand[0] != spans[i]:
            new[i] = " ".join(cand[0])
    if not new:
        out["reason"] = "các ASR không cho cùng chữ"
        return out
    # dựng lại text_tagged: span thứ k (đếm span có token sau chuẩn hoá, như normalize_ref) -> chữ mới
    sid = [-1]

    def sub(m):
        if not cat.apply(basic_tokens(m.group(2)), "strict"):
            return m.group(0)
        sid[0] += 1
        return m.group(1) + new[sid[0]] + m.group(3) if sid[0] in new else m.group(0)
    rebuilt = _EN.sub(sub, text_tagged)
    if sid[0] + 1 != len(spans):  # đếm span lệch với normalize_ref: không đụng vào câu này
        out["reason"] = "không dựng lại được nhãn"
        return out
    out["text_tagged"] = rebuilt
    out["changed"] = [(i, " ".join(spans[i]), new[i]) for i in sorted(new)]
    return out


def plain(text_tagged: str) -> str:
    return " ".join(plain_text(text_tagged).split())
