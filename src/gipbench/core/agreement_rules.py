"""Luật xác nhận span <en> bằng đồng thuận ASR (thuần logic, không đọc file).

Dùng chung cho dataset (lọc vss_cs: gipbench.dataset.pseudolabel.agreement_filter) và model (curate --agreement-hyps).
Chuyển nguyên từ lib/agreement.py.
"""
from typing import Dict, List, Sequence, Tuple

from gipbench.core.textnorm import Catalog, normalize_hyp, normalize_ref


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
