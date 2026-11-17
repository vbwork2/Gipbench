"""Lọc nhãn máy theo đồng thuận: giữ câu CS khi span tiếng Anh của nhãn gốc xuất hiện trong hyp của các model khác.

Dùng cho vss_cs (nhãn Zipformer-30M) và src/gipbench/model/recipe/curate.py (--agreement-hyps); SEA (tùy chọn, không dùng) cũng đi
được đường này. Model xác nhận là model đa ngôn ngữ (faster-whisper large-v3, Qwen3-ASR), không dùng Gipformer vì nó
hiếm khi ra chính tả Anh. Reference và hyp đều đi qua lib.textnorm (normalize_ref / normalize_hyp, strict, cùng catalog).

relabel_spans(): gán lại TỪNG span (không gán lại cả câu) khi mọi hyp cho cùng một chữ ở vị trí span đó và phần
tiếng Việt của nhãn khớp từng hyp (WER tiếng Việt <= ngưỡng).

mine_hard(): chế độ ngược lại (khai thác câu KHÓ): nhãn = text của các thầy ASR mạnh đồng thuận với nhau, giữ câu mà
Gipformer (trò) không có span tiếng Anh mà các thầy đều có. Xem khối "khai thác câu KHÓ" ở cuối file.
"""
import re
from typing import Dict, List, Optional, Sequence, Tuple

from gipbench.core.align import align, attribute
from gipbench.core.textnorm import Catalog, basic_tokens, normalize_hyp, normalize_ref, plain_text


from gipbench.core.agreement_rules import agree, contains, label_spans  # noqa: F401,E402  (import lại)


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


def vi_wer(text_tagged: str, hyp: str, cat: Catalog) -> float:
    """WER phần tiếng Việt của nhãn so với một hyp, cùng cách tính với relabel_spans (span_segments).
    Nhãn không có token tiếng Việt: inf."""
    toks, _ = normalize_ref(text_tagged, cat, "strict")
    _, e_vi, n_vi = span_segments([t.tok for t in toks], [t.span for t in toks], normalize_hyp(hyp or "", cat, "strict"))
    return e_vi / n_vi if n_vi else float("inf")


_EN = re.compile(r"(<en(?:\s[^>]*)?>)(.*?)(</en>)", re.S)
# chữ có dấu tiếng Việt (kể cả đ): span tiếng Anh mới không được chứa token nào có các chữ này
VI_DIACRITIC = re.compile("[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
MAX_EXTRA_TOKENS = 1  # span mới dài hơn span cũ tối đa 1 token


def span_rule_violation(old: Sequence[str], new: Sequence[str]) -> str:
    """Luật chặn span gán lại (Phase 0d): trả về lý do vi phạm, chuỗi rỗng = hợp lệ."""
    if any(VI_DIACRITIC.search(t) for t in new):
        return "span mới có chữ có dấu tiếng Việt"
    if len(new) > len(old) + MAX_EXTRA_TOKENS:
        return f"span mới dài hơn span cũ quá {MAX_EXTRA_TOKENS} token"
    return ""


def relabel_spans(text_tagged: str, hyps: Sequence[Optional[str]], cat: Catalog, min_agree: int = 1,
                  max_vi_wer: float = 0.10) -> Dict:
    """Gán lại các span CHƯA được xác nhận (ít hơn min_agree hyp chứa nó).

    Một span được thay khi: có ít nhất 2 hyp và hyp nào cũng có mặt (không rỗng); đoạn hyp ứng với span
    giống hệt nhau ở MỌI hyp sau chuẩn hoá (strict, catalog) và không rỗng; WER phần tiếng Việt của nhãn so với
    TỪNG hyp <= max_vi_wer; span mới không có chữ có dấu tiếng Việt và không dài hơn span cũ quá 1 token
    (span_rule_violation). Phần tiếng Việt và các span khác giữ nguyên.
    Trả về {"text_tagged", "changed": [(span_idx, cũ, mới)], "rejected": [(span_idx, cũ, mới, lý do)],
    "vi_wer": [...], "reason"}.
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
    new, rejected = {}, []
    for i in todo:
        cand = [tuple(s.get(i, [])) for s in segs]
        if not (cand[0] and all(c == cand[0] for c in cand) and cand[0] != spans[i]):
            continue
        why = span_rule_violation(spans[i], cand[0])
        if why:
            rejected.append((i, " ".join(spans[i]), " ".join(cand[0]), why))
        else:
            new[i] = " ".join(cand[0])
    out["rejected"] = rejected
    if not new:
        out["reason"] = rejected[0][3] if rejected else "các ASR không cho cùng chữ"
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


# ----------------------------------------------------------------------------- khai thác câu KHÓ (hard mining)
# Ngược với agree()/relabel_spans(): ở đó nhãn là nhãn giả của model nhỏ và chỉ giữ câu mà ASR khác xác nhận, nên chỉ còn
# câu DỄ (trò gần như đã đúng). Ở đây nhãn là text của các "thầy" (ASR mạnh, khác họ với Gipformer) đồng thuận với nhau,
# và ta giữ câu mà "trò" (Gipformer) thiếu span tiếng Anh mà các thầy đều có.
# Thứ tự các điều kiện (cũng là thứ tự đếm phễu trong báo cáo của src/gipbench/dataset/pseudolabel/agreement_filter.py); khoá -> mô tả
HARD_STAGES = [
    ("hyp", "đủ hyp thầy (>= 2 thầy, không rỗng)"),
    ("pair", "thầy đồng thuận: WER từng cặp thầy <= ngưỡng"),
    ("has_span", "text thầy chính có span tiếng Anh"),
    ("span_confirmed", "mọi thầy khác có đúng chuỗi token của mọi span"),
    ("leak", "không rò rỉ (reserved_b3, test_only, bench, danh sách chặn)"),
    ("en_frac", "tỉ lệ token tiếng Anh <= --max-en-frac"),
    ("student_vi", "phần tiếng Việt (ngoài span) của trò lệch thầy <= ngưỡng"),
]


def pair_wer(a: Sequence[str], b: Sequence[str]) -> float:
    """WER giữa hai chuỗi token đã chuẩn hoá, chia cho độ dài dài hơn (đối xứng, không phụ thuộc ai là reference)."""
    if not a or not b:
        return float("inf")
    errs = sum(1 for t, _, _ in align(list(a), list(b)) if t != "C")
    return errs / max(len(a), len(b))


def has_span(seq: Sequence[str], span: Sequence[str]) -> bool:
    """Chuỗi token `span` có trong `seq` không: đúng chuỗi token liền (contains), hoặc một đoạn token liền của seq ghép
    bỏ dấu cách trùng với span ghép bỏ dấu cách ("set up" ~ "setup", "a p i" ~ "api")."""
    if contains(seq, span):
        return True
    key = "".join(span)
    for n in range(1, len(span) + 3):
        if any("".join(seq[i:i + n]) == key for i in range(len(seq) - n + 1)):
            return True
    return False


def tag_text(text: str, cat: Catalog, vi_words=None, term_dec=None) -> str:
    """Gắn <en> tự động cho text thô của một ASR bằng đúng cơ chế đã gắn nhãn nháp (converters.common.auto_tag): catalog,
    quyết định theo thuật ngữ, vi_abbrev / vi_loanwords (không gắn), heuristic âm tiết. vi_words / term_dec = None: đọc
    configs/ như converter; truyền frozenset() / {} để tắt (test)."""
    from gipbench.core.tagging import auto_tag
    return auto_tag(text, cat, vi_words=vi_words, term_dec=term_dec)["text_tagged"]


def mine_hard(teachers: Sequence[Optional[str]], student: Optional[str], cat: Catalog, max_pair_wer: float = 0.15,
              max_student_vi_wer: float = 0.30, max_en_frac: Optional[float] = None, leak=None,
              vi_words=None, term_dec=None) -> Dict:
    """Câu KHÓ cho trò: các thầy đồng thuận, có span tiếng Anh mà trò không có.

    teachers: hyp của >= 2 thầy (thầy đầu = thầy chính, nguồn của nhãn), student: hyp Gipformer cùng audio (chuỗi, có thể
    rỗng; None = chưa có: chỉ kiểm các điều kiện phía thầy, đạt thì trả status "need_student"). Điều kiện,
    theo thứ tự HARD_STAGES (câu bị loại ở điều kiện đầu tiên không đạt, khoá ghi ở "reason"):
      hyp            có >= 2 thầy và mọi thầy có hyp không rỗng sau chuẩn hoá;
      pair           (a) pair_wer của MỌI cặp thầy (text chuẩn hoá strict) <= max_pair_wer;
      has_span       (b) text thầy chính, gắn <en> bằng tag_text, có ít nhất một span;
      span_confirmed (b) MỌI span đó có đúng chuỗi token trong hyp của MỌI thầy khác (contains, không ghép);
      leak           leak(text_thầy_chính) (callable -> lý do, "" = qua) không chặn: gắn bộ chặn rò rỉ của người gọi
                     (LeakGuard.reason: reserved_b3, test_only, bench, danh sách chặn);
      en_frac        tỉ lệ token trong span / tổng token <= max_en_frac (None = không kiểm);
      student_vi     (d) WER phần tiếng Việt (ngoài span) của trò so với nhãn thầy chính <= max_student_vi_wer.
    Sau đó (c): span bị "thiếu" ở trò = span mà has_span(trò, span) sai. Có ít nhất một span thiếu -> "hard"; trò có đủ
    mọi span -> "easy" (đạt các điều kiện chất lượng, nhưng trò đã đúng: người gọi quyết có giữ một phần hay không).
    Trả về {"status": "hard"|"easy"|"need_student"|"reject", "reason", "text_tagged", "spans", "teacher_spans" (span thiếu ở trò),
    "student_spans" (đoạn token trò căn với từng span thiếu, song song teacher_spans), "teacher_pair_wer" (lớn nhất),
    "student_vi_wer", "en_frac"}; các trường chưa tính tới là None / rỗng.
    """
    out = {"status": "reject", "reason": "", "text_tagged": None, "spans": [], "teacher_spans": [],
           "student_spans": [], "teacher_pair_wer": None, "student_vi_wer": None, "en_frac": None}
    t_toks = [normalize_hyp(h or "", cat, "strict") for h in teachers]
    if len(teachers) < 2 or not all(t_toks):
        out["reason"] = "hyp"
        return out
    pw = [pair_wer(t_toks[i], t_toks[j]) for i in range(len(t_toks)) for j in range(i + 1, len(t_toks))]
    out["teacher_pair_wer"] = round(max(pw), 4)
    if max(pw) > max_pair_wer:
        out["reason"] = "pair"
        return out
    tagged = tag_text(teachers[0], cat, vi_words, term_dec)
    toks, spans = normalize_ref(tagged, cat, "strict")
    out["text_tagged"] = tagged
    out["spans"] = [" ".join(s) for s in spans]
    if not spans:
        out["reason"] = "has_span"
        return out
    if not all(contains(h, s) for s in spans for h in t_toks[1:]):
        out["reason"] = "span_confirmed"
        return out
    why = leak(plain(tagged)) if leak else ""
    if why:
        out["reason"] = "leak"
        out["leak"] = why
        return out
    ref_tok, ref_span = [t.tok for t in toks], [t.span for t in toks]
    out["en_frac"] = round(sum(1 for s in ref_span if s >= 0) / len(ref_tok), 3)
    if max_en_frac is not None and out["en_frac"] > max_en_frac:
        out["reason"] = "en_frac"
        return out
    if student is None:  # chưa có hyp trò nhưng đã qua mọi điều kiện phía thầy: đáng để chạy decode trò
        out["status"] = "need_student"
        return out
    s_toks = normalize_hyp(student, cat, "strict")
    seg, e_vi, n_vi = span_segments(ref_tok, ref_span, s_toks)
    out["student_vi_wer"] = round(e_vi / n_vi, 4) if n_vi else float("inf")
    if not n_vi or e_vi / n_vi > max_student_vi_wer:
        out["reason"] = "student_vi"
        return out
    missed = [i for i, s in enumerate(spans) if not has_span(s_toks, s)]
    out["teacher_spans"] = [" ".join(spans[i]) for i in missed]
    out["student_spans"] = [" ".join(seg.get(i, [])) for i in missed]
    out["status"] = "hard" if missed else "easy"
    return out
