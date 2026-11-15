"""Chốt tag theo THUẬT NGỮ thay cho duyệt từng câu (decisions §12.4), cho nguồn có text do người gõ (trusted_text).

configs/term_decisions.tsv (TAB, có header; dòng '#' bỏ qua):
  term      thuật ngữ, chuẩn hoá như basic_tokens (viết thường, bỏ dấu câu); nhiều từ được
  decision  en        luôn gắn <en> (auto_tag coi như mục catalog, thắng vi_abbrev / vi_loanwords)
            vi        không gắn <en>: đọc kiểu Việt / từ Việt (auto_tag coi như vi_words, thắng catalog)
            drop_tag  không gắn <en> (không phải từ Anh, cũng không đưa vào vi_abbrev: vd lỗi gõ, mã vô nghĩa)
            (trống)   chưa chốt: câu chứa thuật ngữ vẫn chờ
  note      ghi chú tự do

Một câu trusted_text được build_bench_v0.py --accept trusted-text nhận khi:
  - mọi thuật ngữ còn cờ của câu (needed_terms) đã có decision;
  - không có cờ chặn (cờ không gắn với thuật ngữ, vd numbers_disagree, email_or_url);
  - nhãn của câu khớp với quyết định hiện tại (không "cũ"): span <en> nào có decision vi / drop_tag, hoặc thuật ngữ
    decision en còn cờ mà chưa thành span, nghĩa là converter chưa chạy lại sau khi sửa term_decisions.tsv.
Thuật ngữ còn cờ: giá trị của cờ abbr_reading / maybe_name / vi_word / suspect / name, cộng span chứa từ mơ hồ khi có
cờ ambiguous_joined. Span heuristic khác (không cờ, vd "deep learning") và span catalog được coi là đã rõ.
"""
import csv
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from lib.textnorm import Catalog, basic_tokens, normalize_ref

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TERM_DECISIONS = ROOT / "configs" / "term_decisions.tsv"
DECISIONS = ("en", "vi", "drop_tag")
NO_TAG = ("vi", "drop_tag")
TERM_FLAGS = ("abbr_reading", "maybe_name", "vi_word", "suspect", "name")
INFO_FLAGS = {"heuristic_span", "ambiguous_joined"}
_EMPTY = Catalog()
_CACHE: Dict[str, Dict[str, str]] = {}


def norm_term(s: str) -> str:
    return " ".join(basic_tokens(unicodedata.normalize("NFC", s or "")))


def load_term_decisions(path=None) -> Dict[str, str]:
    """{term: decision} các dòng đã chốt (decision không trống). path None = mặc định; '' = tắt (dict rỗng)."""
    if path == "" or path is False:
        return {}
    p = Path(path) if path else DEFAULT_TERM_DECISIONS
    key = str(p.resolve()) if p.exists() else str(p)
    if key in _CACHE:
        return _CACHE[key]
    out: Dict[str, str] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            rows = csv.DictReader((l for l in f if l.strip() and not l.startswith("#")), delimiter="\t")
            for r in rows:
                t = norm_term(r.get("term") or "")
                d = (r.get("decision") or "").strip().lower()
                if not t or not d:
                    continue
                if d not in DECISIONS:
                    raise ValueError(f"{p}: decision lạ '{d}' cho '{t}' (chỉ nhận {', '.join(DECISIONS)} hoặc trống)")
                out[t] = d
    _CACHE[key] = out
    return out


def read_term_rows(path) -> List[Dict[str, str]]:
    """Mọi dòng của file quyết định (kể cả decision trống), giữ thứ tự."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [dict(r) for r in csv.DictReader((l for l in f if l.strip() and not l.startswith("#")), delimiter="\t")]


def write_term_rows(path, rows: Iterable[Dict[str, str]], header_note: str = ""):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write("# Quyết định gắn tag theo thuật ngữ (decisions §12.4). Xem lib/term_decisions.py.\n"
                "# decision: en | vi | drop_tag | (trống = chưa chốt). Sửa file này xong thì chạy lại converter.\n")
        if header_note:
            f.write(f"# {header_note}\n")
        f.write("term\tdecision\tnote\n")
        for r in rows:
            f.write(f"{norm_term(r['term'])}\t{(r.get('decision') or '').strip()}\t{(r.get('note') or '').strip()}\n")
    _CACHE.clear()


def spans_of(text_tagged: str) -> List[str]:
    _, spans = normalize_ref(text_tagged or "", _EMPTY)
    return [" ".join(s) for s in spans]


def needed_terms(row: Dict, ambig: Optional[Set[str]] = None) -> Tuple[Dict[str, Set[str]], List[str]]:
    """-> ({thuật ngữ: {loại cờ}}, [cờ chặn]). ambig = tập từ mơ hồ của auto_tag (mặc định AMBIG_EN)."""
    from converters.common import is_vietnamese_token
    if ambig is None:
        from converters.common import AMBIG_EN as ambig
    flags = (row.get("auto_tag") or {}).get("flags") or []
    terms: Dict[str, Set[str]] = {}
    blocking = []
    for f in flags:
        kind, _, val = f.partition(":")
        if kind in TERM_FLAGS and val:
            t = norm_term(val)
            if kind == "vi_word" and all(is_vietnamese_token(w) for w in t.split()):
                continue  # âm tiết Việt hợp lệ (co, di, va): không có danh sách cũng không bao giờ bị gắn <en>
            if t:
                terms.setdefault(t, set()).add(kind)
        elif kind in INFO_FLAGS:
            continue
        else:
            blocking.append(f)
    if "ambiguous_joined" in flags:
        for sp in spans_of(row.get("text_tagged", "")):
            toks = sp.split()
            if len(toks) > 1 and any(t in ambig for t in toks):
                terms.setdefault(sp, set()).add("ambiguous_joined")
    return terms, blocking


def term_status(row: Dict, dec: Dict[str, str]) -> Dict[str, List[str]]:
    """{"pending": thuật ngữ chưa chốt, "blocking": cờ chặn, "stale": thuật ngữ mà nhãn câu chưa theo quyết định}."""
    terms, blocking = needed_terms(row)
    spans = spans_of(row.get("text_tagged", ""))
    span_set = set(spans)
    stale = sorted({s for s in spans if dec.get(s) in NO_TAG})
    for t, kinds in terms.items():
        if dec.get(t) == "en" and t not in span_set and kinds & {"vi_word", "suspect", "maybe_name"}:
            stale.append(t)
    pending = sorted(t for t in terms if t not in dec)
    return {"pending": pending, "blocking": sorted(blocking), "stale": sorted(set(stale))}


def resolved(row: Dict, dec: Dict[str, str]) -> bool:
    st = term_status(row, dec)
    return not (st["pending"] or st["blocking"] or st["stale"])
