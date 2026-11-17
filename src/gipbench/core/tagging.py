"""Tự gắn tag <en> (bản nháp cho người duyệt) và nhận diện âm tiết tiếng Việt: dùng chung cho dataset (converter)
và core (term_decisions). Chuyển từ converters/common.py, giữ nguyên logic.

  C = khớp catalog hoặc danh sách thuật ngữ đi kèm dataset  -> tin cậy cao
  H = token không thể là âm tiết tiếng Việt (không dấu, sai cấu trúc âm tiết) -> cần duyệt
  A = từ vừa hợp lệ tiếng Việt vừa là từ Anh hay gặp (the, to, chat...) nằm cạnh từ Anh -> ghép vào span, cần duyệt
"""
import re
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from gipbench.core.paths import REPO as _ROOT
from gipbench.core.textnorm import Catalog, basic_tokens

# ------------------------------------------------------------------ âm tiết tiếng Việt (dạng không dấu)
_VI_SYL = re.compile(
    r"^(ngh|ng|nh|ch|gh|gi|kh|ph|qu|th|tr|[bcdghklmnprstvx])?"
    r"(oai|oay|uya|ai|ao|au|ay|eo|ia|iu|oa|oe|oi|ua|ui|uy|a|e|i|o|u|y)"
    r"(ch|ng|nh|[cmnpt])?$")
_VI_MARK = re.compile(r"[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]")
# từ Anh hay gặp nhưng trùng cấu trúc âm tiết Việt: chỉ ghép vào span khi đứng cạnh từ Anh khác
AMBIG_EN = set("""the to in on at it up no so go he be my me do can man men hit kit bit bot net mic pic tip top pop
cup but not get set let put cut tap map gap cap hot chat chip pin pen ten pay pet mac sim ai an long""".split())
# đứng một mình vẫn đáng ngờ là từ Anh -> chỉ gắn cờ cho người duyệt
SUSPECT_ALONE = set("chat bot net mic chip pin sim top hot pay mac ai".split())
_EDGE = re.compile(r"^([^\w<>]*)(.*?)([^\w<>]*)$", re.S)


# vần kết thúc bằng bán nguyên âm: không bao giờ có phụ âm cuối (chain, main, boat... không phải tiếng Việt)
_GLIDE_NUC = {"ai", "ao", "au", "ay", "eo", "ia", "iu", "oi", "ui", "ua", "oai", "oay", "uya"}


def is_vietnamese_token(tok: str) -> bool:
    """True nếu token có thể là một âm tiết tiếng Việt (viết đúng chính tả, có dấu thanh khi cần).

    Token không có dấu nào chỉ hợp lệ khi đúng cấu trúc âm tiết và còn thoả hai luật chính tả:
      - âm tiết tận cùng -p, -t, -c, -ch chỉ mang thanh sắc hoặc nặng, nên bản không dấu
        (chip, kit, mac, net, hot, tech, op) không phải tiếng Việt;
      - vần có bán nguyên âm cuối (ai, ao, oi, ...) không có phụ âm cuối; tiếng Việt viết -inh, không có -ing.
    Giả định text gốc có dấu đầy đủ (G-Group, ViMedCSS, VIVOS đều vậy).
    """
    t = unicodedata.normalize("NFC", tok.lower())
    if _VI_MARK.search(t) or t.isdigit():
        return True
    m = _VI_SYL.match(t)
    if not m:
        return False
    _, nuc, fin = m.groups()
    if fin and (fin in ("p", "t", "c", "ch") or nuc in _GLIDE_NUC or (nuc == "i" and fin == "ng")):
        return False
    return True


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _fuzzy(tok: str, term: str) -> bool:
    if tok == term:
        return True
    if min(len(tok), len(term)) >= 5 and (tok.startswith(term) or term.startswith(tok)):
        return True
    return min(len(tok), len(term)) >= 6 and _edit(tok, term) <= 2


VI_WORD_FILES = [_ROOT / "configs" / "dataset" / "labeling" / "vi_abbrev.txt", _ROOT / "configs" / "dataset" / "labeling" / "vi_loanwords.txt"]
_VI_WORDS_CACHE: Dict[Tuple[str, ...], frozenset] = {}


def load_vi_words(paths: Optional[Sequence] = None, same_surface: bool = True) -> frozenset:
    """Đọc cột đầu của các file danh sách (bỏ dòng '#' và dòng header), viết thường, NFC.

    paths = None -> các file mặc định trong configs/ (file nào chưa có thì bỏ qua); [] -> tập rỗng.
    same_surface: các dòng kind = same_surface của vi_loanwords.txt (vitamin, protein: trùng mặt chữ tiếng Anh) là
    KHÔNG EN ở MỌI bộ (decisions §12); False chỉ để so sánh / tái lập số cũ (trước §12 chỉ áp cho ViMedCSS).
    """
    paths = VI_WORD_FILES if paths is None else paths
    key = tuple(str(p) for p in paths) + (str(same_surface),)
    if key not in _VI_WORDS_CACHE:
        words = set()
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            for line in open(p, encoding="utf-8"):
                cols = line.rstrip("\n").split("\t")
                w = cols[0].strip()
                if not w or w.startswith("#") or w in ("abbr", "word"):
                    continue
                if not same_surface and len(cols) > 1 and cols[1].strip() == "same_surface":
                    continue
                words.add(unicodedata.normalize("NFC", w).lower())
        _VI_WORDS_CACHE[key] = frozenset(words)
    return _VI_WORDS_CACHE[key]


def auto_tag(text: str, catalog: Optional[Catalog] = None, known_terms: Sequence[str] = (),
             vi_words: Optional[Iterable[str]] = None, heuristic: bool = True,
             term_dec: Optional[Dict[str, str]] = None) -> Dict:
    """Trả về {"text_tagged", "confidence": catalog|heuristic|none, "flags": [...], "spans": [...]}.

    vi_words: từ không bao giờ gắn <en> trừ khi khớp catalog (viết tắt đọc kiểu Việt, từ mượn); None = đọc
    configs/dataset/labeling/vi_abbrev.txt + configs/dataset/labeling/vi_loanwords.txt. Thuật ngữ đi kèm dataset (known_terms) mà mặt chữ khớp
    nằm trong vi_words cũng không được gắn. heuristic=False: chỉ gắn theo catalog + known_terms.
    term_dec: quyết định theo thuật ngữ {term: en|vi|drop_tag} (src/gipbench/core/term_decisions.py); None = configs/dataset/labeling/term_decisions.tsv,
    {} = tắt. en = gắn như mục catalog (thắng vi_words); vi / drop_tag = không bao giờ gắn (thắng catalog).
    """
    from gipbench.core.term_decisions import NO_TAG, load_term_decisions
    vi_words = load_vi_words() if vi_words is None else vi_words
    term_dec = load_term_decisions() if term_dec is None else term_dec
    dec_en = {tuple(t.split()) for t, d in term_dec.items() if d == "en"}
    dec_no = {tuple(t.split()) for t, d in term_dec.items() if d in NO_TAG}
    raw = text.split()
    parts = [_EDGE.match(w).groups() for w in raw]
    core = [unicodedata.normalize("NFC", p[1]).lower() for p in parts]
    n = len(core)
    lab = [""] * n
    span_id = [-1] * n
    spans: List[Tuple[int, int, str]] = []  # (start, end, kiểu)

    # 0) quyết định vi / drop_tag theo thuật ngữ (n-gram dài nhất): nhãn V, thắng catalog
    multi_no = set()
    if dec_no:
        mx = max(len(k) for k in dec_no)
        i = 0
        while i < n:
            for L in range(min(mx, n - i), 0, -1):
                if tuple(core[i:i + L]) in dec_no:
                    for j in range(i, i + L):
                        lab[j] = "V"
                        if L > 1:
                            multi_no.add(j)  # cụm nhiều từ đã chốt: không gắn cờ vi_word cho từng từ
                    i += L - 1
                    break
            i += 1
    # 1) catalog + quyết định en (n-gram dài nhất, kể cả biến thể chính tả) rồi thuật ngữ đi kèm dataset
    keys = set(dec_en)
    max_n = max([len(k) for k in keys] or [1])
    if catalog and catalog.strict_map:
        keys |= set(catalog.strict_map)
        max_n = max(len(k) for k in keys)
    known = [tuple(basic_tokens(k)) for k in known_terms if k and k.strip()]
    known = [k for k in known if k]
    i = 0
    while i < n:
        hit = 0
        for L in range(min(max_n, n - i), 0, -1):
            if tuple(core[i:i + L]) in keys and not any(lab[i:i + L]):
                hit = L
                break
        if not hit:
            for k in sorted(known, key=len, reverse=True):
                L = len(k)
                if i + L <= n and not any(lab[i:i + L]) and all(_fuzzy(core[i + j], k[j]) for j in range(L)) and not all(
                        is_vietnamese_token(core[i + j]) and core[i + j] != k[j] for j in range(L)) \
                        and " ".join(core[i:i + L]) not in vi_words:
                    hit = L
                    break
        if hit and not (catalog and catalog.strict_map and tuple(core[i:i + hit]) in catalog.strict_map)                 and tuple(core[i:i + hit]) in dec_en:
            # quyết định en (không phải catalog): nhãn D, được ghép với từ Anh liền kề như heuristic (stop loss, new york)
            for j in range(i, i + hit):
                lab[j] = "D"
            i += hit
        elif hit:
            for j in range(i, i + hit):
                lab[j] = "C"
            spans.append((i, i + hit, "C"))
            i += hit
        else:
            i += 1
    # 1b) viết tắt đọc kiểu Việt, từ mượn có trong từ điển: nhãn V (không bao giờ thành span, không làm mốc ghép)
    for i in range(n):
        if not lab[i] and core[i] in vi_words:
            lab[i] = "V"
    # 2) heuristic theo cấu trúc âm tiết
    for i in range(n):
        if lab[i] or not core[i] or not heuristic:
            continue
        if not is_vietnamese_token(core[i]) and re.search(r"[a-z]", core[i]) and len(core[i]) > 1:
            lab[i] = "H"
    # 2b) chuỗi chữ cái đơn (a p i) có ít nhất một phụ âm
    i = 0
    while i < n and heuristic:
        j = i
        while j < n and not lab[j] and len(core[j]) == 1 and core[j].isalpha() and not _VI_MARK.search(core[j]):
            j += 1
        if j - i >= 2 and any(c not in "aeiouy" for c in core[i:j]):
            for k in range(i, j):
                lab[k] = "H"
        i = max(j, i + 1)
    # 3) từ mơ hồ đứng cạnh từ Anh
    # vi_word:<từ> = từ chữ Latin được coi là tiếng Việt nhờ danh sách: câu B5 có cờ này phải qua người duyệt
    flags = {f"vi_word:{core[i]}" for i in range(n) if lab[i] == "V" and i not in multi_no}
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not lab[i] and core[i] in AMBIG_EN and (
                    (i > 0 and lab[i - 1] in "HAD" and lab[i - 1]) or (i + 1 < n and lab[i + 1] in "HAD" and lab[i + 1])):
                lab[i] = "A"
                flags.add("ambiguous_joined")
                changed = True
    for i in range(n):
        if not lab[i] and core[i] in SUSPECT_ALONE:
            flags.add(f"suspect:{core[i]}")
    # gộp H/A/D liền nhau thành span (D = quyết định en; span chỉ gồm D có kiểu D: đã chốt, không cờ abbr_reading)
    i = 0
    while i < n:
        if lab[i] in ("H", "A", "D"):
            j = i
            while j < n and lab[j] in ("H", "A", "D"):
                j += 1
            spans.append((i, j, "H" if any(lab[k] != "D" for k in range(i, j)) else "D"))
            i = j
        else:
            i += 1
    spans.sort()
    # viết tắt ngắn đoán bằng heuristic: chưa biết đọc kiểu Anh hay kiểu Việt -> người duyệt nghe rồi quyết
    for s, e, k in spans:
        if k == "H" and e - s == 1 and 2 <= len(core[s]) <= 4 and core[s].isalpha():
            flags.add(f"abbr_reading:{core[s]}")
    out, pos = [], 0
    for s, e, _ in spans:
        out += raw[pos:s]
        pre, post = parts[s][0], parts[e - 1][2]
        inner = " ".join([parts[s][1] + (parts[s][2] if e - s > 1 else "")] +
                         [raw[k] for k in range(s + 1, e - 1)] +
                         ([parts[e - 1][0] + parts[e - 1][1]] if e - s > 1 else []))
        out.append(f"{pre}<en>{inner}</en>{post}")
        pos = e
    out += raw[pos:]
    kinds = {k for _, _, k in spans}
    if "H" in kinds:
        flags.add("heuristic_span")
    conf = "none" if not spans else ("heuristic" if "H" in kinds else "catalog")
    return {"text_tagged": " ".join(out), "confidence": conf, "flags": sorted(flags),
            "spans": [" ".join(core[s:e]) for s, e, _ in spans], "span_kinds": [k for _, _, k in spans]}


def add_vi_words_arg(ap):
    ap.add_argument("--vi-words", nargs="*", default=None,
                    help="file danh sách từ không gắn <en> (mặc định configs/dataset/labeling/vi_abbrev.txt configs/dataset/labeling/vi_loanwords.txt; "
                         "truyền rỗng để tắt)")
    ap.add_argument("--term-decisions", default=None,
                    help="quyết định gắn tag theo thuật ngữ (mặc định configs/dataset/labeling/term_decisions.tsv; '' = tắt)")
