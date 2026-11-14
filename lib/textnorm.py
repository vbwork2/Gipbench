"""Chuẩn hoá text cho bench code-switching Việt-Anh.

Cùng một pipeline được áp cho reference và hypothesis:
  NFC -> lowercase -> số thành chữ (tiếng Việt) -> bỏ dấu câu -> tách token -> bảng chuẩn (catalog).

Reference dùng `text_tagged`, trong đó từ tiếng Anh nằm trong <en>...</en>.
Ranh giới span được giữ lại để tính CS-WER và N-WER.
"""
import csv
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ----------------------------------------------------------------------------- số -> chữ
_DIG = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
_GROUP = ["", "nghìn", "triệu", "tỷ"]


def _read_triple(n: int, full: bool) -> List[str]:
    tram, chuc, dv = n // 100, (n // 10) % 10, n % 10
    out: List[str] = []
    if tram or full:
        out += [_DIG[tram], "trăm"]
    if chuc == 0:
        if dv and (tram or full):
            out.append("linh")
    elif chuc == 1:
        out.append("mười")
    else:
        out += [_DIG[chuc], "mươi"]
    if dv:
        if dv == 1 and chuc >= 2:
            out.append("mốt")
        elif dv == 5 and chuc >= 1:
            out.append("lăm")
        else:
            out.append(_DIG[dv])
    return out


def read_int(n: int) -> List[str]:
    if n == 0:
        return ["không"]
    groups = []
    while n > 0:
        groups.append(n % 1000)
        n //= 1000
    words: List[str] = []
    top = len(groups) - 1
    for gi in range(top, -1, -1):
        g = groups[gi]
        if g == 0:
            continue
        words += _read_triple(g, full=(gi != top))
        if gi <= 3:
            if _GROUP[gi]:
                words.append(_GROUP[gi])
        else:
            words += [_GROUP[gi - 3], "tỷ"]
    return words


_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _read_number_token(tok: str) -> List[str]:
    # 1.000.000 -> nghìn phân cách; 3,5 hoặc 3.5 -> thập phân
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", tok):
        return read_int(int(tok.replace(".", "")))
    m = re.fullmatch(r"(\d+)[.,](\d+)", tok)
    if m:
        frac = m.group(2)
        frac_words = read_int(int(frac)) if not frac.startswith("0") else [_DIG[int(c)] for c in frac]
        return read_int(int(m.group(1))) + ["phẩy"] + frac_words
    if tok.isdigit():
        if len(tok) > 1 and tok.startswith("0"):  # số điện thoại, mã: đọc từng chữ số
            return [_DIG[int(c)] for c in tok]
        return read_int(int(tok))
    return [_DIG[int(c)] if c.isdigit() else c for c in tok]


def numbers_to_words(text: str) -> str:
    text = re.sub(r"(?<=\d)\s*%", " phần trăm", text)
    # tách chữ-số dính nhau: iphone15 -> iphone 15, 4g -> 4 g
    text = re.sub(r"(?<=[^\W\d_])(?=\d)|(?<=\d)(?=[^\W\d_])", " ", text)
    return _NUM.sub(lambda m: " " + " ".join(_read_number_token(m.group(0))) + " ", text)


# ----------------------------------------------------------------------------- chuẩn hoá cơ bản
_KEEP = re.compile(r"[^\w\s']", re.UNICODE)


def basic_tokens(text: str) -> List[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = numbers_to_words(text)
    text = text.replace("-", " ").replace("_", " ")
    text = _KEEP.sub(" ", text)
    toks = []
    for t in text.split():
        t = t.strip("'")
        if t:
            toks.append(t)
    return toks


# ----------------------------------------------------------------------------- catalog / bảng chuẩn
@dataclass
class Term:
    canonical: Tuple[str, ...]
    category: str = ""
    reserved_b3: bool = False


@dataclass
class Catalog:
    terms: Dict[Tuple[str, ...], Term] = field(default_factory=dict)
    strict_map: Dict[Tuple[str, ...], Tuple[str, ...]] = field(default_factory=dict)
    lenient_map: Dict[Tuple[str, ...], Tuple[str, ...]] = field(default_factory=dict)
    max_n: int = 1

    @classmethod
    def load(cls, path: Optional[str]) -> "Catalog":
        cat = cls()
        if not path:
            return cat
        with open(path, encoding="utf-8") as f:
            rows = csv.DictReader((l for l in f if not l.startswith("#")), delimiter="\t")
            for r in rows:
                canon = tuple(basic_tokens(r["term"]))
                if not canon:
                    continue
                cat.terms[canon] = Term(canon, (r.get("category") or "").strip(),
                                        (r.get("reserved_b3") or "0").strip() in ("1", "true", "yes"))
                cat.strict_map[canon] = canon
                cat.lenient_map[canon] = canon
                for v in (r.get("variants") or "").split("|"):
                    vt = tuple(basic_tokens(v))
                    if vt:
                        cat.strict_map[vt] = canon
                        cat.lenient_map[vt] = canon
                for v in (r.get("phonetic") or "").split("|"):
                    vt = tuple(basic_tokens(v))
                    if vt:
                        cat.lenient_map[vt] = canon
        cat.max_n = max([len(k) for k in cat.lenient_map] + [1])
        return cat

    def apply(self, toks: List[str], mode: str = "strict") -> List[str]:
        mp = self.lenient_map if mode == "lenient" else self.strict_map
        if not mp:
            return toks
        out, i = [], 0
        while i < len(toks):
            for n in range(min(self.max_n, len(toks) - i), 0, -1):
                key = tuple(toks[i:i + n])
                if key in mp:
                    out += list(mp[key])
                    i += n
                    break
            else:
                out.append(toks[i])
                i += 1
        return out

    def find_terms(self, toks: List[str]) -> List[Tuple[str, ...]]:
        """Các thuật ngữ catalog xuất hiện trong chuỗi token (khớp dài nhất, không chồng lấn)."""
        found, i = [], 0
        lens = sorted({len(k) for k in self.terms}, reverse=True)
        while i < len(toks):
            for n in lens:
                key = tuple(toks[i:i + n])
                if len(key) == n and key in self.terms:
                    found.append(key)
                    i += n
                    break
            else:
                i += 1
        return found


# ----------------------------------------------------------------------------- reference có tag
_TAG = re.compile(r"<en(?:\s[^>]*)?>(.*?)</en>", re.S)


@dataclass
class RefToken:
    tok: str
    span: int  # -1 = tiếng Việt, >=0 = chỉ số span tiếng Anh


def parse_tagged(text_tagged: str) -> List[Tuple[str, bool]]:
    segs, pos = [], 0
    for m in _TAG.finditer(text_tagged):
        if m.start() > pos:
            segs.append((text_tagged[pos:m.start()], False))
        segs.append((m.group(1), True))
        pos = m.end()
    if pos < len(text_tagged):
        segs.append((text_tagged[pos:], False))
    return segs


def tagged_well_formed(text_tagged: str) -> Optional[str]:
    opens = len(re.findall(r"<en(?:\s[^>]*)?>", text_tagged))
    closes = text_tagged.count("</en>")
    if opens != closes:
        return f"số <en> ({opens}) khác số </en> ({closes})"
    if re.search(r"<en(?:\s[^>]*)?>[^<]*<en", text_tagged):
        return "tag <en> lồng nhau"
    if re.search(r"<en(?:\s[^>]*)?>\s*</en>", text_tagged):
        return "span <en> rỗng"
    stray = re.sub(r"</?en(?:\s[^>]*)?>", "", text_tagged)
    if "<" in stray or ">" in stray:
        return "còn ký tự < > lạ"
    return None


def normalize_ref(text_tagged: str, catalog: Catalog, mode: str = "strict"
                  ) -> Tuple[List[RefToken], List[Tuple[str, ...]]]:
    """Trả về token reference (có nhãn span) và danh sách thuật ngữ (dạng chuẩn) của từng span."""
    toks: List[RefToken] = []
    spans: List[Tuple[str, ...]] = []
    for seg, is_en in parse_tagged(text_tagged):
        st = catalog.apply(basic_tokens(seg), mode)
        if not st:
            continue
        if is_en:
            sid = len(spans)
            spans.append(tuple(st))
            toks += [RefToken(t, sid) for t in st]
        else:
            toks += [RefToken(t, -1) for t in st]
    return toks, spans


def normalize_hyp(text: str, catalog: Catalog, mode: str = "strict") -> List[str]:
    return catalog.apply(basic_tokens(text), mode)


def plain_text(text_tagged: str) -> str:
    return re.sub(r"</?en(?:\s[^>]*)?>", "", text_tagged)


def model_case_text(text_tagged_or_plain: str, case: str = "upper", catalog: Optional[Catalog] = None) -> str:
    """Text dùng để TRAIN (không tag), đúng kiểu chữ của tokenizer (Gipformer: IN HOA)."""
    toks = basic_tokens(plain_text(text_tagged_or_plain))
    if catalog:
        toks = catalog.apply(toks, "strict")
    s = " ".join(toks)
    return s.upper() if case == "upper" else s
