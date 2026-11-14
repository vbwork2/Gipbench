"""Chặn rò rỉ bench -> train dùng chung cho converter và công cụ train (Spec mục 5).

Một câu train bị loại nếu có: speaker hoặc source trùng bench, text (sau chuẩn hoá + catalog) trùng một câu bench,
hoặc thuật ngữ reserved_b3. Hash audio không kiểm ở đây (cần đọc audio): validate.py --train làm việc đó.
"""
from typing import Dict, Iterable, Optional, Sequence

from lib.textnorm import Catalog, basic_tokens, plain_text


def norm_text(text: str, cat: Catalog) -> str:
    return " ".join(cat.apply(basic_tokens(plain_text(text or ""))))


class LeakGuard:
    def __init__(self, bench_rows: Iterable[Dict], cat: Catalog, splits: Optional[Sequence[str]] = None):
        self.cat = cat
        self.reserved = {t for t, v in cat.terms.items() if v.reserved_b3}
        self.spk, self.src, self.txt = set(), set(), set()
        self.n_bench = 0
        for r in bench_rows:
            if splits and r.get("split") not in splits:
                continue
            self.n_bench += 1
            self.spk.add(r.get("speaker"))
            self.src.add(r.get("source"))
            self.txt.add(norm_text(r.get("text_tagged") or r.get("text"), cat))

    def reason(self, row: Dict, text: str) -> str:
        """Lý do loại (chuỗi rỗng = không rò rỉ)."""
        if row.get("speaker") in self.spk:
            return "speaker trùng bench"
        if row.get("source") in self.src:
            return "source trùng bench"
        toks = self.cat.apply(basic_tokens(plain_text(text or "")))
        if self.reserved and any(t in self.reserved for t in self.cat.find_terms(toks)):
            return "có thuật ngữ reserved_b3"
        if " ".join(toks) in self.txt:
            return "text trùng câu bench"
        return ""
