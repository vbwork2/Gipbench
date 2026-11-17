"""Suy domain từ text cho dữ liệu train chưa có nhãn domain (decisions §10, §12.6).

Điểm của domain d = số lần gặp keyword của d trong configs/dataset/labeling/domain_keywords.tsv (n-gram token, tối đa `cap` lần mỗi
keyword) + số thuật ngữ <en> của câu có category catalog = d. Domain thắng khi điểm >= min_score và hơn domain thứ hai
>= margin; không thì 'other'. Kết quả ghi domain_source = inferred (domain cũ giữ ở domain_orig).
"""
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gipbench.core.textnorm import Catalog, basic_tokens, normalize_ref

from gipbench.core.paths import REPO as ROOT
DEFAULT_KEYWORDS = ROOT / "configs" / "dataset" / "labeling" / "domain_keywords.tsv"
# category catalog -> domain (category không có ở đây, vd brand / acronym / name, không cộng điểm)
CATEGORY_DOMAIN = {c: c for c in ("tech", "finance", "edu", "medical", "pubadmin", "business", "daily", "entertainment")}


def load_keywords(path=None) -> Dict[Tuple[str, ...], str]:
    out = {}
    with open(path or DEFAULT_KEYWORDS, encoding="utf-8") as f:
        for r in csv.DictReader((l for l in f if l.strip() and not l.startswith("#")), delimiter="\t"):
            k = tuple(basic_tokens(r["keyword"]))
            if k:
                out[k] = r["domain"].strip()
    return out


class DomainInferer:
    def __init__(self, keywords: Optional[Dict[Tuple[str, ...], str]] = None, catalog: Optional[Catalog] = None,
                 min_score: float = 2, margin: float = 1, cap: int = 3):
        self.kw = keywords if keywords is not None else load_keywords()
        self.max_n = max([len(k) for k in self.kw] or [1])
        self.cat = catalog or Catalog()
        self.min_score, self.margin, self.cap = min_score, margin, cap

    def scores(self, text_tagged: str) -> Counter:
        toks, spans = normalize_ref(text_tagged or "", Catalog())
        words = [t.tok for t in toks]
        hit = Counter()
        sc = Counter()
        i = 0
        while i < len(words):  # n-gram dài nhất trước, không chồng nhau
            for n in range(min(self.max_n, len(words) - i), 0, -1):
                k = tuple(words[i:i + n])
                if k in self.kw:
                    hit[k] += 1
                    if hit[k] <= self.cap:
                        sc[self.kw[k]] += 1
                    i += n - 1
                    break
            i += 1
        for sp in spans:
            t = self.cat.terms.get(tuple(sp)) if self.cat.terms else None
            d = CATEGORY_DOMAIN.get(t.category) if t and t.category else None
            if d:
                sc[d] += 1
        return sc

    def infer(self, text_tagged: str) -> Tuple[str, Counter]:
        sc = self.scores(text_tagged)
        top: List[Tuple[str, float]] = sc.most_common(2)
        if not top or top[0][1] < self.min_score:
            return "other", sc
        if len(top) > 1 and top[0][1] - top[1][1] < self.margin:
            return "other", sc
        return top[0][0], sc
