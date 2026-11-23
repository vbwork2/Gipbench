"""Normalize FPT transcripts while preserving explicitly configured acronyms."""

from collections.abc import Collection
import re
import unicodedata

from gipbench.core.scoring import en_like
from gipbench.core.textnorm import numbers_to_words


WORD = re.compile(r"[^\W_]+(?:\+)?", re.UNICODE)


def tokens(text: str, case: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", numbers_to_words(text).casefold())
    words = WORD.findall(normalized)
    return [word.upper() for word in words] if case == "upper" else words


def cs_positions(words: list[str], terms: Collection[str]) -> list[int]:
    known = {term.casefold() for term in terms}
    return [
        index for index, word in enumerate(words)
        if word.casefold() in known or en_like(word.casefold())
    ]
