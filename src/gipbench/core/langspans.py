"""Trường tuỳ chọn `language_spans` của manifest: tách 3 thuộc tính mà tag <en> cũ gộp làm một.

    "text_tagged": "bạn giúp mình <en>deploy</en> cái <en>backend</en> nhé",
    "language_spans": [
      {"text": "deploy",  "lexical_language": "en", "pronunciation_style": "vi_accented", "entity_type": null},
      {"text": "backend", "lexical_language": "en", "pronunciation_style": null,          "entity_type": "technical_term"}
    ]

- `text_tagged` + `<en>` vẫn là nguồn chính cho scorer cũ (CS-WER); `language_spans` (nếu có) phải khớp 1-1, theo thứ
  tự, với các span `<en>` (cùng text sau NFC, bỏ khoảng trắng thừa).
- lexical_language   : ngôn ngữ từ vựng của span (en, vi, mixed, other)
- pronunciation_style: CHỈ gán sau khi NGHE (en_native, vi_accented, spelled_letters, mixed); null = chưa nghe. Không suy
                       từ chính tả.
- entity_type        : null hoặc brand, product, person_name, org, acronym, technical_term, medical_term, place, other
  (tách "brand Latin" khỏi "từ tiếng Anh", vd `zalo` là brand nhưng không phải từ vựng tiếng Anh).
- Không chứa thông tin cá nhân của người nói.
"""
import unicodedata
from typing import Dict, List

from gipbench.core.textnorm import parse_tagged

LEXICAL = {"en", "vi", "mixed", "other"}
PRONUNCIATION = {None, "en_native", "vi_accented", "spelled_letters", "mixed"}
ENTITY = {None, "brand", "product", "person_name", "org", "acronym", "technical_term", "medical_term", "place", "other"}
ALLOWED_KEYS = {"text", "lexical_language", "pronunciation_style", "entity_type", "note"}


def _norm(s: str) -> str:
    return " ".join(unicodedata.normalize("NFC", s or "").split())


def check_language_spans(row: Dict) -> List[str]:
    """Danh sách lỗi (rỗng = hợp lệ hoặc không có trường language_spans)."""
    ls = row.get("language_spans")
    if ls is None:
        return []
    if not isinstance(ls, list):
        return ["language_spans phải là list"]
    tagged = [_norm(seg) for seg, is_en in parse_tagged(row.get("text_tagged") or "") if is_en]
    errs = []
    if len(ls) != len(tagged):
        errs.append(f"language_spans có {len(ls)} mục nhưng text_tagged có {len(tagged)} span <en>")
    for i, s in enumerate(ls):
        if not isinstance(s, dict):
            errs.append(f"span {i}: phải là object")
            continue
        extra = set(s) - ALLOWED_KEYS
        if extra:
            errs.append(f"span {i}: khoá lạ {sorted(extra)}")
        if i < len(tagged) and _norm(s.get("text")) != tagged[i]:
            errs.append(f"span {i}: text '{s.get('text')}' khác span <en> '{tagged[i]}'")
        if s.get("lexical_language") not in LEXICAL:
            errs.append(f"span {i}: lexical_language '{s.get('lexical_language')}' không thuộc {sorted(LEXICAL)}")
        if s.get("pronunciation_style") not in PRONUNCIATION:
            errs.append(f"span {i}: pronunciation_style '{s.get('pronunciation_style')}' không hợp lệ")
        if s.get("pronunciation_style") is not None and row.get("annotation_status") not in ("human_reviewed", "locked"):
            errs.append(f"span {i}: pronunciation_style chỉ gán khi câu đã được người nghe duyệt "
                        f"(annotation_status = {row.get('annotation_status')})")
        if s.get("entity_type") not in ENTITY:
            errs.append(f"span {i}: entity_type '{s.get('entity_type')}' không hợp lệ")
    return errs
