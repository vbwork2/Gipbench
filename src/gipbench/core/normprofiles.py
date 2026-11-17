"""Hồ sơ chuẩn hoá (evaluation profile) có phiên bản, dùng cho gipbench.bench.evaluate.

Mỗi hồ sơ = một hàm text -> text chạy TRƯỚC pipeline cũ của core.textnorm (NFC, lowercase, số -> chữ, bỏ dấu câu,
tách token theo khoảng trắng, bảng catalog). Hàm được áp ĐỐI XỨNG cho reference và hypothesis; tag <en>...</en> của
reference được giữ nguyên (chỉ phần text giữa các tag bị biến đổi), nên ranh giới span tiếng Anh không đổi.

Hồ sơ hiện có (đổi luật = thêm hồ sơ / phiên bản mới, KHÔNG sửa hồ sơ đã phát hành):
  legacy_v0                      = đúng hành vi của gipbench.bench.score (không biến đổi gì thêm). Dùng để chấm lại
                                   v0-draft cho khớp số cũ.
  gipbench_strict_v1             = legacy_v0 + chuẩn chính tả không đổi cách đọc (xem canon_v1). Catalog: chỉ cột variants.
  gipbench_lenient_diagnostic_v1 = gipbench_strict_v1 + cột `phonetic` của catalog (phiên âm Việt được coi là đúng).
                                   CHỈ để chẩn đoán "nghe đúng, viết sai"; không dùng xếp hạng / chọn model.

Không có hồ sơ "reference_compatible" chung: phần lớn bộ dữ liệu công khai không công bố scorer chính thức. Registry
(configs/bench/datasets.yaml, trường official_scorer) ghi rõ từng bộ; khi chưa có thì ghi "không có", không tự chế.
"""
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict

_TAG = re.compile(r"(</?en(?:\s[^>]*)?>)")

# ---------------------------------------------------------------------------- canon_v1
# 1) Vị trí dấu thanh trong vần mở oa / oe / uy: hoà = hòa, thuỷ = thủy (chuẩn về kiểu cũ: dấu trên nguyên âm đầu).
_TONES = {"a": "àáảãạ", "e": "èéẻẽẹ", "y": "ỳýỷỹỵ"}
_BASE = {"o": "òóỏõọ", "u": "ùúủũụ"}
_OLD_STYLE: Dict[str, str] = {}
for _first, _second in (("o", "a"), ("o", "e"), ("u", "y")):
    for _k in range(5):
        _OLD_STYLE[_first + _TONES[_second][_k]] = _BASE[_first][_k] + _second
# chỉ ở cuối âm tiết (vần mở): "hoàn", "khoảng" không đổi. "qu" là phụ âm đầu nên "quý" giữ nguyên.
_OPEN = re.compile(r"(?<![\w])(\w*?)(" + "|".join(_OLD_STYLE) + r")(?!\w)")
# 2) i / y CÓ DẤU THANH sau phụ âm đầu, âm tiết một nguyên âm: kỹ = kĩ, lý = lí (chuẩn về i). "y" không dấu giữ nguyên
#    để không đụng từ tiếng Anh (my, by).
_Y_TO_I = str.maketrans("ỳýỷỹỵ", "ìíỉĩị")
_CONS_Y = re.compile(r"(?<![\w])([bcdghklmnpqrstvxđ]h?)([ỳýỷỹỵ])(?!\w)")
# 3) chữ viết tắt có dấu chấm: A.I. / a.i -> ai, U.S.A -> usa (làm trước khi dấu chấm bị bỏ thành khoảng trắng)
_DOTTED = re.compile(r"(?<![\w.])((?:[^\W\d_]\.)+[^\W\d_])\.?(?!\w)")
# 4) email / tên miền: "@" đọc "a còng", "." trong tên miền đọc "chấm" (quy ước đọc tiếng Việt)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|vn|net|org|edu|gov|io|ai|info)\b", re.I)
# 5) dấu nháy: bỏ hẳn (don't = dont); đồng nhất ’ ‘ ʼ ` với '
_QUOTES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "`": "'"})
# 6) chính tả Anh-Anh -> Anh-Mỹ: bảng cố định soạn tay từ quy tắc chính tả (KHÔNG lấy từ lỗi model trên test)
EN_SPELLING_V1 = {
    "colour": "color", "colours": "colors", "favourite": "favorite", "favourites": "favorites", "centre": "center",
    "centres": "centers", "organise": "organize", "organised": "organized", "organisation": "organization",
    "optimise": "optimize", "optimised": "optimized", "optimisation": "optimization", "analyse": "analyze",
    "analysed": "analyzed", "licence": "license", "programme": "program", "programmes": "programs",
    "behaviour": "behavior", "travelling": "traveling", "cancelled": "canceled", "catalogue": "catalog",
    "defence": "defense", "realise": "realize", "recognise": "recognize", "minimise": "minimize",
    "maximise": "maximize", "customise": "customize", "customised": "customized", "labour": "labor",
}
_EN_SPELL = re.compile(r"(?<![\w])(" + "|".join(sorted(EN_SPELLING_V1, key=len, reverse=True)) + r")(?!\w)")


def _canon_segment(s: str) -> str:
    s = unicodedata.normalize("NFC", s).translate(_QUOTES)
    s = _EMAIL.sub(lambda m: m.group(0).replace("@", " a còng ").replace(".", " chấm "), s)
    s = _DOMAIN.sub(lambda m: m.group(0).replace(".", " chấm "), s)
    s = _DOTTED.sub(lambda m: m.group(1).replace(".", ""), s)
    s = s.lower().replace("'", "")
    s = _OPEN.sub(lambda m: m.group(0) if m.group(1).endswith("q") else m.group(1) + _OLD_STYLE[m.group(2)], s)
    s = _CONS_Y.sub(lambda m: m.group(1) + m.group(2).translate(_Y_TO_I), s)
    return _EN_SPELL.sub(lambda m: EN_SPELLING_V1[m.group(1)], s)


def canon_v1(text: str) -> str:
    """Chuẩn chính tả v1, giữ nguyên tag <en>. Ví dụ: 'Hoà' -> 'hòa', 'kỹ' -> 'kĩ', 'A.I.' -> 'ai',
    'a@b.com' -> 'a a còng b chấm com', "don't" -> 'dont', 'colour' -> 'color'."""
    parts = _TAG.split(text or "")
    return "".join(p if _TAG.fullmatch(p) else _canon_segment(p) for p in parts)


def identity(text: str) -> str:
    return text or ""


@dataclass(frozen=True)
class Profile:
    name: str
    version: str
    pre: Callable[[str], str]
    lenient: bool    # True: dùng cột phonetic của catalog (mode "lenient" của core.textnorm)
    rankable: bool   # False: chỉ chẩn đoán, không được dùng xếp hạng / chọn model
    description: str

    @property
    def catalog_mode(self) -> str:
        return "lenient" if self.lenient else "strict"

    def spec(self) -> Dict:
        d = {"name": self.name, "version": self.version, "lenient": self.lenient, "rankable": self.rankable,
             "pre": self.pre.__name__}
        if self.pre is canon_v1:
            d["en_spelling_table"] = EN_SPELLING_V1
        return d

    def sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.spec(), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


PROFILES = {
    "legacy_v0": Profile("legacy_v0", "0", identity, False, True,
                         "Hành vi của gipbench.bench.score (strict) trước 16/11/2026."),
    "gipbench_strict_v1": Profile("gipbench_strict_v1", "1", canon_v1, False, True,
                                  "legacy_v0 + canon_v1 (dấu thanh oa/oe/uy, i/y, viết tắt có chấm, email/tên miền, "
                                  "dấu nháy, chính tả Anh-Anh/Anh-Mỹ)."),
    "gipbench_lenient_diagnostic_v1": Profile("gipbench_lenient_diagnostic_v1", "1", canon_v1, True, False,
                                              "gipbench_strict_v1 + phiên âm Việt trong cột phonetic của catalog. "
                                              "Chỉ chẩn đoán."),
}


def get_profile(name: str) -> Profile:
    if name not in PROFILES:
        raise KeyError(f"không có hồ sơ chuẩn hoá '{name}'; có: {', '.join(PROFILES)}")
    return PROFILES[name]
