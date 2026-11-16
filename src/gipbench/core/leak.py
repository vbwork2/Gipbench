"""Chặn rò rỉ bench -> train dùng chung cho converter và công cụ train (Spec mục 5).

Một câu train bị loại nếu có: speaker hoặc source trùng bench, text (sau chuẩn hoá + catalog) trùng một câu bench,
thuật ngữ reserved_b3, là dữ liệu test_only (is_test_only), hoặc text trùng danh sách chặn (block_files: text đã
chuẩn hoá của các bộ CHỈ ĐƯỢC TEST, vd consensus; mặc định DEFAULT_BLOCK_FILES nếu file có trên đĩa).
Hash audio không kiểm ở đây (cần đọc audio): validate.py --train làm việc đó.
"""
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Set

from gipbench.core.textnorm import Catalog, basic_tokens, plain_text

from gipbench.core.paths import REPO as _ROOT
# text chuẩn hoá của các bộ chỉ được dùng để test (src/gipbench/dataset/ingest/consensus.py ghi ra)
DEFAULT_BLOCK_FILES = [_ROOT / "data" / "raw" / "consensus" / "leak_block.txt"]
TEST_ONLY_LICENSE = "test_only"
# source / dataset của bộ test_only: không bao giờ được có trong train / dev, kể cả khi dòng mất trường license
TEST_ONLY_SOURCES = {"consensus"}


def norm_text(text: str, cat: Catalog) -> str:
    return " ".join(cat.apply(basic_tokens(plain_text(text or ""))))


def is_test_only(row: Dict, extra_sources: Iterable[str] = ()) -> str:
    """Lý do dòng thuộc dữ liệu chỉ được test (chuỗi rỗng = không)."""
    if str(row.get("license") or "").strip().lower() == TEST_ONLY_LICENSE:
        return f"license {TEST_ONLY_LICENSE}"
    srcs = TEST_ONLY_SOURCES | set(extra_sources)
    if row.get("source") in srcs:
        return f"source {row.get('source')} (test_only)"
    if str(row.get("dataset") or "").split(":")[0] in srcs:
        return f"dataset {row.get('dataset')} (test_only)"
    return ""


def load_block_texts(paths: Optional[Sequence] = None, cat: Optional[Catalog] = None, quiet: bool = False) -> Set[str]:
    """Đọc danh sách chặn (1 câu / dòng, bỏ dòng '#'), chuẩn hoá lại theo catalog hiện tại.

    paths = None -> DEFAULT_BLOCK_FILES (file nào chưa có thì báo và bỏ qua); [] -> tập rỗng.
    """
    cat = cat or Catalog()
    default = paths is None
    out = set()
    for p in (DEFAULT_BLOCK_FILES if default else paths):
        p = Path(p)
        if not p.exists():
            if not quiet:
                print(f"  LƯU Ý: chưa có danh sách chặn {p}"
                      + (" (chưa chạy src/gipbench/dataset/ingest/consensus.py?)" if default else ""))
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                t = norm_text(line, cat)
                if t:
                    out.add(t)
    return out


def add_block_arg(ap):
    ap.add_argument("--block-text", nargs="*", default=None,
                    help="danh sách chặn: text đã chuẩn hoá của bộ chỉ được test (mặc định "
                         + " ".join(str(p.relative_to(_ROOT)) for p in DEFAULT_BLOCK_FILES) + "; truyền rỗng để tắt)")


class LeakGuard:
    def __init__(self, bench_rows: Iterable[Dict], cat: Catalog, splits: Optional[Sequence[str]] = None,
                 block_files: Optional[Sequence] = None):
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
        self.block = load_block_texts(block_files, cat)

    def reason(self, row: Dict, text: str) -> str:
        """Lý do loại (chuỗi rỗng = không rò rỉ)."""
        why = is_test_only(row)
        if why:
            return f"dữ liệu test_only ({why})"
        if row.get("speaker") in self.spk:
            return "speaker trùng bench"
        if row.get("source") in self.src:
            return "source trùng bench"
        toks = self.cat.apply(basic_tokens(plain_text(text or "")))
        if self.reserved and any(t in self.reserved for t in self.cat.find_terms(toks)):
            return "có thuật ngữ reserved_b3"
        s = " ".join(toks)
        if s in self.txt:
            return "text trùng câu bench"
        if s in self.block:
            return "text trùng danh sách chặn (bộ test_only)"
        return ""
