"""Tiện ích file dùng chung cho mọi tầng: sha256 của file, JSONL thô, tổng giờ audio. Chỉ stdlib.

Khác `gipbench.core.manifest.read_jsonl` / `write_jsonl` (dành cho manifest bench: thêm `_line`, `_audio_path`, bỏ trường
`_`): các hàm ở đây đọc / ghi đúng như trong file, không thêm hay bỏ trường nào.
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

CHUNK = 1 << 20


def sha256_file(path, bufsize: int = CHUNK) -> str:
    """sha256 (hex) của nội dung file, đọc theo khối. Lỗi nếu không có file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bufsize), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file_or_none(path) -> Optional[str]:
    """Như `sha256_file`, nhưng trả None khi `path` rỗng / None hoặc file không tồn tại."""
    if not path or not Path(path).exists():
        return None
    return sha256_file(path)


def iter_jsonl(path) -> Iterator[Dict]:
    """Từng dòng JSON của file, bỏ dòng trống."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl_rows(path) -> List[Dict]:
    return list(iter_jsonl(path))


def write_jsonl_rows(path, rows: Iterable[Dict]) -> None:
    """Ghi mỗi dòng một JSON (ensure_ascii=False), tạo thư mục cha nếu chưa có. Không đổi trường nào."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def hours(rows: Iterable[Dict]) -> float:
    """Tổng `duration` (giây) của các dòng, đổi ra giờ."""
    return sum(float(r["duration"]) for r in rows) / 3600
