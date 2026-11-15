"""Cache hyps ASR theo (id, sha256 audio).

Một hyp chỉ được dùng lại cho một câu khi CẢ id lẫn hash audio khớp: id của VSS do converter đánh số theo thứ tự
(vss_0000123), nên chạy lại converter có thể đổi id; hash audio bảo đảm hyp thuộc đúng file âm thanh.

- Chỉ mục hash (`HashIndex`): TSV `audio_rel<TAB>sha256`, audio_rel = đường dẫn audio tương đối với gốc dataset
  (= đường dẫn trong repo HF). sha256 lấy từ metadata LFS của HF (`fill_from_hf`, không phải tải audio) hoặc tính
  từ file cục bộ. Nhờ đó tra cache không cần audio (tools/rebuild_vss_cs.sh chạy không GPU, không tải).
- File hyps (SCHEMA.md mục 3) có thêm `audio_sha256`, `audio_rel` (tools/asr_hyps.py ghi; bản cũ thì bổ sung bằng
  `python tools/hyp_cache.py backfill`).
- `HypLookup`: đọc nhiều file hyps (hoặc thư mục chứa chúng) cho mỗi ASR; tra theo (id, sha) khi có chỉ mục hash,
  theo id khi không có (chế độ cũ, dùng cho dữ liệu nhỏ / test).
"""
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def audio_rel(row: Dict, audio_root: Optional[str] = None) -> str:
    """Đường dẫn audio tương đối với gốc dataset (= đường dẫn trong repo HF)."""
    if row.get("audio_rel"):
        return row["audio_rel"]
    # os.path.abspath chỉ xử lý chuỗi (không gọi hệ thống file như Path.resolve): nhanh trên /mnt/d của WSL
    p = os.path.abspath(row.get("_audio_path") or row["audio"])
    if audio_root:
        root = os.path.abspath(audio_root)
        if p.startswith(root + os.sep):
            return Path(os.path.relpath(p, root)).as_posix()
    s = Path(p).as_posix()
    return s[s.index("audio/"):] if "audio/" in s else s


class HashIndex:
    def __init__(self, path: Optional[str]):
        self.path = Path(path) if path else None
        self.m: Dict[str, str] = {}
        if self.path and self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for r in csv.reader(f, delimiter="\t"):
                    if len(r) >= 2 and r[0] != "audio_rel":
                        self.m[r[0]] = r[1]
        self.dirty = False

    def get(self, rel: str) -> Optional[str]:
        return self.m.get(rel)

    def put(self, rel: str, sha: str):
        if self.m.get(rel) != sha:
            self.m[rel] = sha
            self.dirty = True

    def save(self):
        if not (self.path and self.dirty):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write("audio_rel\tsha256\n" + "".join(f"{k}\t{v}\n" for k, v in sorted(self.m.items())))
        tmp.replace(self.path)
        self.dirty = False

    def fill_from_hf(self, repo: str, rels: Iterable[str], batch: int = 500) -> Tuple[int, List[str]]:
        """Lấy sha256 (LFS) từ HF cho các rel chưa có trong chỉ mục. Trả về (số mục mới, các rel không có LFS)."""
        from huggingface_hub import HfApi
        need = sorted({r for r in rels if r not in self.m})
        api, n, no_lfs = HfApi(), 0, []
        for i in range(0, len(need), batch):
            part = need[i:i + batch]
            got = {f.path: f for f in api.get_paths_info(repo, part, repo_type="dataset")}
            for rel in part:
                f = got.get(rel)
                lfs = getattr(f, "lfs", None) if f is not None else None
                if lfs is not None and getattr(lfs, "sha256", None):
                    self.put(rel, lfs.sha256)
                    n += 1
                else:
                    no_lfs.append(rel)
        return n, no_lfs


def iter_hyp_files(path: str) -> List[Path]:
    p = Path(path)
    if p.is_dir():
        return sorted(x for x in p.glob("*.jsonl") if not x.name.endswith(".runs.jsonl"))
    return [p] if p.exists() else []


class HypTable:
    """Hyps của MỘT ASR (một file hoặc một thư mục file hyps)."""

    def __init__(self, path: str):
        self.by_key: Dict[Tuple[str, str], str] = {}
        self.by_id: Dict[str, str] = {}
        self.n_legacy = 0
        self.n_bad = 0
        for fp in iter_hyp_files(path):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        h = json.loads(line)
                    except json.JSONDecodeError:  # dòng bị cắt khi tiến trình bị dừng giữa lúc ghi
                        self.n_bad += 1
                        continue
                    hyp = h.get("hyp", "")
                    self.by_id.setdefault(h["id"], hyp)
                    if h.get("audio_sha256"):
                        self.by_key.setdefault((h["id"], h["audio_sha256"]), hyp)
                    else:
                        self.n_legacy += 1


class HypLookup:
    def __init__(self, paths: Sequence[str], index: Optional[HashIndex] = None, audio_root: Optional[str] = None):
        self.tables = [HypTable(p) for p in paths]
        self.index = index
        self.audio_root = audio_root

    def row_sha(self, row: Dict) -> Optional[str]:
        return self.index.get(audio_rel(row, self.audio_root)) if self.index else None

    def get(self, row: Dict) -> List[Optional[str]]:
        """Hyp của từng ASR cho câu này (None = chưa có trong cache)."""
        if self.index is None:
            return [t.by_id.get(row["id"]) for t in self.tables]
        sha = self.row_sha(row)
        if not sha:
            return [None] * len(self.tables)
        return [t.by_key.get((row["id"], sha)) for t in self.tables]
