"""Đường dẫn gốc của repo và các thư mục dữ liệu / thí nghiệm.

REPO: gốc repo (chứa configs/, data/, results/). Thay cho `Path(__file__).parents[1]` của các script cũ.
Các thư mục khác đọc từ biến môi trường, rồi configs/paths.yaml (bản local, gitignore), rồi giá trị mặc định = vị trí
dùng trước refactor. Đổi máy hoặc đổi ổ đĩa: sửa configs/paths.yaml, không sửa code.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

_DEFAULTS = {
    "data_root": "data",
    "results_root": "results",
    "exp_root": "/mnt/c/Users/goose/gipbench_exp",   # checkpoint / artifact (D: thiếu chỗ, 16/11/2026)
    "icefall": "~/icefall",
    "python_cpu": "~/.venvs/gipbench/bin/python",
    "python_ft": "~/.venvs/gipbench-ft/bin/python",
    "python_asr": "~/.venvs/gipbench-asr/bin/python",
    "leak_block_files": ["data/raw/consensus/leak_block.txt"],
}
_ENV = {"data_root": "GIPBENCH_DATA", "results_root": "GIPBENCH_RESULTS", "exp_root": "GIPBENCH_EXP",
        "icefall": "ICEFALL", "python_cpu": "GIPBENCH_PY", "python_ft": "GIPBENCH_FT_PY", "python_asr": "GIPBENCH_ASR_PY"}
_cache = None


def _load():
    global _cache
    if _cache is None:
        cfg = dict(_DEFAULTS)
        f = REPO / "configs" / "paths.yaml"
        if f.exists():
            import yaml
            cfg.update(yaml.safe_load(f.read_text(encoding="utf-8")) or {})
        for k, env in _ENV.items():
            if os.environ.get(env):
                cfg[k] = os.environ[env]
        _cache = cfg
    return _cache


def get(key):
    """Giá trị thô (chuỗi hoặc danh sách) của một khoá."""
    return _load()[key]


def _abs(v):
    p = Path(os.path.expanduser(str(v)))
    return p if p.is_absolute() else REPO / p


def path(key):
    """Đường dẫn tuyệt đối: ~ được mở rộng, đường dẫn tương đối tính từ REPO."""
    return _abs(_load()[key])


def paths(key):
    return [_abs(v) for v in _load()[key]]
