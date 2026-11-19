#!/usr/bin/env python3
"""Kiểm cấu hình / notebook Colab bằng file tĩnh, không cần GPU / mạng.

  python tests/dataset/test_colab_config.py
"""
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    example = yaml.safe_load((REPO / "configs/paths.example.yaml").read_text(encoding="utf-8"))
    colab = yaml.safe_load((REPO / "configs/paths.colab.yaml").read_text(encoding="utf-8"))
    check("paths.colab có đủ khoá như paths.example", set(example) <= set(colab), set(example) - set(colab))
    check("Colab dùng chung python3", all(colab[k] == "python3" for k in ("python_cpu", "python_ft", "python_asr")))
    check("giữ danh sách chặn rò rỉ mặc định", colab["leak_block_files"] == example["leak_block_files"])
    raw = (REPO / "scripts/colab/colab_ft.ipynb").read_text(encoding="utf-8")
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as e:
        check("notebook là JSON hợp lệ", False, e)
    else:
        check("notebook nbformat 4", nb.get("nbformat") == 4 and isinstance(nb.get("cells"), list))
        code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
        check("code cell chưa chạy, không output", bool(code) and all(
            c.get("outputs") == [] and c.get("execution_count", "thiếu") is None for c in code))
        check("mỗi code cell có markdown ngay trước", all(
            i > 0 and nb["cells"][i - 1].get("cell_type") == "markdown"
            for i, c in enumerate(nb.get("cells", [])) if c.get("cell_type") == "code"))
        for i, cell in enumerate(code):
            try:
                compile("".join(cell["source"]), f"colab_ft:cell_{i}", "exec")
            except (SyntaxError, KeyError, TypeError) as e:
                check(f"cú pháp code cell {i}", False, e)
        # kernel không thấy `pip install -e` cài sau khi khởi động: cell import gipbench phải tự thêm src vào sys.path
        imp = [c for c in code if "from gipbench" in "".join(c["source"])]
        check("cell import gipbench thêm src vào sys.path trước", bool(imp) and all(
            "sys.path.insert" in "".join(c["source"]).split("from gipbench")[0] for c in imp))
        check("không token hay đường dẫn máy local trong notebook",
              all(s not in raw for s in ("ghp_", "github_pat_", "/mnt/", "D:")))
    result = subprocess.run(["git", "check-ignore", "--no-index", "--verbose", "configs/paths.yaml"],
                            cwd=REPO, capture_output=True, text=True)
    check("gitignore bỏ qua configs/paths.yaml", result.returncode == 0 and
          ".gitignore:" in result.stdout and "configs/paths.yaml" in result.stdout, result.stdout or result.stderr)
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'ĐẠT tất cả'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
