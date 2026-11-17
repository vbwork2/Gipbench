#!/usr/bin/env python3
"""Kiểm ranh giới giữa các pipeline (docs/ARCHITECTURE.md): thoát mã 1 nếu vi phạm.

  python tests/test_boundaries.py

Luật (theo thư mục con đầu tiên của src/gipbench/):
  core    -> chỉ stdlib / thư viện ngoài, không import gipbench.<khác>
  infer   -> core
  dataset -> core, infer
  model   -> core, infer
  bench   -> core, infer
Thêm: không file nào trong src/ dùng sys.path.insert; không gọi `python -m gipbench.<pipeline bị cấm>` qua subprocess.
Kiểm bằng AST (cả import nằm trong hàm) và chuỗi "gipbench.<x>" trong code.
"""
import ast
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "gipbench"
ALLOWED = {
    "core": {"core"},
    "infer": {"core", "infer"},
    "dataset": {"core", "infer", "dataset"},
    "model": {"core", "infer", "model"},
    "bench": {"core", "infer", "bench"},
}
MOD_STR = re.compile(r"""["']gipbench\.(\w+)""")


def layer(path):
    rel = path.relative_to(SRC).parts
    return rel[0] if len(rel) > 1 else None


def main():
    errors = []
    n = 0
    for f in sorted(SRC.rglob("*.py")):
        me = layer(f)
        if me is None:
            continue
        n += 1
        src = f.read_text(encoding="utf-8")
        rel = f.relative_to(SRC.parents[1]).as_posix()
        if me not in ALLOWED:
            errors.append(f"{rel}: thư mục '{me}' không thuộc tầng nào ({', '.join(ALLOWED)})")
            continue
        for node in ast.walk(ast.parse(src)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                mods = [node.module]
            elif isinstance(node, ast.ImportFrom) and node.level > 0:
                errors.append(f"{rel}:{node.lineno}: import tương đối, dùng gipbench.<...>")
            for m in mods:
                parts = m.split(".")
                if parts[0] == "gipbench" and len(parts) > 1 and parts[1] not in ALLOWED[me]:
                    errors.append(f"{rel}:{node.lineno}: tầng {me} không được import {m}")
                if parts[0] in ("lib", "converters", "ft", "tools", "score", "eda", "decode", "leaderboard", "validate"):
                    errors.append(f"{rel}:{node.lineno}: import module cũ trước refactor: {m}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "insert"
                    and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "path"):
                errors.append(f"{rel}:{node.lineno}: sys.path.insert (dùng package gipbench)")
        for i, line in enumerate(src.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for m in MOD_STR.finditer(line):
                if m.group(1) in ALLOWED and m.group(1) not in ALLOWED[me]:
                    errors.append(f"{rel}:{i}: tầng {me} gọi module gipbench.{m.group(1)} (chuỗi trong code)")
    for e in errors:
        print("LỖI", e)
    print(f"{n} file, {len(errors)} vi phạm ranh giới")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
