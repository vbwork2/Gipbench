#!/usr/bin/env python3
"""So hai thư mục output của tests/golden/capture.py (vd trước / sau refactor). Thoát mã 1 nếu có file dữ liệu khác nhau.

  python tests/golden/compare.py /tmp/golden_old /tmp/golden_new [--show 20]

Chuẩn hoá trước khi so (chỉ những thứ sau, không gì khác):
  - đường dẫn tuyệt đối của chính thư mục output -> "<OUT>" (hai lần chạy ghi ra hai thư mục khác nhau)
  - file .html: bỏ dòng chứa thời điểm tạo (VOLATILE)
  - JSON: so sau khi sort_keys; bỏ các khoá trong VOLATILE_KEYS
File .log (stdout / stderr của lệnh) chỉ so mã thoát (dòng "rc=") và báo khác biệt stdout ở dạng CẢNH BÁO: thông điệp có
thể chứa tên script / module (đổi theo layout).
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path

VOLATILE_KEYS = {"created", "generated", "generated_at", "elapsed_s"}
VOLATILE = re.compile(r"(Tạo lúc|generated|Sinh lúc|\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})")


def roots(d):
    """Thư mục output gốc lúc chụp (ROOT.txt) và vị trí hiện tại: output có thể đã được copy đi chỗ khác."""
    r = [str(d)]
    if (d / "ROOT.txt").exists():
        r.insert(0, (d / "ROOT.txt").read_text(encoding="utf-8").strip())
    return r


def norm(path, root):
    t = path.read_text(encoding="utf-8", errors="replace")
    for r in roots(root):
        t = t.replace(r, "<OUT>")
    if path.suffix == ".json":
        try:
            obj = json.loads(t)
        except json.JSONDecodeError:
            return t

        def strip(o):
            if isinstance(o, dict):
                return {k: strip(v) for k, v in o.items() if k not in VOLATILE_KEYS}
            if isinstance(o, list):
                return [strip(v) for v in o]
            return o
        return json.dumps(strip(obj), ensure_ascii=False, sort_keys=True, indent=1)
    if path.suffix == ".html":
        return "\n".join(l for l in t.splitlines() if not VOLATILE.search(l))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--show", type=int, default=20, help="số dòng diff in cho mỗi file khác")
    args = ap.parse_args()
    A, B = Path(args.a).resolve(), Path(args.b).resolve()
    fa = {p.relative_to(A) for p in A.rglob("*") if p.is_file() and p.name != "ROOT.txt"}
    fb = {p.relative_to(B) for p in B.rglob("*") if p.is_file() and p.name != "ROOT.txt"}
    bad, warn = [], []
    for r in sorted(fa - fb):
        (warn if r.suffix == ".log" else bad).append(f"chỉ có ở A: {r}")
    for r in sorted(fb - fa):
        (warn if r.suffix == ".log" else bad).append(f"chỉ có ở B: {r}")
    same = 0
    for r in sorted(fa & fb):
        ta, tb = norm(A / r, A), norm(B / r, B)
        if r.suffix == ".log":
            rca, rcb = ta.splitlines()[0], tb.splitlines()[0]
            if rca != rcb:
                bad.append(f"mã thoát khác: {r}: {rca} / {rcb}")
            elif ta != tb:
                warn.append(f"log khác (chỉ cảnh báo): {r}")
            else:
                same += 1
            continue
        if ta == tb:
            same += 1
            continue
        d = list(difflib.unified_diff(ta.splitlines(), tb.splitlines(), str(r), str(r), lineterm="", n=0))
        bad.append(f"KHÁC: {r}\n    " + "\n    ".join(d[2:2 + args.show]))
    for w in warn:
        print("CẢNH BÁO", w)
    for b in bad:
        print("LỖI", b)
    print(f"\n{same} file giống, {len(warn)} cảnh báo, {len(bad)} lỗi")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
