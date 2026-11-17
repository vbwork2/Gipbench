#!/usr/bin/env python3
"""Hồ sơ chuẩn hoá có phiên bản (core.normprofiles): ví dụ cố định + bất biến.  python tests/core/test_normalization.py"""
import sys

from gipbench.core.normprofiles import PROFILES, canon_v1, get_profile
from gipbench.core.textnorm import basic_tokens

CASES = [  # (đầu vào, kết quả canon_v1)
    ("Hoà bình", "hòa bình"), ("thuỷ lợi", "thủy lợi"), ("khoẻ mạnh", "khỏe mạnh"),
    ("quỳ xuống", "quỳ xuống"),                      # qu là phụ âm đầu: giữ nguyên
    ("hoàn thành khoảng", "hoàn thành khoảng"),      # không phải vần mở: giữ nguyên
    ("kỹ thuật lý do", "kĩ thuật lí do"), ("my day", "my day"),   # y không dấu (từ Anh) giữ nguyên
    ("công nghệ A.I. mới", "công nghệ ai mới"), ("U.S.A", "usa"),
    ("gửi a.b@gmail.com", "gửi a chấm b a còng gmail chấm com"), ("vào shopee.vn", "vào shopee chấm vn"),
    ("I don’t know", "i dont know"), ("Colour favourite", "color favorite"),
    ("<en>Colour</en> đẹp", "<en>color</en> đẹp"),   # tag giữ nguyên
    ("tôi đi. Bạn về", "tôi đi. bạn về"),            # dấu chấm câu không bị coi là viết tắt
]


def main():
    fail = []
    for src, want in CASES:
        got = canon_v1(src)
        ok = got == want
        print(("ĐẠT  " if ok else "HỎNG ") + f"{src!r} -> {got!r}" + ("" if ok else f" (mong {want!r})"))
        fail += [] if ok else [src]
    for src, _ in CASES:  # idempotent
        if canon_v1(canon_v1(src)) != canon_v1(src):
            fail.append("idempotent:" + src)
            print("HỎNG idempotent", src)
    if get_profile("legacy_v0").pre("Hoà A.I.") != "Hoà A.I.":  # legacy_v0 = identity: giữ số cũ
        fail.append("legacy identity")
        print("HỎNG legacy_v0 phải là identity")
    if len({p.sha256() for p in PROFILES.values()}) != len(PROFILES):
        fail.append("sha")
        print("HỎNG sha256 các hồ sơ phải khác nhau")
    if PROFILES["gipbench_lenient_diagnostic_v1"].rankable:
        fail.append("rankable")
        print("HỎNG lenient phải rankable=False")
    for a, b in [("Hoà thuỷ kỹ A.I.", "hòa thủy kĩ AI"), ("email a@b.com", "email a a còng b chấm com")]:
        if basic_tokens(canon_v1(a)) != basic_tokens(canon_v1(b)):
            fail.append(a)
            print("HỎNG hai cách viết phải cho cùng token:", a, "|", b)
    print(f"\n{len(fail)} ca hỏng" if fail else "\nTất cả đạt")
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
