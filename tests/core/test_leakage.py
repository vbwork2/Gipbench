#!/usr/bin/env python3
"""Kiểm nhiễm bẩn test -> train / hotwords (core.decontam).  python tests/core/test_leakage.py"""
import sys
import tempfile
from pathlib import Path

from gipbench.core.decontam import audit, dup_key, find_terms_in_texts, near_duplicates, test_terms_from_rows

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    d = Path(tempfile.mkdtemp())
    (d / "a.wav").write_bytes(b"same-bytes")
    (d / "b.wav").write_bytes(b"same-bytes")
    (d / "c.wav").write_bytes(b"other")
    test = [
        {"id": "t1", "speaker": "spkA", "source": "vidA", "text_tagged": "hôm nay tôi <en>deploy</en> cái <en>backend</en>",
         "_audio_path": str(d / "a.wav")},
        {"id": "t2", "speaker": "spkB", "source": "vidB", "text_tagged": "Bệnh nhân cần xét nghiệm máu định kỳ mỗi tháng một lần",
         "_audio_path": str(d / "c.wav")},
        {"id": "t3", "speaker": "spkC", "source": "vidC", "text_tagged": "xin chào", "_audio_path": str(d / "c.wav")},
    ]
    train = [
        {"id": "x1", "speaker": "spkA", "source": "vidZ", "text": "khác hẳn", "_audio_path": str(d / "b.wav")},
        {"id": "x2", "speaker": "spkQ", "source": "vidB", "text": "bệnh nhân cần xét nghiệm máu định kì mỗi tháng 1 lần nhé"},
        {"id": "x3", "speaker": "spkR", "source": "vidR", "text": "XIN CHÀO!"},
        {"id": "x4", "speaker": "spkS", "source": "vidS", "text": "mình sẽ deploy lên server"},
    ]
    rep = audit(test, train, reserved_terms=["deploy", "kubernetes"], runtime_vocab=["BACKEND", "zalo"],
                test_terms=test_terms_from_rows(test))
    kinds = {f["kind"]: f for f in rep["findings"]}
    check("trùng speaker", "speaker_overlap" in kinds)
    check("trùng source", "source_overlap" in kinds)
    check("trùng sha256 audio (khác tên file)", "audio_sha256_overlap" in kinds)
    check("trùng text sau chuẩn hoá (hoa/thường, dấu câu)", kinds.get("text_exact_overlap", {}).get("count") == 1,
          kinds.get("text_exact_overlap"))
    check("thuật ngữ FT-unseen có trong train",
          [e["term"] for e in kinds.get("reserved_term_in_train", {}).get("examples", [])] == ["deploy"])
    check("hotwords chứa thuật ngữ lấy từ test", kinds.get("test_term_in_runtime_vocab", {}).get("examples") == ["backend"])
    check("không có id trùng", "id_overlap" not in kinds)
    check("đếm critical >= 6", rep["critical"] >= 6, rep["critical"])
    nd = near_duplicates(test, train, threshold=0.6)
    check("gần trùng (kỳ/kì, số 1, thêm 'nhé')", [x["test_id"] for x in nd] == ["t2"], nd)
    check("câu ngắn không xét gần trùng", all(x["test_id"] != "t3" for x in nd))
    check("dup_key chuẩn chính tả + số", dup_key("Định kỳ 1 lần!") == dup_key("định kì một lần"))
    check("find_terms khớp cụm nhiều từ",
          find_terms_in_texts(["pull request"], ["tạo Pull Request mới", "pull nhé"]) == {"pull request": 1})
    clean = audit([test[2]], [{"id": "z", "speaker": "q", "source": "q", "text": "tạm biệt"}], check_audio_sha=False)
    check("tập sạch -> 0 phát hiện", clean["findings"] == [], clean)
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
