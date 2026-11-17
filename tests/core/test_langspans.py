#!/usr/bin/env python3
"""Trường tuỳ chọn language_spans (core.langspans) + hồi quy evaluate với run_meta có context_vocab = null.

  python tests/core/test_langspans.py
"""
import json
import sys
import tempfile
from pathlib import Path

from gipbench.core.langspans import check_language_spans

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    base = {"text_tagged": "bạn giúp mình <en>deploy</en> cái <en>backend</en> nhé", "annotation_status": "human_reviewed"}
    ok = dict(base, language_spans=[
        {"text": "deploy", "lexical_language": "en", "pronunciation_style": "vi_accented", "entity_type": None},
        {"text": "backend", "lexical_language": "en", "pronunciation_style": None, "entity_type": "technical_term"}])
    check("ví dụ hợp lệ", check_language_spans(ok) == [], check_language_spans(ok))
    check("không có trường -> hợp lệ (tương thích ngược)", check_language_spans(base) == [])
    bad = dict(ok, language_spans=ok["language_spans"][:1])
    check("thiếu một span -> lỗi", any("1 mục" in e for e in check_language_spans(bad)))
    bad = dict(ok, language_spans=[dict(ok["language_spans"][0], text="deploy it"), ok["language_spans"][1]])
    check("text lệch span <en> -> lỗi", any("khác span" in e for e in check_language_spans(bad)))
    bad = dict(ok, annotation_status="upstream_human")
    check("pronunciation_style khi chưa nghe duyệt -> lỗi", any("nghe" in e for e in check_language_spans(bad)))
    zalo = {"text_tagged": "nhắn <en>Zalo</en> nha", "annotation_status": "draft",
            "language_spans": [{"text": "Zalo", "lexical_language": "other", "pronunciation_style": None, "entity_type": "brand"}]}
    check("brand Latin không phải từ tiếng Anh (lexical other, entity brand)", check_language_spans(zalo) == [])
    bad = dict(zalo, language_spans=[dict(zalo["language_spans"][0], entity_type="company")])
    check("entity_type lạ -> lỗi", check_language_spans(bad) != [])
    # hồi quy: run_meta.json có "context_vocab": null không được làm evaluate hỏng
    from gipbench.bench import evaluate as E
    d = Path(tempfile.mkdtemp())
    man = d / "m.jsonl"
    man.write_text(json.dumps({"id": "u0", "dataset": "toy", "speaker": "s", "text_tagged": "xin chào"}) + "\n" +
                   json.dumps({"id": "u1", "dataset": "toy", "speaker": "t", "text_tagged": "tạm biệt"}) + "\n", encoding="utf-8")
    (d / "sys").mkdir()
    (d / "sys" / "hyps.jsonl").write_text('{"id": "u0", "hyp": "xin chào"}\n{"id": "u1", "hyp": "tạm biệt"}\n', encoding="utf-8")
    (d / "sys" / "run_meta.json").write_text(json.dumps({"inference_mode": "open_vocab", "context_vocab": None}), encoding="utf-8")
    s = E.main(["--manifest", str(man), "--hyps", str(d / "sys" / "hyps.jsonl"), "--iters", "20"])
    check("evaluate đọc run_meta có context_vocab null, lấy inference_mode từ run_meta",
          s["inference_mode"] == "open_vocab" and s["profiles"]["gipbench_strict_v1"]["overall"]["wer"] == 0)
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
