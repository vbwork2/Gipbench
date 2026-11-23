#!/usr/bin/env python3
"""Mẫu dev ngẫu nhiên (gipformer_mech.splits.development_sample) và phép đo cơ chế của sweep blank penalty
(scripts/06_eval_blank_penalty.py). CPU, dữ liệu giả, không cần model hay audio.

  python tests/mech/test_blank_penalty.py      # thoát mã 1 nếu có ca hỏng
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from gipformer_mech.model_audit import checksum
from gipformer_mech.splits import development_sample

ROOT = Path(__file__).resolve().parents[2]
FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def sealed(directory, pool_size=20):
    rows = [{"utt_id": f"u{i:02d}", "audio_idx": i} for i in range(pool_size)]
    write(directory / "full_scored.jsonl", rows)
    write(directory / "bug_mining.jsonl", rows[:3])
    write(directory / "clean_control.jsonl", rows[3:6])
    (directory / "heldout_lock.json").write_text(json.dumps({
        "development_pool_sha256": checksum(directory / "full_scored.jsonl"),
        "development_sha256": {split: checksum(directory / f"{split}.jsonl") for split in ("bug_mining", "clean_control")},
    }), encoding="utf-8")


def test_development_sample():
    d = Path(tempfile.mkdtemp())
    sealed(d)
    sample = development_sample(d, 5, 42)
    ids = {row["utt_id"] for row in sample}
    check("sample: đúng cỡ, không trùng", len(sample) == 5 and len(ids) == 5)
    check("sample: loại bug_mining và clean_control", not ids & {f"u{i:02d}" for i in range(6)}, ids)
    check("sample: cùng seed -> cùng mẫu", [r["utt_id"] for r in development_sample(d, 5, 42)] == [r["utt_id"] for r in sample])
    check("sample: seed khác -> mẫu khác", {r["utt_id"] for r in development_sample(d, 5, 7)} != ids)
    check("sample: thiếu dòng -> ValueError", raises(lambda: development_sample(d, 15, 42)))
    with (d / "full_scored.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utt_id": "x", "audio_idx": 99}) + "\n")
    check("sample: pool khác bản niêm phong -> ValueError", raises(lambda: development_sample(d, 5, 42)))


def load_script():
    spec = importlib.util.spec_from_file_location("eval_blank_penalty", ROOT / "scripts/06_eval_blank_penalty.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stack(rank):
    return {"full": {"best_rank": rank, "top10": [{"token_id": 0}]}, "acoustic": {"best_rank": rank},
            "reference_full": {"best_rank": rank}}


def test_mechanism():
    tool = load_script()
    near = {"unknown": False, "stacks": [stack(40)] * 5 + [stack(3)]}
    far = {"unknown": False, "stacks": [stack(50)] * 6}
    d = Path(tempfile.mkdtemp())
    report = {"findings": [{"utt_id": "a", "errors": [
        {"word_index": 1, "category": "INCONCLUSIVE", "pieces": [near]},
        {"word_index": 4, "category": "TRUE_GAP", "pieces": [far]},
    ]}]}
    (d / "p2.json").write_text(json.dumps(report), encoding="utf-8")
    spans = tool.phase2_spans(d / "p2.json", {"a"})
    check("phase2_spans: phân loại lại từ rank, bỏ qua category cũ",
          spans == {("a", 1): "NEAR_MISS", ("a", 4): "OTHER"}, spans)
    check("phase2_spans: recording ngoài bug_mining -> ValueError",
          raises(lambda: tool.phase2_spans(d / "p2.json", {"b"})))
    m = tool.mechanism(spans, base={("a", 1), ("a", 4)}, candidate={("a", 4), ("a", 9)})
    check("mechanism: near miss sửa được 1/1", m["NEAR_MISS"] == {"wrong_at_zero": 1, "fixed": 1, "fix_rate": 1.0}, m)
    check("mechanism: span khác 0/1, lỗi mới không tính là sửa",
          m["OTHER"] == {"wrong_at_zero": 1, "fixed": 0, "fix_rate": 0.0}, m)
    m0 = tool.mechanism(spans, base={("a", 4)}, candidate=set())
    check("mechanism: span đúng sẵn ở penalty 0 không vào mẫu số",
          m0["NEAR_MISS"] == {"wrong_at_zero": 0, "fixed": 0, "fix_rate": None}, m0)
    scores = {"a": {"cs_word_errors": [{"ref_index": 2}]}, "b": {"cs_word_errors": []}}
    check("cs_error_positions", tool.cs_error_positions([{"utt_id": "a"}, {"utt_id": "b"}], scores) == {("a", 2)})


def main():
    test_development_sample()
    test_mechanism()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
