#!/usr/bin/env python3
"""Phân loại lỗi Phase 2 (gipformer_mech.logit_lens.classify / summarize_findings) và công cụ phân loại lại.
CPU, dữ liệu giả, không cần model.

  python tests/mech/test_logit_lens_classify.py      # thoát mã 1 nếu có ca hỏng
"""
import importlib.util
import sys
from pathlib import Path

from gipformer_mech.logit_lens import blank_led_near_miss, classify, summarize_findings

ROOT = Path(__file__).resolve().parents[2]
BLANK = 0
TARGET = 7
OTHER = 9
FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def piece(full, acoustic=None, reference=None, leader=None, unknown=False):
    """full/acoustic/reference: rank của piece ở 6 stack (stack 0 … stack 5 = cuối)."""
    acoustic = acoustic or full
    reference = reference or full
    stacks = []
    for index, (f, a, r) in enumerate(zip(full, acoustic, reference, strict=True)):
        top = leader if (leader is not None and index == len(full) - 1 and f > 1) else (TARGET if f == 1 else OTHER)
        stacks.append({
            "stack": index,
            "full": {"best_rank": f, "top10": [{"token_id": top, "piece": "x", "logprob": -0.1}]},
            "acoustic": {"best_rank": a, "top10": []},
            "reference_full": {"best_rank": r},
        })
    return {"target_id": TARGET, "unknown": unknown, "stacks": stacks}


FAR = [50] * 6


def span(*pieces, category=None, reference_category="INCONCLUSIVE"):
    pieces = list(pieces)
    return {"category": category or classify(pieces), "reference_prefix_category": reference_category, "pieces": pieces}


def test_classify():
    near = [40, 30, 20, 15, 8, 2]
    check("NEAR_MISS: mọi piece trong top-10 stack cuối nhưng chưa rank 1",
          classify([piece(near), piece([30, 20, 9, 5, 3, 1])]) == "NEAR_MISS")
    check("NEAR_MISS: rank 10 ở stack cuối vẫn tính", classify([piece([50, 40, 30, 20, 12, 10])]) == "NEAR_MISS")
    check("rank 11 ở stack cuối không phải NEAR_MISS",
          classify([piece([50, 40, 30, 20, 12, 11])]) != "NEAR_MISS")
    check("NEAR_MISS: mọi piece rank 1 ở stack cuối", classify([piece([9, 5, 3, 2, 1, 1])]) == "NEAR_MISS")
    check("một piece ngoài top-10 ở cuối, một piece trong: vẫn INCONCLUSIVE",
          classify([piece(near), piece([60, 50, 40, 30, 20, 15])]) == "INCONCLUSIVE")
    check("tokenizer unknown đi trước NEAR_MISS -> TRUE_GAP",
          classify([piece(near, unknown=True)]) == "TRUE_GAP")
    check("ENCODER_BUG không đổi: stack giữa top-10, stack cuối rớt",
          classify([piece([40, 5, 30, 20, 15, 12])]) == "ENCODER_BUG")
    check("PREDICTOR_OVERRIDE_BUG không đổi: acoustic cuối top-10, full cuối rớt",
          classify([piece([40, 30, 20, 15, 12, 11], acoustic=[40, 30, 20, 15, 12, 4])]) == "PREDICTOR_OVERRIDE_BUG")
    check("TRUE_GAP không đổi: mọi stack và chế độ ngoài top-10", classify([piece(FAR)]) == "TRUE_GAP")
    p = piece(near, reference=[50, 40, 30, 20, 15, 14])
    check("context reference_full dùng rank của chính nó",
          classify([p]) == "NEAR_MISS" and classify([p], "reference_full") != "NEAR_MISS")


def test_blank_led():
    near = [40, 30, 20, 15, 8, 2]
    check("blank dẫn ở stack cuối -> blank_led",
          blank_led_near_miss(span(piece(near, leader=BLANK)), BLANK))
    check("token khác dẫn -> không blank_led",
          not blank_led_near_miss(span(piece(near, leader=OTHER)), BLANK))
    check("piece rank 1 không tính blank_led",
          not blank_led_near_miss(span(piece([9, 5, 3, 2, 1, 1], leader=BLANK)), BLANK))
    check("chỉ tính span NEAR_MISS",
          not blank_led_near_miss(span(piece(near, leader=BLANK), category="INCONCLUSIVE"), BLANK))


def test_summary():
    near = span(piece([40, 30, 20, 15, 8, 2], leader=BLANK))
    gap = span(piece(FAR))
    rows = [{"utt_id": "a", "errors": [near, near, near]}, {"utt_id": "b", "errors": [gap] * 7}]
    s = summarize_findings(rows, BLANK)
    check("summary: đếm category", s["category_counts"] == {"NEAR_MISS": 3, "TRUE_GAP": 7}, s["category_counts"])
    check("summary: tỉ lệ", abs(s["category_share"]["NEAR_MISS"] - 0.3) < 1e-12, s["category_share"])
    check("summary: NEAR_MISS 30% -> prioritize_score_calibration", s["decision"]["prioritize_score_calibration"])
    check("summary: các quyết định cũ giữ nguyên tên khoá",
          {"prioritize_hotwords_and_joiner", "allow_encoder_sweep", "prepare_finetune"} <= s["decision"].keys())
    check("summary: đếm NEAR_MISS do blank dẫn", s["near_miss_blank_led_count"] == 3, s["near_miss_blank_led_count"])
    low = summarize_findings([{"utt_id": "a", "errors": [near] + [gap] * 9}], BLANK)
    check("summary: NEAR_MISS 10% -> không ưu tiên", not low["decision"]["prioritize_score_calibration"])
    try:
        summarize_findings([{"utt_id": "a", "errors": []}], BLANK)
        check("summary: không có span -> ValueError", False, "không lỗi")
    except ValueError:
        check("summary: không có span -> ValueError", True)


def test_reclassify_tool():
    path = ROOT / "scripts/diagnostics/fpt_phase2_reclassify.py"
    spec = importlib.util.spec_from_file_location("fpt_phase2_reclassify", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    stale = span(piece([40, 30, 20, 15, 8, 2]), category="INCONCLUSIVE")
    old = {"error_span_count": 1, "category_counts": {"INCONCLUSIVE": 1}, "predictor_context": "greedy",
           "findings": [{"utt_id": "a", "errors": [{**stale, "word_index": 3}]}]}
    new = tool.reclassify(old, BLANK)
    error = new["findings"][0]["errors"][0]
    check("reclassify: INCONCLUSIVE cũ -> NEAR_MISS", error["category"] == "NEAR_MISS", error["category"])
    check("reclassify: cả reference_prefix_category được tính lại", error["reference_prefix_category"] == "NEAR_MISS")
    check("reclassify: giữ trường khác", error["word_index"] == 3 and new["predictor_context"] == "greedy")
    check("reclassify: summary cập nhật", new["category_counts"] == {"NEAR_MISS": 1}, new["category_counts"])
    check("reclassify: không sửa input", old["findings"][0]["errors"][0]["category"] == "INCONCLUSIVE")


def main():
    test_classify()
    test_blank_led()
    test_summary()
    test_reclassify_tool()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
