#!/usr/bin/env python3
"""Chọn sự kiện (fpt_event_replay.events_of) và đếm/kiểm dây chuyền (fpt_extra_report). CPU, dữ liệu giả.

  python tests/mech/test_extra_diagnosis.py      # thoát mã 1 nếu có ca hỏng
"""
from collections import Counter
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/diagnostics"))
BLANK = 0
FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/diagnostics/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def piece(rank, leader=BLANK, frame=5, target=7):
    """Rank ở tầng cuối; mọi stack giống nhau. leader = token đứng đầu top10 khi rank > 1."""
    top = target if rank == 1 else leader
    stack = {"full": {"best_rank": rank, "frame": frame,
                      "top10": [{"token_id": top, "piece": "<blk>" if top == BLANK else "x", "logprob": -0.1}]},
             "acoustic": {"best_rank": rank, "top10": []}, "reference_full": {"best_rank": rank}}
    return {"target_id": target, "unknown": False, "stacks": [stack] * 6}


def error(ranks, leaders=None, word="WORD", index=0):
    leaders = leaders or [BLANK] * len(ranks)
    return {"word_index": index, "reference_word": word,
            "pieces": [piece(r, l, frame=i) for i, (r, l) in enumerate(zip(ranks, leaders))]}


def test_events():
    replay = load("fpt_event_replay")
    findings = [{"utt_id": "a", "errors": [
        error([1, 3, 2], [BLANK, BLANK, 9]),  # near miss: piece 1 blank dẫn đầu, piece 2 bị piece khác dẫn đầu
        error([1, 40], index=4),               # không phải near miss (piece ngoài top-10)
    ]}]
    events = replay.events_of(findings, BLANK)
    check("chỉ lấy piece near miss có rank > 1 và blank dẫn đầu", [(e["word_index"], e["piece"]) for e in events] == [(0, 1)],
          events)
    check("sự kiện giữ frame và target", events[0]["frame"] == 1 and events[0]["target"] == 7)
    hit = {"utt_id": "a", "word": "W", "replicated": True}
    summary = replay.describe([hit, {**hit, "replicated": False}, {**hit, "utt_id": "b", "word": "V"}])
    check("describe đếm sự kiện, bản ghi và từ", summary["events"] == 3 and summary["replicated"] == 2
          and summary["recordings_with_replicated"] == 2 and summary["distinct_words_replicated"] == 2, summary)


def test_report():
    report = load("fpt_extra_report")
    near = {**error([1, 3, 4], [BLANK, BLANK, 9]), "category": "NEAR_MISS"}
    check("blank_led_pieces đếm piece rank > 1 có blank dẫn đầu", report.blank_led_pieces(near) == 1)
    check("blank_led_pieces = 0 ngoài near miss", report.blank_led_pieces({**near, "category": "INCONCLUSIVE"}) == 0)
    held, _ = report.cascade_test(Counter({("CHAIN", True): 6, ("LOCAL", True): 4, ("CHAIN", False): 3, ("START", False): 7}))
    check("dây chuyền: 60% ở nhóm đạt được, 30% ở nhóm còn lại -> giữ", held)
    held, _ = report.cascade_test(Counter({("CHAIN", True): 5, ("LOCAL", True): 5, ("CHAIN", False): 1}))
    check("dây chuyền: đúng 50% -> không giữ", not held)
    profile = report.profile(report.relabel([{"utt_id": "a", "errors": [near, error([1, 40, 2])]}]))
    check("profile: nhãn lại theo classify", profile["categories"] == Counter({"NEAR_MISS": 1, "INCONCLUSIVE": 1}),
          profile["categories"])
    check("profile: span INCONCLUSIVE piece đầu rank 1 rồi thiếu -> LOCAL", profile["cascade"] == Counter({("LOCAL", False): 1}),
          profile["cascade"])


def main():
    test_events()
    test_report()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
