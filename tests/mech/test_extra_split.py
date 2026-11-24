#!/usr/bin/env python3
"""Tách heldout mới từ pool FPT bổ sung (scripts/diagnostics/fpt_extra_split.py). CPU, dữ liệu giả.

  python tests/mech/test_extra_split.py      # thoát mã 1 nếu có ca hỏng
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def load_tool():
    path = ROOT / "scripts/diagnostics/fpt_extra_split.py"
    spec = importlib.util.spec_from_file_location("fpt_extra_split", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    return tool


def row(index, text, set_name="Set001", cs=False, errors=0):
    return {"utt_id": f"u{index:05d}", "audio_idx": index, "reference": text,
            "audio_file": f"FPTOpenSpeechData_{set_name}_V0.1_{index:06d}.mp3",
            "cs_tokens": ["X"] if cs else [], "score_counts": {"cs_word_errors": [{}] * errors}}


def synthetic():
    rows, index = [], 0
    for set_name in ("Set001", "Set002"):
        for cs in (False, True):
            for group in range(40):
                for _ in range(1 + group % 3):  # nhóm 1–3 dòng cùng văn bản
                    rows.append(row(index, f"{set_name} {'cs' if cs else 'vi'} cau {group}", set_name, cs, int(cs)))
                    index += 1
    rows.append(row(index, "trùng dev", "Set001", True, 2))
    rows.append(row(index + 1, "Set001 vi cau 0", "Set002"))  # nhóm nằm ở cả hai Set
    return rows


def test_split():
    tool = load_tool()
    rows = synthetic()
    dev = {tool.text_key({"reference": "trùng dev"}, "upper")}
    heldout, diagnostic, report = tool.split(rows, dev, "upper")
    key = lambda r: tool.text_key(r, "upper")
    check("mọi dòng nằm đúng một phía", len(heldout) + len(diagnostic) == len(rows)
          and not {r["utt_id"] for r in heldout} & {r["utt_id"] for r in diagnostic})
    check("không nhóm văn bản nào nằm ở cả hai phía", not {key(r) for r in heldout} & {key(r) for r in diagnostic})
    check("dòng trùng văn bản dev bị ép sang phía chẩn đoán",
          all(key(r) not in dev for r in heldout) and report["forced_development_text_rows"] == 1
          and report["forced_development_text_cs_spans"] == 2)
    check("giữ thứ tự dòng", [r["audio_idx"] for r in heldout] == sorted(r["audio_idx"] for r in heldout)
          and [r["audio_idx"] for r in diagnostic] == sorted(r["audio_idx"] for r in diagnostic))
    check("bốn tầng Set × có CS", sorted(report["strata"]) == ["Set001|cs", "Set001|no_cs", "Set002|cs", "Set002|no_cs"],
          sorted(report["strata"]))
    for name, stratum in report["strata"].items():
        check(f"{name}: ≥ 30% và lệch không quá một nhóm (3 dòng)",
              0.3 * stratum["rows"] <= stratum["heldout_rows"] < 0.3 * stratum["rows"] + 3, stratum)
    check("nhóm ở hai Set theo dòng có audio_idx thấp nhất",
          report["strata"]["Set001|no_cs"]["rows"] == sum(1 + g % 3 for g in range(40)) + 1)
    again = tool.split(rows, dev, "upper")
    check("cùng salt -> cùng kết quả", [r["utt_id"] for r in again[0]] == [r["utt_id"] for r in heldout])
    other = tool.split(rows, dev, "upper", salt="khác")
    check("salt khác -> heldout khác", [r["utt_id"] for r in other[0]] != [r["utt_id"] for r in heldout])
    try:
        tool.set_of({"audio_file": "x.mp3"})
        check("tên file không có Set -> ValueError", False)
    except ValueError:
        check("tên file không có Set -> ValueError", True)


def main():
    test_split()
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
