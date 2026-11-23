"""Re-apply the current Phase 2 categories to a stored bug-vs-gap report without the model."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.logit_lens import INTERPRETATION, classify, summarize_findings


def reclassify(report: dict, blank: int) -> dict:
    findings = [
        {**row, "errors": [
            {**error, "category": classify(error["pieces"]),
             "reference_prefix_category": classify(error["pieces"], "reference_full")}
            for error in row["errors"]
        ]}
        for row in report["findings"]
    ]
    return {**report, **summarize_findings(findings, blank), "interpretation": INTERPRETATION, "findings": findings}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/phase2_bug_vs_gap.json")
    parser.add_argument("--out", type=Path, help="Defaults to rewriting --report in place.")
    args = parser.parse_args()
    model_map = json.loads((ROOT / "configs/model_map.json").read_text(encoding="utf-8"))
    report = reclassify(json.loads(args.report.read_text(encoding="utf-8")), model_map["blank_id"])
    out = args.out or args.report
    out.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in (
        "error_span_count", "category_counts", "reference_prefix_category_counts",
        "near_miss_blank_led_count", "decision",
    )}
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
