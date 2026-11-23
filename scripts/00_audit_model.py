"""Generate the Phase 0 model map and checkpoint audit."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gipformer_mech.model_audit import inspect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "base_model")
    args = parser.parse_args()

    model_map, report = inspect(args.model_dir)
    for relative, data in (("configs/model_map.json", model_map), ("reports/phase0_audit.json", report)):
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {relative}")


if __name__ == "__main__":
    main()
