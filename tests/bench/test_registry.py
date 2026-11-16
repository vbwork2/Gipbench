#!/usr/bin/env python3
"""Registry đánh giá (bench.registry): file trong repo hợp lệ; luật chặn hoạt động.  python tests/bench/test_registry.py"""
import copy
import sys

from gipbench.bench import registry as R

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    reg = R.load_datasets()
    check("datasets.yaml hợp lệ", R.validate_datasets(reg) == [], R.validate_datasets(reg))
    for n in sorted(p.stem for p in (R.REG_DIR / "benchmarks").glob("*.yaml")):
        e = R.validate_benchmark(R.load_benchmark(n), reg)
        check(f"benchmark {n} hợp lệ", e == [], e)
    g = R.load_benchmark("general_v1_draft")
    bad = copy.deepcopy(g)
    bad["allowed_inference_modes"].append("oracle_bias")
    check("G0 cấm oracle_bias", any("oracle_bias" in e for e in R.validate_benchmark(bad, reg)))
    bad = dict(g, evidence_level="locked", manifest_sha256=None)
    check("locked cần manifest_sha256", any("manifest_sha256" in e for e in R.validate_benchmark(bad, reg)))
    bad = dict(g, sources=[{"dataset": "khong_co"}])
    check("nguồn lạ bị bắt", any("khong_co" in e for e in R.validate_benchmark(bad, reg)))
    r2 = copy.deepcopy(reg)
    r2["datasets"]["vivos"]["license"] = "unknown"
    check("license unknown bị bắt", any("license" in e for e in R.validate_datasets(r2)))
    rows = [{"id": "b", "audio": "/x/b.wav", "text": "hai"}, {"id": "a", "audio": "C:/a.wav", "text": "một", "_line": 3}]
    h = R.manifest_sha256(rows)
    check("manifest_sha256 không phụ thuộc thứ tự / đường dẫn audio",
          h == R.manifest_sha256([dict(rows[1], audio="rel/a.wav"), dict(rows[0], audio="z")]))
    check("manifest_sha256 đổi khi text đổi", h != R.manifest_sha256([rows[0], dict(rows[1], text="ba")]))
    check("check_frozen bắt manifest đổi", R.check_frozen({"manifest_sha256": h}, [rows[0]]) is not None
          and R.check_frozen({"manifest_sha256": h}, rows) is None)
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
