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
    from gipbench.dataset.release.build_general import manifest_sha256 as bg_sha
    check("build_general và registry cùng định nghĩa manifest_sha256", bg_sha(rows) == h)
    # IP-07: bench require_audio_sha không khoá được khi còn câu thiếu audio_sha256 (E0 v1.0 khoá bằng metadata)
    ra = {"require_audio_sha": True, "evidence_level": "locked"}
    check("require_audio_sha + locked: thiếu audio_sha256 -> lỗi",
          "thiếu audio_sha256" in (R.check_frozen(ra, [dict(rows[0], audio_sha256="a" * 64), rows[1]]) or ""))
    check("require_audio_sha + locked: đủ audio_sha256 -> không lỗi",
          R.check_frozen(ra, [dict(r, audio_sha256="a" * 64) for r in rows]) is None)
    check("require_audio_sha + draft: thiếu sha vẫn chấm được (chưa khoá)",
          R.check_frozen({"require_audio_sha": True, "evidence_level": "draft"}, rows) is None)
    bad = dict(R.load_benchmark("external_v1_1"), boot_units=None)
    check("boot_unit per_dataset cần boot_units", any("boot_units" in e for e in R.validate_benchmark(bad, reg)))
    from gipbench.bench.evaluate import resolve_boot_unit, unit_of
    bu = resolve_boot_unit(R.load_benchmark("external_v1_1"))
    check("per_dataset: FLEURS theo sentence_id, bộ khác theo speaker",
          unit_of({"id": "a", "dataset": "fleurs_vi", "sentence_id": "7", "speaker": "a"}, {}, bu) == "7"
          and unit_of({"id": "b", "dataset": "lsvsc", "speaker": "s1"}, {}, bu) == "s1", bu)
    for n in ("general_v1", "code_switch_v1", "cs_ggroup_v1"):
        b = R.load_benchmark(n)
        check(f"{n} đã khoá (locked + manifest_sha256)",
              b["evidence_level"] == "locked" and len(b.get("manifest_sha256") or "") == 64)
    # evaluate: catalog khác catalog_sha256 của bản khoá -> thoát mã 2
    import json
    import tempfile
    from pathlib import Path

    import yaml

    from gipbench.bench import evaluate as E
    d = Path(tempfile.mkdtemp())
    man = d / "m.jsonl"
    mrows = [{"id": "u1", "dataset": "vivos", "text_tagged": "xin chào <en>ok</en>", "speaker": "s1", "duration": 1.0}]
    man.write_text(json.dumps(mrows[0], ensure_ascii=False) + "\n", encoding="utf-8")
    (d / "sys").mkdir()
    (d / "sys" / "hyps.jsonl").write_text(json.dumps({"id": "u1", "hyp": "xin chào ok"}) + "\n", encoding="utf-8")
    cat = d / "cat.tsv"
    cat.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nok\t\t\t\t0\n", encoding="utf-8")
    bm = dict(R.load_benchmark("code_switch_v1"), manifest=str(man), catalog=str(cat),
              manifest_sha256=R.manifest_sha256(mrows), sources=[{"dataset": "vivos", "split": "test"}],
              catalog_sha256="0" * 64)
    for k in ("_path", "_name"):
        bm.pop(k, None)
    by = d / "bm.yaml"

    def ev():
        by.write_text(yaml.safe_dump(bm, allow_unicode=True), encoding="utf-8")
        try:
            E.main(["--benchmark", str(by), "--hyps", str(d / "sys" / "hyps.jsonl"), "--inference-mode", "open_vocab",
                    "--iters", "10", "--out", str(d / "out")])
            return 0
        except SystemExit as e:
            return e.code
    check("evaluate: catalog lệch bản khoá -> thoát mã 2", ev() == 2)
    bm["catalog_sha256"] = R.file_sha256(cat)
    check("evaluate: catalog đúng bản khoá -> chấm bình thường", ev() in (0, None))
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
