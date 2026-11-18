#!/usr/bin/env python3
"""package.py: chính sách share (IP-08) bằng dữ liệu giả. Không cần mạng / dữ liệu thật.

  python tests/dataset/test_package.py

Kiểm: allowlist + regex cấm làm dừng đóng gói và không để lại gói dở; datasets.yaml của gói không có bộ riêng tư;
bộ trong text_not_redistributable bị bỏ lời thoại; distinct_utterances gộp đúng câu trùng (C1 ⊂ G0, CV17 ∩ CV27);
release.json có scorer_code_sha256 trùng hash evaluate tính; docs bỏ khối share:omit.
"""
import json
import sys
import tempfile
from pathlib import Path

import yaml

from gipbench.bench import evaluate as EVAL
from gipbench.dataset.release import package as K

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({info})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def row(i, dataset, up, sha, **kw):
    return dict({"id": f"t{i}", "dataset": dataset, "upstream_id": up, "audio_sha256": sha, "text_tagged": f"câu {i}",
                 "speaker": f"s{i}", "license": "CC0-1.0", "audio": f"a/{i}.wav"}, **kw)


ROWS = [
    row(1, "commonvoice_vi_27", "common_voice_vi_100.mp3", "a" * 64),
    row(2, "cv17_vi", "vi/clips/common_voice_vi_100.mp3", "b" * 64),     # cùng clip CV, sha khác (mirror)
    row(3, "vivos", "VIVOSSPK01_R001", "c" * 64),
    row(4, "vivos", "VIVOSSPK01_R002", "c" * 64),                         # cùng sha với t3
    row(5, "vimd", "vimd_0001", "d" * 64, text_orig="Câu 5"),
    row(6, "vimd", "vimd_0002", "e" * 64),
]


def run_main(argv):
    try:
        K.main(argv)
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def main():
    # --- đơn vị ---
    check("distinct_utterances: CV17∩CV27 theo mã clip, trùng sha256 gộp, còn lại tách", K.distinct_utterances(ROWS) == 4,
          K.distinct_utterances(ROWS))
    g0 = [row(i, "ggroup_tech", f"g{i}", f"{i:064x}") for i in range(5)]
    c1 = [dict(r, id="c" + r["id"]) for r in g0[:3]]
    check("distinct_utterances: C1 là tập con của G0 -> không cộng thêm", K.distinct_utterances(g0 + c1) == 5)
    doc = ("a\n<!-- share:omit -->\nlịch sử v0_draft\n<!-- /share:omit -->\nb x<!-- share:omit --> nội bộ<!-- /share:omit -->"
           " y\n| r1 | v0_draft | <!-- share:omit-line -->\n| r2 | ok |\n")
    check("omit_share_blocks: khối / giữa dòng / cả dòng", K.omit_share_blocks(doc) == "a\nb x y\n| r2 | ok |\n",
          repr(K.omit_share_blocks(doc)))
    cfg = yaml.safe_load((K.SHARE_CODE_CONFIG).read_text(encoding="utf-8"))
    check("share_code.scorer_files trùng evaluate.SCORER_FILES", cfg["scorer_files"] == EVAL.SCORER_FILES,
          (cfg["scorer_files"], EVAL.SCORER_FILES))
    pol = K.share_policy()
    text = (K.P.REPO / "configs" / "bench" / "datasets.yaml").read_text(encoding="utf-8")
    f = yaml.safe_load(K.filtered_datasets_yaml(text, ["vivos", "vimd"]))
    check("filtered_datasets_yaml chỉ giữ bộ được đóng gói", list(f["datasets"]) == ["vivos", "vimd"] and "models" in f)
    try:
        K.filtered_datasets_yaml(text, ["khong_co"])
        check("filtered_datasets_yaml: bộ không có trong registry -> dừng", False)
    except SystemExit:
        check("filtered_datasets_yaml: bộ không có trong registry -> dừng", True)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # --- scan_package ---
        root = td / "scan"
        (root / "code" / "src" / "gipbench").mkdir(parents=True)
        (root / "README.md").write_text("ok\n", encoding="utf-8")
        check("scan_package: gói sạch -> 0 vi phạm", K.scan_package(root, pol) == [], K.scan_package(root, pol))
        (root / "code" / "src" / "gipbench" / "x.py").write_text("from gipbench.model import run\n", encoding="utf-8")
        (root / "notes.txt").write_text("x\n", encoding="utf-8")
        bad = K.scan_package(root, pol)
        check("scan_package: file import gipbench.model -> vi phạm", any("x.py:1" in b for b in bad), bad)
        check("scan_package: file ngoài allowlist -> vi phạm", any(b.startswith("notes.txt") for b in bad), bad)
        (root / "code" / "src" / "gipbench" / "x.py").write_text("p = '/mnt/d/code'  # C:\\Users\\a\n", encoding="utf-8")
        check("scan_package: đường dẫn máy -> vi phạm", len(K.scan_package(root, pol)) == 3, K.scan_package(root, pol))

        # --- main end-to-end trên benchmark giả ---
        man = td / "m.jsonl"
        man.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS), encoding="utf-8")
        orig_mp = K.manifest_path
        K.manifest_path = lambda name: (man, {"evidence_level": "draft", "track": "G0"})
        try:
            out = td / "share"
            rc = run_main(["--benchmarks", "general_v1", "--name", "fake-v9.9", "--out", str(out)])
            dst = out / "fake-v9.9"
            check("đóng gói giả: thoát 0, có gói, không còn thư mục tạm",
                  rc == 0 and dst.is_dir() and not (out / ".fake-v9.9.partial").exists(), rc)
            rel = json.loads((dst / "release.json").read_text(encoding="utf-8"))
            check("release.json: distinct_utterances", rel["distinct_utterances"] == 4, rel.get("distinct_utterances"))
            check("release.json: scorer_code_sha256 trùng evaluate.scorer_code_sha256()",
                  rel["scorer_code_sha256"] == EVAL.scorer_code_sha256())
            check("release.json: ghi bộ bị giữ lời thoại", rel["text_withheld_datasets"] == ["vimd"], rel["text_withheld_datasets"])
            got = [json.loads(l) for l in open(dst / "manifests" / "general_v1.jsonl", encoding="utf-8")]
            vimd = [r for r in got if r["dataset"] == "vimd"]
            check("bộ text_not_redistributable: bỏ text_tagged / text_orig, không thêm trường",
                  all("text_tagged" not in r and "text_orig" not in r and "text_redistributable" not in r for r in vimd))
            check("bộ khác giữ lời thoại", all("text_tagged" in r for r in got if r["dataset"] != "vimd"))
            for p in (dst / "configs" / "datasets.yaml", dst / "code" / "configs" / "bench" / "datasets.yaml"):
                ds = yaml.safe_load(p.read_text(encoding="utf-8"))["datasets"]
                check(f"{p.relative_to(dst).as_posix()}: không có consensus_private, chỉ bộ được đóng gói",
                      "consensus_private" not in ds and set(ds) == {r["dataset"] for r in ROWS}, sorted(ds))
            proto = (dst / "docs" / "benchmark_protocol_v1.md").read_text(encoding="utf-8")
            check("docs trong gói: bỏ khối share:omit", "share:omit" not in proto and "v0_draft" not in proto)
            check("gói đạt chính sách khi quét lại", K.scan_package(dst, pol) == [], K.scan_package(dst, pol)[:3])

            # vi phạm -> dừng, không để lại gì
            orig_pol = K.share_policy
            K.share_policy = lambda: dict(orig_pol(), forbidden=orig_pol()["forbidden"] + [__import__("re").compile("VIVOSSPK01")])
            try:
                rc = run_main(["--benchmarks", "general_v1", "--name", "fake-v9.10", "--out", str(out)])
            finally:
                K.share_policy = orig_pol
            check("nội dung chạm regex cấm -> dừng, không có gói và thư mục tạm",
                  rc not in (0, None) and not (out / "fake-v9.10").exists() and not (out / ".fake-v9.10.partial").exists(), rc)
            rc = run_main(["--benchmarks", "general_v1", "--name", "fake-v9.9", "--out", str(out)])
            check("không ghi đè gói đã có", rc not in (0, None))
        finally:
            K.manifest_path = orig_mp
    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
