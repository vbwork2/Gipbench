#!/usr/bin/env python3
"""catalog/train-v1 (dataset.release.build_train_catalog), đề xuất thuật ngữ giữ riêng (propose_reserved_terms) và chốt chặn
của make_text (catalog from_bench, --reserved-terms bắt buộc). Dữ liệu GIẢ trong thư mục tạm, CPU (audit IP-05).

  python tests/dataset/test_train_catalog.py        # cần pyyaml
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from gipbench.core import paths
from gipbench.core import release as R
from gipbench.core.decontam import read_lines
from gipbench.core.paths import REPO as G
from gipbench.core.textnorm import Catalog

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def row(i, text, domain="other", **kw):
    return dict({"id": i, "dataset": "x", "domain": domain, "speaker": f"s_{i}", "source": f"src_{i}", "split": "train",
                 "duration": 2.0, "text_tagged": text}, **kw)


def main():
    from gipbench.dataset.release import build_train_catalog as B
    from gipbench.dataset.release import propose_reserved_terms as P
    d = Path(tempfile.mkdtemp())
    os.environ["GIPBENCH_DATA"] = str(d / "data")
    os.environ["GIPBENCH_RELEASES_YAML"] = str(d / "releases.yaml")
    paths._cache = None
    R.write_release("bench/fb1", manifest_rows=[row("c0_1", "uống <en>vitamin</en> mỗi ngày", "medical"),
                                                row("c0_2", "bị <en>stress</en> kéo dài", "medical")])
    R.write_release("pool/tp@v1", manifest_rows=[row("p1", "tiêm <en>vaccine</en> phòng bệnh", "medical", text_tagged_orig="tiêm <en>vaccine</en> phòng bệnh"),
                                                 row("p2", "uống <en>vitamin</en> buổi sáng", "medical"),
                                                 row("p3", "chạy <en>kubernetes</en> trên máy chủ", "tech"),
                                                 row("p4", "dùng <en>docker</en> để đóng gói", "tech"),
                                                 row("p5", "không có từ nào cả", "other")])
    (d / "releases.yaml").write_text(yaml.safe_dump({"releases": {"bench/fb1": {"flags": {"locked": True}},
                                                                  "pool/tp@v1": {}}}), encoding="utf-8")
    res = d / "reserved.txt"
    res.write_text("# giữ riêng\nkubernetes\n", encoding="utf-8")

    # ---- build_train_catalog
    kept, meta = B.build(["pool/tp@v1:text_tagged"], str(res))
    check("catalog = thuật ngữ train trừ thuật ngữ bench trừ thuật ngữ giữ riêng", sorted(kept) == ["docker", "vaccine"], sorted(kept))
    check("đếm từng bước: 4 thuật ngữ train, loại 1 (bench: vitamin), loại 1 (giữ riêng: kubernetes)",
          (meta["n_train_terms"], meta["n_removed_bench"], meta["n_removed_reserved"], meta["n_catalog"]) == (4, 1, 1, 2), meta)
    check("meta ghi bench đã loại và số thuật ngữ bench cũng có trong train", meta["benches_excluded"] == [
        {"bench": "bench/fb1", "span_terms": 2, "also_in_train": 1}], meta["benches_excluded"])
    out = d / "cat" / "catalog.tsv"
    B.write_catalog(kept, out)
    cat = Catalog.load(str(out))
    check("catalog.tsv đọc được bằng Catalog.load; category lấy domain phổ biến nhất, reserved_b3 = 0",
          {" ".join(k): v.category for k, v in cat.terms.items()} == {"docker": "tech", "vaccine": "medical"}
          and not any(v.reserved_b3 for v in cat.terms.values()))
    m = B.main(["--source", "pool/tp@v1", "--reserved-terms", str(res), "--out", str(d / "cat2" / "catalog.tsv")])
    mj = json.loads((d / "cat2/catalog.tsv.meta.json").read_text(encoding="utf-8"))
    check("main ghi catalog + meta.json (catalog_sha256 khớp file)", mj["catalog_sha256"] == hashlib.sha256((d / "cat2/catalog.tsv").read_bytes()).hexdigest()
          and mj["n_catalog"] == m["n_catalog"] == 2)
    B.main(["--source", "pool/tp@v1", "--reserved-terms", str(res), "--preview"])
    check("--preview không ghi file", not (d / "cat3").exists())
    empty = d / "empty.txt"
    empty.write_text("# chưa chốt\n", encoding="utf-8")
    try:
        B.build(["pool/tp@v1"], str(empty))
        check("reserved_terms rỗng: từ chối dựng catalog", False)
    except SystemExit as e:
        check("reserved_terms rỗng: từ chối dựng catalog", "rỗng" in str(e), str(e))
    rows, field = B.load_source("pool/tp@v1:text_tagged_orig")
    check("load_source: tách NGUỒN:TRƯỜNG; không có :trường thì text_tagged", field == "text_tagged_orig" and len(rows) == 5
          and B.load_source("pool/tp@v1")[1] == "text_tagged")
    p = d / "m.jsonl"
    p.write_text(json.dumps(row("q", "xin <en>chào</en>")) + "\n", encoding="utf-8")
    check("load_source: đường dẫn manifest thường (kể cả :trường) đọc được", B.load_source(str(p))[1] == "text_tagged"
          and B.load_source(str(p) + ":text_tagged")[1] == "text_tagged")
    # catalog dựng xong, đăng ký thành release train, không bị chặn
    R.write_release("catalog/train-v1", catalog_src=out, flags={"from_bench": False})
    check("catalog/train-v1 (from_bench: false) qua forbid_bench_catalog", R.forbid_bench_catalog("catalog/train-v1", "thử") == {"from_bench": False})

    # ---- propose_reserved_terms
    terms, picked, st = P.propose(["bench/fb1"], ["pool/tp@v1"], 1.0, "salt")
    check("đề xuất: thuật ngữ bench đã có trong pool (vitamin) không được giữ riêng (sẽ buộc loại câu train); stress thì được",
          picked == {"stress"} and st["seen_in_pools"] == 1 and st["candidates"] == 1, (picked, st))
    check("fraction 0 không giữ riêng thuật ngữ nào; ổn định theo salt",
          P.propose(["bench/fb1"], ["pool/tp@v1"], 0.0, "salt")[1] == set() and P.stable_frac("stress", "a") == P.stable_frac("stress", "a")
          and P.stable_frac("stress", "a") != P.stable_frac("stress", "b"))
    prop = d / "proposal.txt"
    P.main(["--bench", "bench/fb1", "--pool", "pool/tp@v1", "--fraction", "1.0", "--out", str(prop)])
    txt = prop.read_text(encoding="utf-8")
    check("file đề xuất: đọc được bằng read_lines (bỏ chú thích), ghi rõ CHƯA CHỐT", read_lines(str(prop)) == ["stress"] and "CHƯA CHỐT" in txt, txt)

    # ---- make_text: chốt chặn
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    cb = d / "cat_bench.tsv"
    cb.write_text("term\tcategory\tvariants\tphonetic\treserved_b3\nstress\tmedical\t\t\t0\n", encoding="utf-8")
    R.write_release("catalog/benchcat", catalog_src=cb, flags={"from_bench": True})

    def mt(*a):
        r = subprocess.run([sys.executable, "-m", "gipbench.dataset.synth.make_text", *map(str, a)], capture_output=True, text=True,
                           encoding="utf-8", cwd=G, env=env)
        return r.returncode, r.stdout + r.stderr
    rc, out_ = mt("--catalog", cb, "--reserved-terms", res, "--out-dir", d / "mt1")
    check("make_text từ chối catalog from_bench", rc != 0 and "from_bench" in out_, out_)
    rc, out_ = mt("--catalog", out, "--out-dir", d / "mt2")
    check("make_text: thiếu --reserved-terms bị từ chối (bắt buộc)", rc == 2 and "reserved-terms" in out_, out_)
    rc, out_ = mt("--catalog", out, "--reserved-terms", empty, "--out-dir", d / "mt3")
    check("make_text: --reserved-terms rỗng bị từ chối", rc != 0 and "rỗng" in out_, out_)
    rc, out_ = mt("--catalog", out, "--reserved-terms", res, "--out-dir", d / "mt4", "--holdout-frac", "0", "--no-templates")
    check("make_text với catalog train hợp lệ qua được chốt chặn (kịch bản rỗng vì --no-templates)", "from_bench" not in out_ and "reserved-terms" not in out_, out_)

    print(f"\n{len(FAIL)} ca hỏng" if FAIL else "\nTất cả đạt")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
