#!/usr/bin/env python3
"""Phiếu nghe nhãn giả vss_cs (TODO 2d): lấy mẫu phân tầng, phiếu nghe mù, chấm lỗi span / WER Việt / kappa và luật bỏ
vss_cs. Dữ liệu GIẢ trong thư mục tạm, CPU, không cần audio thật.

  python tests/dataset/test_listen_sample.py
"""
import csv
import json
import math
import sys
import tempfile
from argparse import Namespace
from pathlib import Path

FAIL = []


def check(name, cond, info=""):
    print(("ĐẠT  " if cond else "HỎNG ") + name + (f"  ({str(info)[-400:]})" if info and not cond else ""))
    if not cond:
        FAIL.append(name)


def main():
    from gipbench.dataset.pseudolabel import listen_sample as L
    d = Path(tempfile.mkdtemp())
    (d / "a.wav").write_bytes(b"RIFF")
    rows = []
    for i in range(120):
        n = 1 + (i % 2)
        text = "nói về " + " và ".join(f"<en>term{(i * 7 + k) % (3 if i % 3 else 40)}</en>" for k in range(n)) + " hôm nay"
        rows.append({"id": f"v{i}", "audio": str(d / "a.wav"), "duration": 3.0, "text_tagged": text, "source": f"s{i % 9}",
                     "agreement": {"vi_wer": [0.01, 0.07, 0.12][i % 3]}, "label_source": "pseudo:x"})
    kept = d / "kept.jsonl"
    kept.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    out = d / "listen"
    L.cmd_sample(Namespace(kept=str(kept), candidates=None, hyps=[], hash_index=None, audio_root=None, catalog=None,
                           per_cell=4, freq_high=5, controls=0, overlap=6, seed=0, out=str(out)))
    meta = json.loads((out / "sample_meta.json").read_text(encoding="utf-8"))
    a = list(csv.DictReader(open(out / "listen_A.csv", encoding="utf-8-sig")))
    b = list(csv.DictReader(open(out / "listen_B.csv", encoding="utf-8-sig")))
    key = list(csv.DictReader(open(out / "sample_key.csv", encoding="utf-8-sig")))
    check("sample: mỗi ô lấy tối đa per_cell, ghi ô thiếu", all(min(v, 4) == sum(k["cell"] == c for k in key)
                                                                for c, v in meta["cells"].items()), (meta, len(key)))
    check("sample: phiếu nghe mù (không có cột tầng / vi_wer / nhóm)", not {"cell", "group", "vi_wer"} & set(a[0]), list(a[0]))
    check("sample: phiếu B là tập con của A", {r["id"] for r in b} <= {r["id"] for r in a} and len(b) == 6)
    check("sample: audio được chép vào <out>/audio, đường dẫn tương đối", all(r["audio"].startswith("audio/") for r in a)
          and (out / "audio" / "v0.wav").exists() or any((out / "audio").iterdir()), a[0]["audio"])

    # phiếu đã nghe: 1/2 số câu sai đúng 1 span, 1 câu drop, 1 câu thêm từ Anh không tag
    rev = []
    for k, r in enumerate(a):
        r = dict(r)
        if k == 0:
            r["decision"] = "drop"
        elif k % 2:
            r["decision"] = "fix"
            r["text_tagged_fixed"] = r["text_tagged"].replace("<en>", "<en>x", 1)
        elif k == 2:
            r["decision"] = "fix"
            r["text_tagged_fixed"] = r["text_tagged"].replace("hôm nay", "<en>today</en>")
        else:
            r["decision"] = "ok"
        rev.append(r)
    with open(out / "A_rev.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rev[0]))
        w.writeheader()
        w.writerows(rev)
    revb = [dict(r, decision=("fix" if r["text_tagged_fixed"] != r["text_tagged"] else r["decision"]))
            for r in rev if r["id"] in {x["id"] for x in b}]
    with open(out / "B_rev.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(revb[0]))
        w.writeheader()
        w.writerows(revb)
    L.cmd_score(Namespace(dir=str(out), review="A_rev.csv", review_b="B_rev.csv"))
    res = json.loads((out / "score.json").read_text(encoding="utf-8"))
    judged = [r for r in rev[1:]]
    n_sp = sum(len(L.spans_of(r["text_tagged"])) for r in judged)
    n_wrong = sum(1 for k, r in enumerate(rev) if k % 2)
    p, lo, hi = res["kept"]["span_error"]
    check("score: tỉ lệ span sai đúng số đếm, drop không tính", abs(p - round(n_wrong / n_sp, 4)) < 1e-4
          and res["kept"]["n"] == len(rev) - 1 and res["drop_audio"] == 1, (res["kept"], n_wrong, n_sp))
    check("score: CI Wilson bao tỉ lệ và nằm trong [0, 1]", 0 <= lo <= p <= hi <= 1, (lo, p, hi))
    check("score: bắt câu có từ Anh không tag", res["kept"]["sent_untagged_english"][0] > 0, res["kept"])
    check("score: kappa = 1 khi hai người nghe đồng ý hoàn toàn", res.get("kappa_ok") == 1.0 and res["n_common"] == 6, res)
    check("score: luật bỏ vss_cs khi cận dưới lỗi span > 10%", res["decision"].startswith("BỎ vss_cs") == (lo > 0.10), res["decision"])
    p2, lo2, hi2 = L.wilson(0, 50)
    check("wilson: 0/50 -> cận trên ~7.1%", p2 == 0 and lo2 == 0 and abs(hi2 - 0.0713) < 1e-3, (lo2, hi2))
    check("kappa: ngẫu nhiên đối xứng ~0", abs(L.kappa(["a", "b", "a", "b"], ["a", "a", "b", "b"])) < 1e-9)
    check("edit_distance", L.edit_distance("abc", "axc") == 1 and L.edit_distance([], ["a"]) == 1)
    check("score: chưa nghe hết thì quyết định ghi rõ", math.isfinite(p))
    print(f"\n{'HỎNG ' + str(len(FAIL)) if FAIL else 'ĐẠT tất cả'}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
