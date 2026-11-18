#!/usr/bin/env python3
"""Kiểm thử make_dev_split.py, review_priority.py, make_distractors.py và phần trộn hotwords của decode.py
bằng dữ liệu GIẢ trong thư mục tạm (không cần audio, sherpa-onnx, mạng).

  python tests/integration/test_dev_tools.py        # thoát mã != 0 nếu có test hỏng
"""
import contextlib
import csv
import io
import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

from gipbench.core import release
from gipbench.core.paths import REPO as G
from gipbench.infer import decode
from gipbench.dataset.release import make_dev_split as mds
from gipbench.model.systems import hotword_distractors as mdi
from gipbench.dataset.label import review_priority as rp
from gipbench.core.textnorm import Catalog

CATALOG_TSV = ("term\tcategory\tvariants\tphonetic\treserved_b3\n"
               "facebook\tbrand\tfb|face book\t\t0\n"
               "meeting\ttech\t\t\t0\n"
               "pull request\ttech\t\t\t0\n"
               "bilirubin\tmedical\t\t\t1\n")


def jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def rd(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def row(i, split, dom, spk, src, cluster=None, dur=4.0, text="mình dùng <en>meeting</en> nè", **kw):
    r = {"id": i, "split": split, "audio": f"audio/{i}.wav", "duration": dur, "text_tagged": text, "speaker": spk,
         "source": src, "domain": dom}
    if cluster:
        r["speaker_cluster"] = cluster
    r.update(kw)
    return r


def fake_bench():
    rows = []
    for dom, n in (("tech", 14), ("finance", 12)):
        for c in range(n):
            for j in range(3):
                rows.append(row(f"{dom}_{c}_{j}", "B1", dom, f"s_{dom}_{c}_{j}", f"src_{dom}_{c}_{j}", cluster=f"cl_{dom}_{c}"))
    rows.append(row("b2_0", "B2", "product", "s_b2", "src_b2", cluster="cl_tech_0"))  # cụm này có dòng ở B2 -> ép vào test
    for k in range(5):
        rows.append(row(f"b5_{k}", "B5", "read", f"s_b5_{k}", f"src_b5_{k}", text="chỉ tiếng việt"))
    return rows


def units_sides(dev, test):
    return ({mds.unit_of(r) for r in dev}, {mds.unit_of(r) for r in test})


def test_independent_dev_excludes_locked_and_training_videos():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        bench = "bench/locked"
        with patch.object(release, "root", return_value=t / "releases"), patch.object(mds.R, "locked_bench_refs", return_value=[bench]):
            release.write_release(bench, manifest_rows=[
                row("held", "test", "tech", "held-speaker", "https://youtu.be/abcdefghijk", text="held text")
            ])
            jl(t / "train.jsonl", [
                row("train", "train", "finance", "yt_bcdefghijkl", "training", text="training text", cluster="shared")
            ])
            jl(t / "candidates.jsonl", [
                row("from-bench", "dev", "tech", "yt_abcdefghijk", "first", text="có <en>cloud</en> nhé"),
                row("from-train", "dev", "tech", "speaker-one", "youtube:bcdefghijkl", text="có <en>data</en> nhé"),
                row("same-speaker", "dev", "tech", "speaker-two", "unrelated", cluster="shared"),
                row("unknown", "dev", "tech", "unknown", "unknown"),
                row("clean", "dev", "tech", "speaker-three", "third"),
            ])
            out = t / "filtered" / "dev.jsonl"
            args = ["--candidates", str(t / "candidates.jsonl"), "--kind", "cs", "--exclude-from",
                    str(t / "train.jsonl"), "--out-dev", str(out)]
            assert quiet(mds.run, args) == 0
            rows = rd(out)
            assert [r["id"] for r in rows] == ["clean"]
            assert (out.parent / rows[0]["audio"]).resolve() == t / "audio" / "clean.wav"
            report = json.loads(out.with_suffix(".jsonl.report.json").read_text(encoding="utf-8"))
            assert report["verify"][bench]["critical"] == 0
            assert report["filter"]["rows_by_key"]["source_norm"] == 2
            assert not report["qualified"]
            try:
                quiet(mds.run, args + ["--release", "dev/dev_v2_cs@1"])
            except SystemExit as exc:
                assert "release thresholds" in str(exc)
            else:
                raise AssertionError("An underqualified split was registered")
            assert not release.release_dir("dev/dev_v2_cs@1").exists()


def test_independent_vi_release_carries_exclusions():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        bench = "bench/locked"
        dev = "dev/dev_v2_vi@1"
        with patch.object(release, "root", return_value=t / "releases"), patch.object(mds.R, "locked_bench_refs", return_value=[bench]):
            release.write_release(bench, manifest_rows=[
                row("held", "test", "tech", "held-speaker", "held-source", text="lời khác")
            ])
            train = t / "train.jsonl"
            jl(train, [row("train", "train", "edu", "train-speaker", "train-source", text="lời huấn luyện")])
            jl(t / "candidates.jsonl", [
                row("clean", "source", "edu", "speaker-one", "first", dur=3600, text="chúng mình nói tiếng Việt"),
                row("read", "source", "read", "speaker-two", "second", text="xin chào"),
                row("cs", "source", "edu", "speaker-three", "third"),
                row("video", "source", "edu", "yt_abcdefghijk", "fourth", text="hôm nay trời đẹp"),
            ])
            wav = t / "audio" / "clean.wav"
            wav.parent.mkdir()
            wav.touch()
            out = t / "dev.jsonl"
            assert quiet(mds.run, ["--candidates", str(t / "candidates.jsonl"), "--kind", "vi", "--exclude-from",
                                  str(train), "--out-dev", str(out), "--release", dev]) == 0
            registered = release.resolve(dev)
            assert registered.meta["flags"] == {"independent": True, "exclusion_sets": [bench, str(train)]}
            assert [r["id"] for r in rd(registered.manifest)] == ["clean"]


def split_args(d, bench, extra=(), train=(), **kw):
    a = ["--bench", str(bench), "--out-dev", str(d / "dev.jsonl"), "--out-test", str(d / "test.jsonl"), "--seed", "3"]
    if extra:
        a += ["--extra-dev"] + [str(x) for x in extra]
    if train:
        a += ["--train"] + [str(x) for x in train]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}", str(v)]
    return a


# ----------------------------------------------------------------------------- make_dev_split
def test_split_no_shared_unit_and_deterministic():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        jl(t / "bench.jsonl", fake_bench())
        assert quiet(mds.run, split_args(t, t / "bench.jsonl")) == 0
        dev, test = rd(t / "dev.jsonl"), rd(t / "test.jsonl")
        du, tu = units_sides(dev, test)
        assert dev and test and not (du & tu), f"đơn vị chung dev/test: {du & tu}"
        assert all(r["split"] == "dev" and r["bench_split"] == "B1" and r["dev_origin"] == "bench" for r in dev)
        assert {r["id"] for r in dev}.isdisjoint({r["id"] for r in test})
        assert len(dev) + len(test) == len(fake_bench()), "mất / nhân đôi dòng"
        assert "cl_tech_0" in tu, "cụm có dòng ở split khác phải vào test"
        assert all(r["split"] != "dev" for r in test), "test phải giữ split gốc"
        by = {r["id"]: r for r in fake_bench()}
        assert all(by[r["id"]] == r for r in test), "test phải giữ nguyên mọi trường"
        # mỗi domain có cả dev lẫn test
        assert {r["domain"] for r in dev} == {"tech", "finance"}
        first = (t / "dev.jsonl").read_bytes(), (t / "test.jsonl").read_bytes()
        assert quiet(mds.run, split_args(t, t / "bench.jsonl")) == 0
        assert first == ((t / "dev.jsonl").read_bytes(), (t / "test.jsonl").read_bytes()), "chạy lại phải ra đúng kết quả cũ"
        assert (t / "dev.jsonl.report.md").exists()


def test_split_stable_when_new_domain_added():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        base = fake_bench()
        jl(t / "bench.jsonl", base)
        quiet(mds.run, split_args(t, t / "bench.jsonl"))
        old = {r["id"]: "dev" for r in rd(t / "dev.jsonl")}
        more = base + [row(f"edu_{c}_{j}", "B1", "edu", f"s_edu_{c}_{j}", f"src_edu_{c}_{j}", cluster=f"cl_edu_{c}")
                       for c in range(8) for j in range(2)]
        jl(t / "bench2.jsonl", more)
        quiet(mds.run, split_args(t, t / "bench2.jsonl"))
        new = {r["id"] for r in rd(t / "dev.jsonl")}
        assert set(old) <= new and {r["id"] for r in base} & new == set(old), "thêm domain mới làm đổi phía của đơn vị cũ"
        assert any(i.startswith("edu_") for i in new)


def test_split_merges_units_sharing_source():
    rows = [row("a1", "B1", "tech", "s1", "vid1", cluster="c1"), row("a2", "B1", "tech", "s2", "vid1", cluster="c2")] + \
           [row(f"z{k}", "B1", "tech", f"sz{k}", f"vz{k}", cluster=f"cz{k}") for k in range(10)]
    comp = mds.merge_by_source(rows)
    assert comp["c1"] == comp["c2"] and comp["cz0"] != comp["c1"]
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        jl(t / "bench.jsonl", rows)
        assert quiet(mds.run, split_args(t, t / "bench.jsonl", dev_frac=0.5)) == 0
        dev, test = rd(t / "dev.jsonl"), rd(t / "test.jsonl")
        assert {r["source"] for r in dev}.isdisjoint({r["source"] for r in test})


def test_extra_dev_filter_and_domain_cap():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        jl(t / "bench.jsonl", fake_bench())
        train = [row("tr1", "train", "medical", "spk_in_train", "src_in_train")]
        jl(t / "train.jsonl", train)
        ex = [row(f"m{k}", "dev", "medical", f"med_spk_{k}", f"med_src_{k}", dur=10.0) for k in range(120)]
        ex += [row("x_train_spk", "dev", "medical", "spk_in_train", "other_src", dur=10.0),
               row("x_train_src", "dev", "medical", "other_spk", "src_in_train", dur=10.0),
               row("x_test_spk", "dev", "medical", "s_tech_0_0", "other_src2", dur=10.0),   # speaker của một dòng test
               row("x_test_src", "dev", "medical", "other_spk2", "src_b5_0", dur=10.0),     # source của một dòng test
               row("x_other", "dev", "other", "o_spk", "o_src", dur=10.0)]
        jl(t / "extra.jsonl", ex)
        assert quiet(mds.run, split_args(t, t / "bench.jsonl", extra=[t / "extra.jsonl"], train=[t / "train.jsonl"],
                                         max_domain_share=0.4)) == 0
        dev = rd(t / "dev.jsonl")
        ids = {r["id"] for r in dev}
        assert not ids & {"x_train_spk", "x_train_src", "x_test_spk", "x_test_src"}, "extra trùng train / test phải bị loại"
        assert "x_other" in ids
        h = {}
        for r in dev:
            h[r["domain"]] = h.get(r["domain"], 0) + r["duration"]
        tot = sum(h.values())
        med = [r for r in dev if r["domain"] == "medical"]
        assert 0 < len(med) < 120, f"medical phải bị cắt bớt, còn {len(med)}"
        assert h["medical"] / tot <= 0.4 + 1e-9, f"tỉ trọng medical {h['medical'] / tot:.3f} vượt trần"
        assert all(r["dev_origin"] == "extra:extra" and r["bench_split"] == "dev" for r in med)
        # cắt có seed: chạy lại ra đúng tập cũ
        again = tempfile.mkdtemp(dir=t)
        quiet(mds.run, split_args(Path(again), t / "bench.jsonl", extra=[t / "extra.jsonl"], train=[t / "train.jsonl"],
                                  max_domain_share=0.4))
        assert ids == {r["id"] for r in rd(Path(again) / "dev.jsonl")}


def test_split_exit_1_when_dev_overlaps_train():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        jl(t / "bench.jsonl", fake_bench())
        quiet(mds.run, split_args(t, t / "bench.jsonl"))
        dev_spk = rd(t / "dev.jsonl")[0]["speaker"]
        jl(t / "train.jsonl", [row("tr", "train", "tech", dev_spk, "zzz")])
        assert quiet(mds.run, split_args(t, t / "bench.jsonl", train=[t / "train.jsonl"])) == 1
        assert "VI PHẠM" in (t / "dev.jsonl.report.md").read_text(encoding="utf-8")


def test_cap_domains_infeasible_is_skipped():
    fixed = [row("a", "B1", "tech", "s", "c")]
    keep, log, warn = mds.cap_domains(fixed, [("x", row("b", "dev", "medical", "s2", "c2"))], 0.4, 0)
    assert len(keep) == 1 and warn, "2 domain x 0.4 < 1: không cắt, phải cảnh báo"


# ----------------------------------------------------------------------------- review_priority
def prio_fixture():
    rows = [
        row("r1", "B1", "tech", "s1", "c1", text="hôm nay nghe <en>postcard</en>"),
        row("r2", "B1", "tech", "s2", "c2", text="<en>meeting</en> lúc ba giờ"),
        row("r3", "B1", "tech", "s3", "c3", text="<en>deadline</en> tuần sau"),
        row("r4", "B1", "tech", "s4", "c4", text="<en>email</en> cho tôi"),
        row("r5", "B1", "tech", "s5", "c5", text="<en>email</en> cho tôi", auto_tag={"flags": ["suspect:chat"]}),
        row("r6", "B1", "tech", "s6", "c6", text="<en>pull request</en> mới"),
        row("r7", "B1", "tech", "s7", "c7", text="<en>podcast</en> hay"),
        row("r8", "B5", "read", "s8", "c8", text="không có tiếng anh"),
    ]
    hy = {
        "gip": {"r1": "hôm nay nghe PODCAST", "r2": "MÍT TINH lúc ba giờ", "r3": "DEADLINE tuần sau", "r4": "EMAIL cho tôi",
                "r5": "email cho tôi", "r6": "pullrequest mới", "r7": "pót cát hay"},
        "whisper": {"r1": "hôm nay nghe podcast", "r2": "mi ting lúc ba giờ", "r3": "deadline tuần sau", "r4": "email cho tôi",
                    "r5": "email cho tôi", "r6": "pull request mới", "r7": "pót cát hay"},
        "qwen": {"r1": "Hôm nay nghe podcast.", "r2": "mịt tinh lúc ba giờ", "r3": "đét lai tuần sau", "r4": "email cho tôi",
                 "r5": "email cho tôi", "r6": "pull re quest mới"},   # qwen không có hyp của r7
    }
    return rows, hy


def test_review_priority_order_on_hand_cases():
    rows, hy = prio_fixture()
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "cat.tsv").write_text(CATALOG_TSV, encoding="utf-8")
        cat = Catalog.load(str(Path(t) / "cat.tsv"))
    ranked = rp.rank(rows, hy, cat)
    ids = [r["id"] for r, _, _ in ranked]
    res = {r["id"]: x for r, x, _ in ranked}
    assert "r8" not in ids, "câu không có <en> phải bị bỏ"
    assert ids == ["r1", "r7", "r2", "r3", "r5", "r4", "r6"], ids
    assert res["r1"]["level"] == "P1" and res["r1"]["score"] == 14.0, res["r1"]       # mọi hệ bỏ + đồng thuận 'podcast'
    assert res["r2"]["score"] == 10.0 and res["r2"]["spans"][0]["form_n"] == 0, res["r2"]  # mọi hệ bỏ, mỗi hệ một kiểu
    assert res["r7"]["n_sys"] == 2, "hệ thiếu hyp phải bị bỏ khỏi phép đếm"
    assert res["r7"]["spans"][0]["miss"] == ["gip", "whisper"] and res["r7"]["score"] > res["r2"]["score"]
    assert 3.0 < res["r3"]["score"] < 4.0 and res["r3"]["spans"][0]["miss"] == ["qwen"]  # 1/3 hệ bỏ
    assert res["r5"]["score"] == 2.0 and res["r4"]["score"] == 0.0, "cờ suspect cộng điểm"
    assert res["r6"]["score"] == 0.0, "ghép bỏ dấu cách 'pull re quest' ~ 'pull request' phải tính là có"
    assert "postcard" in rp.reason_of(res["r1"]) and "podcast" in rp.reason_of(res["r1"])


def test_review_priority_csv_matches_review_columns():
    rows, hy = prio_fixture()
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        jl(t / "bench.jsonl", rows)
        (t / "cat.tsv").write_text(CATALOG_TSV, encoding="utf-8")
        args = ["--bench", str(t / "bench.jsonl"), "--catalog", str(t / "cat.tsv"), "--splits", "B1", "B5", "--top", "3",
                "--out", str(t / "p.csv"), "--hyps"]
        for n, h in hy.items():
            jl(t / f"{n}.jsonl", [{"id": k, "hyp": v} for k, v in h.items()])
            args.append(f"{n}={t / (n + '.jsonl')}")
        assert quiet(rp.run, args) == 0
        got = list(csv.DictReader(open(t / "p.csv", encoding="utf-8-sig")))
        assert [r["id"] for r in got] == ["r1", "r7", "r2"], [r["id"] for r in got]
        pr = [float(r["priority"]) for r in got]
        assert pr == sorted(pr, reverse=True)
        assert set(csv.DictReader(open(t / "p.csv", encoding="utf-8-sig")).fieldnames) >= {
            "id", "split", "audio", "duration", "text_tagged", "text_tagged_orig", "confidence", "flags", "decision",
            "text_tagged_fixed", "viet_levels", "reading_note", "region", "gender", "note", "priority", "reason", "hint"}
        assert got[0]["decision"] == "" and "gip:" in got[0]["hint"] and "||" in got[0]["hint"]
        assert (t / "p.csv.report.md").exists()


# ----------------------------------------------------------------------------- make_distractors
def test_distractors_exclude_bench_variants_reserved():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "cat.tsv").write_text(CATALOG_TSV, encoding="utf-8")
        jl(t / "train.jsonl", [row("t1", "train", "tech", "a", "b",
                                   text="<en>fb</en> <en>zoom</en> <en>bilirubin</en> <en>meeting</en> <en>zoom</en> "
                                        "<en>bilirubin test</en> <en>face book</en> <en>cà phê</en> <en>x</en>")])
        (t / "words.txt").write_text("# ghi chú\nzoom\nslack\nFacebook\nbilirubin\nMeeting\n", encoding="utf-8")
        out = t / "dis.tsv"
        assert quiet(mdi.run, ["--catalog", str(t / "cat.tsv"), "--manifest", str(t / "train.jsonl"),
                               "--wordlist", str(t / "words.txt"), "--out", str(out)]) == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("# term\ttotal\ttrain.jsonl\twords.txt")
        got = {l.split("\t")[0]: l.split("\t")[1:] for l in lines[1:]}
        assert set(got) == {"zoom", "slack"}, got
        assert got["zoom"] == ["3", "2", "1"] and got["slack"] == ["1", "0", "1"]
        assert set(decode.read_distractors(out)) == {"zoom", "slack"}, "decode phải đọc được file (bỏ dòng tiêu đề #)"
        for banned in ("fb", "face book", "facebook", "meeting", "bilirubin", "bilirubin test", "x", "cà phê"):
            assert banned not in got, banned


# ----------------------------------------------------------------------------- decode.py: trộn hotwords
def test_mix_hotwords():
    real = ["FACEBOOK", "MEETING"]
    pool = [f"WORD{i:02d}" for i in range(50)] + ["facebook", "Meeting", "WORD01", "word02"]
    full, info = decode.mix_hotwords(real, pool)
    assert full[:2] == real and info == {"n_real": 2, "n_distractors": 50, "n_distractor_pool": 50}, info
    assert len({t.casefold() for t in full}) == len(full), "không được trùng thuật ngữ thật / trùng nhau"
    a, ia = decode.mix_hotwords(real, pool, 7, seed=1)
    b, _ = decode.mix_hotwords(real, list(reversed(pool)), 7, seed=1)
    c, _ = decode.mix_hotwords(real, pool, 7, seed=2)
    assert a == b, "cùng seed phải ra cùng mẫu, không phụ thuộc thứ tự file"
    assert a != c, "seed khác phải ra mẫu khác"
    assert ia["n_distractors"] == 7 and len(a) == 9 and set(a[2:]) <= {f"WORD{i:02d}" for i in range(50)}
    assert decode.mix_hotwords(real, pool, 0)[0] == real
    assert decode.mix_hotwords(real, pool, 999)[0] == full
    try:
        decode.mix_hotwords(real, pool, -1)
        raise AssertionError("n < 0 phải báo lỗi")
    except ValueError:
        pass


def test_write_hotwords_unchanged_without_distractors():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        (t / "cat.tsv").write_text(CATALOG_TSV, encoding="utf-8")
        cat = Catalog.load(str(t / "cat.tsv"))
        n = decode.write_hotwords(cat, t / "h.txt", "upper", include_reserved=True)
        assert n == 4 and (t / "h.txt").read_text(encoding="utf-8") == "FACEBOOK\nMEETING\nPULL REQUEST\nBILIRUBIN\n"
        n = decode.write_hotwords(cat, t / "h2.txt", "lower", include_reserved=False)
        assert n == 3 and (t / "h2.txt").read_text(encoding="utf-8") == "facebook\nmeeting\npull request\n"


def main():
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    fails = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            fails += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - fails}/{len(tests)} test đạt")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
