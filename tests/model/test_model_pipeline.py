#!/usr/bin/env python3
"""Kiểm thử pipeline model + hợp đồng release / system bằng dữ liệu GIẢ (không cần GPU, icefall, lhotse, dữ liệu thật).

  python tests/model/test_model_pipeline.py        # thoát mã != 0 nếu có ca sai

Gồm: experiment YAML (e3_v1 / e3_v1b / e3_v2), tách nhóm (so với logic heredoc cũ), icefall_runner (watchdog với tiến
trình giả: RAM thấp -> resume, lỗi 2 lần, ổ đầy, dọn checkpoint), model.run --dry-run, release (ghi / đọc / bất biến /
materialize), bench.run (system YAML -> tham số decode).
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path

from gipbench.core.paths import REPO as G

FAILS = []


def check(name, cond, info=""):
    print(("  ok   " if cond else "  SAI  ") + name + ("" if cond else f"  [{info}]"))
    if not cond:
        FAILS.append(name)


def jl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_experiments():
    print("experiment YAML")
    from gipbench.model.recipe import experiment as X
    e1 = X.load(G / "configs/model/experiments/e3_v1.yaml")
    e1b = X.load(G / "configs/model/experiments/e3_v1b.yaml")
    e2 = X.load(G / "configs/model/experiments/e3_v2.yaml")
    w = lambda e: {g["group"]: g["weight"] for g in X.groups(e)}  # noqa: E731
    check("e3_v1 trọng số = train_mix.tsv lúc chạy (5171980)", w(e1) == {"cs_real": 0.35, "cs_pseudo": 0.10, "synth": 0.05,
                                                                       "replay_human": 0.25, "distill": 0.20, "confusable": 0.05})
    check("e3_v1b chỉ khác e3_v1 ở trần domain", e1["curate"]["max_domain_share"] == 0.35
          and e1b["curate"]["max_domain_share"] == 0.50 and w(e1) == w(e1b) and e1["pools"] == e1b["pools"])
    check("e3_v2 trọng số = train_mix.tsv cuối", w(e2)["cs_real"] == 0.20 and w(e2)["synth"] == 0.20)
    check("thứ tự input = TRAIN_IN của run_e3_v1b.sh", X.input_refs(e1b) == [
        "pool/vimedcss_train@full", "pool/vss_cs@e3_v1", "pool/vivos_train@e3_v1", "pool/cv_train@v1", "pool/distill@e3_v1",
        "pool/confusable_train@e3_v1"])
    o = X.load(G / "configs/model/experiments/e3_v1b.yaml", ["train.lr=0.001", "pools.synth.weight=0.3"])
    check("--set ghi đè", o["train"]["lr"] == 0.001 and w(o)["synth"] == 0.3)
    os.environ["MIX_SYNTH"] = "0"
    check("MIX_* vẫn ghi đè trọng số", w(X.load(G / "configs/model/experiments/e3_v1b.yaml"))["synth"] == 0.0)
    del os.environ["MIX_SYNTH"]
    check("${exp_root} được thay", "${" not in e1b["output"]["exp_dir"] and e1b["output"]["exp_dir"].endswith("/e3_v1b"))


def old_heredoc_groups(manifest, mix_rows):
    groups = {}
    for g, ds in mix_rows:
        for d in ds.split(","):
            groups[d.strip().lower()] = g
    out = {}
    for l in open(manifest, encoding="utf-8"):
        r = json.loads(l)
        d = r["dataset"].split(":")[0].lower()
        g = groups.get(d) or ("distill" if r.get("distill") else None)
        if g is None:
            raise SystemExit("x")
        out.setdefault(g, []).append(l)
    return out


def test_split_groups(T):
    print("tách nhóm (so với heredoc stage 2 cũ)")
    from gipbench.model.recipe import experiment as X
    from gipbench.model.recipe.split_groups import split
    rows = [{"id": "a", "dataset": "ViMedCSS:train", "duration": 3}, {"id": "b", "dataset": "vss_cs", "duration": 2},
            {"id": "c", "dataset": "vivos:train", "duration": 1}, {"id": "d", "dataset": "commonvoice", "duration": 1},
            {"id": "e", "dataset": "VSS:train", "distill": True, "duration": 4}, {"id": "f", "dataset": "confusable", "duration": 1}]
    jl(T / "cur.jsonl", rows)
    groups = X.groups(X.load(G / "configs/model/experiments/e3_v1b.yaml"))
    files = split(T / "cur.jsonl", groups, T / "g")
    old = old_heredoc_groups(T / "cur.jsonl", [(g["group"], ",".join(g["datasets"])) for g in groups])
    new = {g: open(p, encoding="utf-8").readlines() for g, p in files.items()}
    check("giống heredoc cũ từng dòng", new == old, (sorted(new), sorted(old)))
    check("distill: true -> nhóm distill", [json.loads(l)["id"] for l in new["distill"]] == ["e"])
    jl(T / "bad.jsonl", [{"id": "x", "dataset": "la", "duration": 1}])
    try:
        split(T / "bad.jsonl", groups, T / "g2")
        check("dataset lạ -> lỗi", False)
    except SystemExit as e:
        check("dataset lạ -> lỗi", "không biết nhóm" in str(e))


FAKE = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    a = sys.argv[1:]
    exp = Path(a[a.index("--exp-dir") + 1])
    mode = Path(exp / "mode").read_text().strip()
    print("args", " ".join(a), flush=True)
    if mode == "fail":
        sys.exit(3)
    if "--start-batch" in a:
        print("Done!", flush=True)
        sys.exit(0)
    (exp / "checkpoint-5.pt").write_text("x")
    (exp / "checkpoint-3.pt").write_text("x")
    time.sleep(30)
''')


def test_icefall_runner(T):
    print("icefall_runner (tiến trình giả)")
    from gipbench.model.train import icefall_runner as IR
    check("STEPS / SEG / EPS như shell", IR.steps_per_epoch(160, 150) == 4129 and IR.segment_plan(4129, 4) == (1033, 4132))
    d = T / "ck"
    d.mkdir()
    for n in (1033, 2066, 4132, 5000):
        (d / f"checkpoint-{n}.pt").write_text("x")
    IR.cleanup(d, 4132)
    check("dọn checkpoint: giữ bội EPS + mới nhất", IR.ckpt_nums(d) == [4132, 5000], IR.ckpt_nums(d))
    check("resume args", IR.resume_args(d, "init.pt") == ["--do-finetune", "0", "--start-epoch", "2", "--num-epochs", "1",
                                                          "--start-batch", "5000"])
    fake = T / "fake_ft.py"
    fake.write_text(FAKE, encoding="utf-8")

    def runner(name, mode, mem, disk=100):
        e = T / name
        e.mkdir()
        (e / "mode").write_text(mode)
        wd = {"poll_s": 0.2, "min_ram_mb": 1024, "ram_grace_s": 0, "min_free_disk_gb": 3, "hang_s": 600}
        r = IR.Runner(lambda res: [sys.executable, str(fake), *res, "--exp-dir", str(e)], cwd=T, env=None, exp_dir=e,
                      eps=4, log=e / "finetune.log", seg_log=e / "segments.log", watchdog=wd, disk_path=str(T),
                      mem_fn=lambda: mem, disk_fn=lambda p: disk, sleep=lambda s: None)
        return r.run("init.pt", max_segments=5), e
    res, e = runner("ram", "ok", mem=500)
    seg = (e / "segments.log").read_text(encoding="utf-8")
    check("RAM thấp -> SIGINT -> resume từ checkpoint mới nhất -> Done!", res == "done" and "--start-batch 5" in seg
          and "WATCHDOG: ram" in seg, seg)
    check("dọn checkpoint trong lúc chạy (eps=4: bỏ checkpoint-3, giữ 5 là mới nhất)", IR.ckpt_nums(e) == [5], IR.ckpt_nums(e))
    res, e = runner("fail", "fail", mem=10 ** 6)
    check("lỗi 2 lần liên tiếp -> dừng", res == "failed"
          and (e / "segments.log").read_text(encoding="utf-8").count("đoạn mới") == 2)
    res, e = runner("disk", "ok", mem=10 ** 6, disk=2)
    check("ổ < 3 GB -> dừng hẳn", res == "disk")


def test_dry_run():
    print("model.run --dry-run")
    p = subprocess.run([sys.executable, "-m", "gipbench.model.run", str(G / "configs/model/experiments/e3_v1b.yaml"),
                        "--stages", "0-6", "--dry-run"], capture_output=True, text=True, encoding="utf-8", cwd=G)
    out = p.stdout
    check("dry-run chạy được", p.returncode == 0, p.stderr[-500:])
    for k in ["--holdout-domain finance", "--max-domain-share 0.5", "--cap-exempt other", "cs_real=", ":0.35",
              "--epoch-hours 160 --epochs 6", "--speed-perturb --phone-sim 0.3", "--freeze-modules encoder_embed.",
              "--base-lr 0.002 --max-duration 150", "--save-every-n 1033", "--alphas 0.3 0.5 0.7 1.0", "wise_1/onnx"]:
        check(f"dry-run có `{k}`", k in out)
    check("model pipeline không gọi bench", "gipbench.bench" not in out)


def test_release(T):
    print("release (ghi / đọc / bất biến / materialize)")
    from gipbench.core import paths
    from gipbench.core import release as R
    from gipbench.core.manifest import read_jsonl
    os.environ["GIPBENCH_DATA"] = str(T / "data")
    paths._cache = None
    try:
        (T / "data/raw/x/audio").mkdir(parents=True)
        jl(T / "data/raw/x/m.jsonl", [{"id": "u1", "audio": "audio/u1.wav", "duration": 1.5, "text_tagged": "a <en>b</en>",
                                        "speaker": "s", "source": "y", "split": "train"}])
        rows = read_jsonl(str(T / "data/raw/x/m.jsonl"))
        rl = R.write_release("pool/x@v1", manifest_rows=rows, source=T / "data/raw/x/m.jsonl", notes="thử")
        got = json.loads(rl.manifest.read_text(encoding="utf-8").splitlines()[0])
        check("audio trong release là đường dẫn tương đối", got["audio"] == "../../../raw/x/audio/u1.wav", got["audio"])
        check("resolve đọc lại + thống kê", R.resolve("pool/x@v1").meta["files"]["manifest.jsonl"]["rows"] == 1)
        out = R.materialize("pool/x@v1", T / "mat.jsonl")
        m = json.loads(out.read_text(encoding="utf-8"))
        check("materialize: audio tuyệt đối trỏ đúng file gốc",
              os.path.normpath(m["audio"]) == os.path.normpath(str(T / "data/raw/x/audio/u1.wav")), m["audio"])
        try:
            R.write_release("pool/x@v1", manifest_rows=rows)
            check("ghi đè release -> lỗi", False)
        except R.ReleaseError:
            check("ghi đè release -> lỗi", True)
        with open(rl.manifest, "a", encoding="utf-8") as f:
            f.write("{}\n")
        try:
            R.resolve("pool/x@v1")
            check("sửa tay release -> resolve báo lỗi sha256", False)
        except R.ReleaseError as e:
            check("sửa tay release -> resolve báo lỗi sha256", "sha256" in str(e))
        for bad in ("pool/x", "abc/x@1", "bench"):
            try:
                R.parse(bad)
                check(f"ref sai {bad!r} -> lỗi", False)
            except R.ReleaseError:
                check(f"ref sai {bad!r} -> lỗi", True)
    finally:
        del os.environ["GIPBENCH_DATA"]
        paths._cache = None


def test_bench_systems():
    print("bench: system YAML")
    from gipbench.bench.run import decode_args, load_system
    ok = True
    for p in sorted((G / "configs/bench/systems").glob("*.yaml")):
        ok &= load_system(p)["system"] == p.stem
    check("mọi system YAML hợp lệ, tên = tên file", ok)
    a = decode_args({"beam": 4, "hotwords_exclude_b3": True, "hotwords_catalog": "x.tsv", "splits": ["B1", "B2"]})
    check("decode_args", a == ["--beam", "4", "--hotwords-exclude-b3", "--hotwords-catalog", "x.tsv", "--splits", "B1", "B2"], a)


def main():
    with tempfile.TemporaryDirectory() as t:
        T = Path(t)
        cases = [test_experiments, lambda: test_split_groups(T / "sg"), lambda: test_icefall_runner(T / "ir"),
                 test_dry_run, lambda: test_release(T / "rel"), test_bench_systems]
        for d in ("sg", "ir", "rel"):
            (T / d).mkdir()
        for i, fn in enumerate(cases):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                FAILS.append(f"ngoại lệ ở ca {i}")
    print(f"\n{'KHÔNG ĐẠT: ' + str(len(FAILS)) + ' ca sai' if FAILS else 'Tất cả đạt'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
