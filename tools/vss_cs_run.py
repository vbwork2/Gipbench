#!/usr/bin/env python3
"""Chạy tiếp ASR cho vss_cs theo chunk, chế độ CHỈ LỌC, dừng sớm khi đủ số giờ sạch.

  python tools/vss_cs_run.py --pre data/raw/vss_full/vss_cs_pre_rebuild.jsonl \
      --catalog data/raw/_proxy/catalog_proxy.tsv --seed 1 --max-new 6000 --target-clean-h 6

1. Mẫu: các câu của --pre (đã qua lọc rẻ + rò rỉ) CHƯA có hyp tầng 1 trong cache (tra theo id + hash audio),
   xáo trộn với --seed, lấy tối đa --max-new câu. Mẫu được lưu ở <state-dir>/vss_cs_run_s<seed>.jsonl ngay lần đầu,
   chạy lại thì dùng đúng mẫu đó (resume).
2. Mỗi chunk (--chunk câu):
   a. kiểm tra dung lượng --disk-path (mặc định ổ chứa --tmp-dir), dừng (exit 3) nếu < --min-free-gb (3 GB);
   b. tính số giờ sạch hiện có (mọi câu trong --pre có hyp, gồm cả pilot) bằng tools/agreement_filter.py --tiered
      (đồng thuận span + WER tiếng Việt <= --max-vi-wer); đủ --target-clean-h thì dừng;
   c. Qwen3-ASR-0.6B cho chunk, rồi faster-whisper large-v3 int8 CHỈ cho câu còn span Qwen chưa xác nhận
      (tools/asr_hyps.py, chạy trong venv ASR --asr-py; audio tải từ HF vào --tmp-dir, xoá sau mỗi chunk con).
3. Cuối cùng ghi <out> (vss_cs sạch, chỉ lọc) + <out>.report.md và <state-dir>/vss_cs_run_s<seed>.progress.jsonl.
Không gán lại span (--relabel-spans tắt). Chạy lại đúng lệnh này để chạy tiếp sau khi bị ngắt.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

G = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(G))
from lib.agreement import agree  # noqa: E402
from lib.hyp_cache import HashIndex, HypLookup  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402

REPO = "thanhnew2001/VietSuperSpeech"


def mem_available_gb():
    """MemAvailable của máy đang chạy (trong WSL: RAM trống của VM WSL), GB; None nếu không đọc được."""
    try:
        for l in open("/proc/meminfo", encoding="utf-8"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return None


def runs_gpu_h(paths):
    s = 0.0
    for p in paths:
        rp = Path(str(p) + ".runs.jsonl")
        if rp.exists():
            for l in open(rp, encoding="utf-8"):
                if l.strip():
                    r = json.loads(l)
                    s += r.get("asr_s", 0) + r.get("load_s", 0)
    return s / 3600


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pre", required=True, help="manifest sau lọc rẻ + rò rỉ (tools/vss_cs_prefilter.py)")
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-new", type=int, default=6000)
    ap.add_argument("--target-clean-h", type=float, default=6.0)
    ap.add_argument("--chunk", type=int, default=400)
    ap.add_argument("--max-vi-wer", type=float, default=0.15)
    ap.add_argument("--hyps-dirs", nargs=2, default=["results/asr_hyps/qwen3_asr_0.6b", "results/asr_hyps/fw_large_v3_int8"])
    ap.add_argument("--hash-index", default="results/asr_hyps/vss_audio_sha256.tsv")
    ap.add_argument("--audio-root", default="data/dl/VietSuperSpeech")
    ap.add_argument("--tmp-dir", default="data/tmp/vss_audio")
    ap.add_argument("--disk-path", help="ổ cần kiểm tra (mặc định ổ chứa --tmp-dir)")
    ap.add_argument("--min-free-gb", type=float, default=3.0)
    ap.add_argument("--asr-py", default=str(Path.home() / ".venvs/gipbench-asr/bin/python"))
    ap.add_argument("--qwen-batch", type=int, default=8, help="lô giải mã Qwen3-ASR (câu xếp theo độ dài trong chunk)")
    ap.add_argument("--min-free-ram-gb", type=float, default=1.0,
                    help="RAM trống của máy (MemAvailable) dưới ngưỡng này thì giảm lô Qwen xuống --qwen-batch-low")
    ap.add_argument("--qwen-batch-low", type=int, default=4)
    ap.add_argument("--state-dir", default="data/raw/vss_full")
    ap.add_argument("--out", default="data/raw/vss_full/vss_cs_clean.jsonl")
    args = ap.parse_args()
    os.chdir(G)
    st = Path(args.state_dir)
    st.mkdir(parents=True, exist_ok=True)
    Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)
    disk = args.disk_path or args.tmp_dir
    cat = Catalog.load(args.catalog)
    name = f"vss_cs_run_s{args.seed}"
    hq = Path(args.hyps_dirs[0]) / f"{name}.jsonl"
    hw = Path(args.hyps_dirs[1]) / f"{name}.jsonl"
    sample_p, prog_p = st / f"{name}.jsonl", st / f"{name}.progress.jsonl"

    pre = read_jsonl(args.pre)
    if sample_p.exists():
        sample = read_jsonl(str(sample_p))
        print(f"Mẫu cũ: {len(sample)} câu ({sample_p}), chạy tiếp")
    else:
        look = HypLookup([args.hyps_dirs[0]], HashIndex(args.hash_index), args.audio_root)
        todo = [r for r in pre if look.get(r)[0] is None]
        random.Random(args.seed).shuffle(todo)
        sample = todo[:args.max_new]
        write_jsonl(str(sample_p), sample)
        print(f"Mẫu mới (seed {args.seed}): {len(sample)} / {len(todo)} câu chưa chạy -> {sample_p}")

    nv = subprocess.run([args.asr_py, "-c", "import site; print(site.getsitepackages()[0])"],
                        capture_output=True, text=True, check=True).stdout.strip() + "/nvidia"
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "TMPDIR": str(Path(args.tmp_dir).resolve() / "py"),
           "LD_LIBRARY_PATH": f"{nv}/cublas/lib:{nv}/cudnn/lib" + (":" + os.environ["LD_LIBRARY_PATH"]
                                                                   if os.environ.get("LD_LIBRARY_PATH") else "")}
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    fetch = ["--chunk-size", "100", "--hf-repo", REPO, "--audio-root", args.audio_root, "--tmp-dir", args.tmp_dir,
             "--min-free-gb", str(args.min_free_gb), "--disk-path", disk, "--hash-index", args.hash_index]

    def clean_now(out):
        cmd = [sys.executable, "tools/agreement_filter.py", "--manifest", args.pre, "--hyps", *args.hyps_dirs,
               "--catalog", args.catalog, "--hash-index", args.hash_index, "--audio-root", args.audio_root, "--tiered",
               "--max-vi-wer", str(args.max_vi_wer), "--out", out]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if p.returncode:
            sys.exit(p.stdout + p.stderr)
        rows = read_jsonl(out)
        return sum(float(r["duration"]) for r in rows) / 3600, len(rows), p.stdout.strip().splitlines()[-1]

    stop, t_start, qb, batch_notes = "hết mẫu", time.time(), args.qwen_batch, []
    for k in range(0, len(sample), args.chunk):
        free = shutil.disk_usage(disk).free / 1e9
        if free < args.min_free_gb:
            stop = f"dung lượng {disk} còn {free:.1f} GB < {args.min_free_gb} GB"
            break
        ram = mem_available_gb()
        if ram is not None and ram < args.min_free_ram_gb and qb > args.qwen_batch_low:
            batch_notes.append(f"chunk {k // args.chunk}: RAM trống {ram:.2f} GB < {args.min_free_ram_gb} GB -> "
                               f"lô Qwen {qb} -> {args.qwen_batch_low}")
            print("  " + batch_notes[-1])
            qb = args.qwen_batch_low
        h, n, line = clean_now(str(st / f"{name}_clean_tmp.jsonl"))
        rec = {"chunk": k // args.chunk, "start": k, "clean_h": round(h, 3), "clean_n": n, "free_gb": round(free, 1),
               "ram_avail_gb": None if ram is None else round(ram, 2), "qwen_batch": qb,
               "gpu_h_run": round(runs_gpu_h([hq, hw]), 3), "t": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with open(prog_p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[chunk {k // args.chunk}] sạch {h:.2f} h / {n} câu (mục tiêu {args.target_clean_h} h), "
              f"GPU {rec['gpu_h_run']:.2f} h, D: {free:.1f} GB, RAM trống {rec['ram_avail_gb']} GB, lô Qwen {qb}")
        if h >= args.target_clean_h:
            stop = f"đạt {h:.2f} h sạch >= {args.target_clean_h} h"
            break
        part = sample[k:k + args.chunk]
        cm = st / f"{name}_chunk.jsonl"
        write_jsonl(str(cm), part)
        p = subprocess.run([args.asr_py, "tools/asr_hyps.py", "--engine", "qwen3-asr", "--model", "Qwen/Qwen3-ASR-0.6B",
                            "--device", "cuda:0", "--batch-size", str(qb), "--manifest", str(cm), "--out", str(hq), *fetch],
                           env=env)
        if p.returncode == 3:
            stop = "asr_hyps (Qwen) dừng vì thiếu dung lượng"
            break
        if p.returncode:
            sys.exit(f"asr_hyps (Qwen) lỗi, exit {p.returncode}")
        look = HypLookup([str(hq)], HashIndex(args.hash_index), args.audio_root)
        need = []
        for r in part:
            q = look.get(r)[0]
            if q is None:
                continue  # tải audio lỗi: để lần sau
            res = agree(r["text_tagged"], [q], cat, 1, 1.0)
            if any(v < 1 for v in res["confirmed"]):
                need.append(r["id"])
        ids = st / f"{name}_need_fw.txt"
        ids.write_text("".join(i + "\n" for i in need), encoding="utf-8")
        if need:
            p = subprocess.run([args.asr_py, "tools/asr_hyps.py", "--engine", "faster-whisper", "--model", "large-v3",
                                "--compute-type", "int8", "--device", "cuda", "--manifest", str(cm), "--ids", str(ids),
                                "--out", str(hw), *fetch], env=env)
            if p.returncode == 3:
                stop = "asr_hyps (faster-whisper) dừng vì thiếu dung lượng"
                break
            if p.returncode:
                sys.exit(f"asr_hyps (faster-whisper) lỗi, exit {p.returncode}")
    h, n, line = clean_now(args.out)
    shutil.rmtree(env["TMPDIR"], ignore_errors=True)
    gpu = runs_gpu_h([hq, hw])
    md = [f"# vss_cs: chạy tiếp ASR (seed {args.seed}, chỉ lọc)", "",
          f"- Dừng: {stop}.", f"- Mẫu: {len(sample)} câu ({sample_p}).",
          f"- Tổng sạch (tính cả pilot): **{h:.2f} h, {n} câu** -> `{args.out}`.",
          f"- GPU-h của lần chạy này (ASR + nạp model): {gpu:.2f} h; thời gian thực {(time.time() - t_start) / 3600:.2f} h.",
          f"- agreement_filter: {line}",
          f"- Lô Qwen: {args.qwen_batch} (câu xếp theo độ dài trong chunk)"
          + ("; " + "; ".join(batch_notes) if batch_notes else "; không phải giảm lô (RAM trống luôn >= "
             f"{args.min_free_ram_gb} GB)") + ".",
          "", "Tiến độ theo chunk: `" + str(prog_p) + "`."]
    Path(args.out + ".run.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if stop.startswith("dung lượng") or "thiếu dung lượng" in stop:
        sys.exit(3)


if __name__ == "__main__":
    main()
