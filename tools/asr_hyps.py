#!/usr/bin/env python3
"""Chạy một model ASR đa ngôn ngữ trên manifest và ghi hyps.jsonl (SCHEMA.md mục 3) cho bộ lọc đồng thuận.

  python tools/asr_hyps.py --engine faster-whisper --model large-v3 --compute-type int8 \
      --manifest data/raw/vss/vss_cs_candidates.jsonl --out results/fw_large_v3/vss_cs/hyps.jsonl --limit 20
  python tools/asr_hyps.py --engine qwen3-asr --model Qwen/Qwen3-ASR-0.6B \
      --manifest data/raw/vss/vss_cs_candidates.jsonl --out results/qwen3_asr_0.6b/vss_cs/hyps.jsonl --limit 20

Chỉ GHI hyps; việc lọc làm ở tools/agreement_filter.py (và ft/curate_train.py --agreement-hyps), nên chạy model một
lần rồi thử nhiều ngưỡng được. Câu đã có trong --out thì bỏ qua (chạy tiếp được khi bị ngắt).
Chạy lớn: --chunk-size N --hf-repo REPO --audio-root DIR --tmp-dir DIR tải audio thiếu của từng chunk vào --tmp-dir
rồi xoá sau chunk; trước mỗi chunk kiểm tra dung lượng --disk-path, dừng (exit 3) nếu < --min-free-gb (3 GB).
--ids FILE: chỉ giải mã các câu trong FILE (tầng thứ hai, vd faster-whisper cho câu Qwen chưa xác nhận).
Thời gian (nạp model, ASR, tải) ghi vào <out>.runs.jsonl để tính GPU-h.
Cần cài thêm: pip install -r requirements-asr.txt (faster-whisper, qwen-asr).
Ngôn ngữ: faster-whisper mặc định ép "vi" (từ Anh vẫn ra chính tả Anh); Qwen3-ASR mặc định tự nhận
(ép "Vietnamese" bằng --language Vietnamese). Cả hai là lựa chọn chưa kiểm chứng: so trên vài câu rồi chốt.
"""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.manifest import read_jsonl  # noqa: E402


def load_engine(args):
    if args.engine == "faster-whisper":
        from faster_whisper import WhisperModel
        m = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        lang = args.language or "vi"

        def run(path):
            segs, _ = m.transcribe(path, language=lang, beam_size=args.beam_size, vad_filter=False,
                                   condition_on_previous_text=False)
            return " ".join(s.text.strip() for s in segs).strip()
        return run
    import torch
    from qwen_asr import Qwen3ASRModel
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    m = Qwen3ASRModel.from_pretrained(args.model, dtype=dtype, device_map=args.device if args.device != "auto" else "cuda:0",
                                      max_inference_batch_size=1, max_new_tokens=args.max_new_tokens)

    def run(path):
        return m.transcribe(audio=path, language=args.language)[0].text.strip()
    return run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", choices=["faster-whisper", "qwen3-asr"], required=True)
    ap.add_argument("--model", required=True, help="large-v3 | Qwen/Qwen3-ASR-1.7B | Qwen/Qwen3-ASR-0.6B | đường dẫn")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="auto", help="faster-whisper: auto|cuda|cpu; qwen3-asr: cuda:0|cpu")
    ap.add_argument("--compute-type", default="int8", help="faster-whisper: int8 | int8_float16 | float16")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"], help="qwen3-asr")
    ap.add_argument("--language", default=None, help="faster-whisper: mã (vi); qwen3-asr: tên (Vietnamese); mặc định xem trên")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--ids", help="file id (mỗi dòng một id): chỉ giải mã các câu này (tầng ASR thứ hai)")
    ap.add_argument("--chunk-size", type=int, default=0, help="số câu mỗi chunk (0 = một chunk)")
    ap.add_argument("--hf-repo", help="tải audio thiếu từ repo dataset HF theo từng chunk, vd thanhnew2001/VietSuperSpeech")
    ap.add_argument("--audio-root", help="thư mục gốc audio cục bộ, ứng với gốc repo HF (bắt buộc khi có --hf-repo)")
    ap.add_argument("--tmp-dir", help="nơi để audio tạm của chunk (xoá sau mỗi chunk)")
    ap.add_argument("--workers", type=int, default=8, help="số luồng tải audio")
    ap.add_argument("--disk-path", help="ổ cần kiểm tra dung lượng trước mỗi chunk (mặc định --tmp-dir)")
    ap.add_argument("--min-free-gb", type=float, default=3.0, help="dừng (exit 3) nếu dung lượng trống thấp hơn")
    args = ap.parse_args()
    if args.hf_repo and not (args.audio_root and args.tmp_dir):
        sys.exit("--hf-repo cần --audio-root và --tmp-dir")

    rows = read_jsonl(args.manifest)
    if args.ids:
        want = {l.strip() for l in open(args.ids, encoding="utf-8") if l.strip()}
        rows = [r for r in rows if r["id"] in want]
    if args.limit:
        rows = rows[:args.limit]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out, encoding="utf-8") if l.strip()}
    todo = [r for r in rows if r["id"] not in done]
    print(f"{args.engine} {args.model}: {len(todo)} câu cần giải mã ({len(done)} đã có)")
    if not todo:
        return
    disk = args.disk_path or args.tmp_dir or str(out.parent)
    tmp = Path(args.tmp_dir) if args.tmp_dir else None
    log = {"engine": args.engine, "model": args.model, "n": 0, "audio_s": 0.0, "load_s": 0.0, "asr_s": 0.0,
           "download_s": 0.0, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "stop": "xong"}
    run = None
    chunk = args.chunk_size or len(todo)
    try:
        with open(out, "a", encoding="utf-8") as f:
            for c0 in range(0, len(todo), chunk):
                part = todo[c0:c0 + chunk]
                free = shutil.disk_usage(disk).free / 1e9
                if free < args.min_free_gb:
                    log["stop"] = f"dung lượng {disk} còn {free:.1f} GB < {args.min_free_gb} GB"
                    print(f"  DỪNG trước chunk {c0 // chunk + 1}: {log['stop']}")
                    break
                cdir = None
                if args.hf_repo:
                    cdir = tmp / f"chunk_{os.getpid()}_{c0}"
                    t1 = time.time()
                    paths, bad = fetch_chunk(part, args, cdir)
                    log["download_s"] += time.time() - t1
                    if bad:
                        print(f"  {len(bad)} câu tải audio lỗi, để lần chạy sau: {bad[:3]}")
                else:
                    paths = {r["id"]: r["_audio_path"] for r in part}
                if run is None:
                    t1 = time.time()
                    run = load_engine(args)
                    log["load_s"] = time.time() - t1
                    print(f"  nạp model {log['load_s']:.1f} s")
                t1 = time.time()
                for r in part:
                    if r["id"] not in paths:
                        continue
                    hyp = run(paths[r["id"]])
                    log["audio_s"] += float(r["duration"])
                    log["n"] += 1
                    f.write(json.dumps({"id": r["id"], "hyp": hyp, "engine": args.engine, "model": args.model},
                                       ensure_ascii=False) + "\n")
                    f.flush()
                log["asr_s"] += time.time() - t1
                if cdir is not None:
                    shutil.rmtree(cdir, ignore_errors=True)
                print(f"  {min(c0 + chunk, len(todo))}/{len(todo)} câu, RTF {log['asr_s'] / max(log['audio_s'], 1e-6):.3f}, "
                      f"{shutil.disk_usage(disk).free / 1e9:.1f} GB trống ở {disk}")
    finally:
        log["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(str(out) + ".runs.jsonl", "a", encoding="utf-8") as lf:
            lf.write(json.dumps(log, ensure_ascii=False) + "\n")
    print(f"Đã ghi {out} ({log['n']} câu mới, {log['audio_s'] / 3600:.2f} h audio, ASR {log['asr_s'] / 3600:.2f} h, "
          f"tải {log['download_s'] / 60:.1f} phút)")
    if log["stop"] != "xong":
        sys.exit(3)


def fetch_chunk(part, args, cdir):
    """Tải audio của một chunk từ repo HF vào cdir (file đã có ở --audio-root thì dùng luôn).
    Đường dẫn trong repo = đường dẫn audio tương đối với --audio-root."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    root = Path(args.audio_root).resolve()
    paths, bad, need = {}, [], []
    for r in part:
        p = Path(r["_audio_path"])
        if p.exists():
            paths[r["id"]] = str(p)
        else:
            need.append((r["id"], p.resolve().relative_to(root).as_posix()))
    cdir.mkdir(parents=True, exist_ok=True)

    def get(item):
        uid, rel = item
        err = None
        for k in range(3):
            try:
                return uid, hf_hub_download(args.hf_repo, rel, repo_type="dataset", local_dir=str(cdir))
            except Exception as e:  # noqa: BLE001 — mạng chập chờn: thử lại, sau 3 lần thì để lần sau
                err = e
                time.sleep(2 * (k + 1))
        return uid, f"LỖI {type(err).__name__}: {err}"
    with ThreadPoolExecutor(args.workers) as ex:
        for uid, res in ex.map(get, need):
            if res.startswith("LỖI"):
                bad.append((uid, res[:120]))
            else:
                paths[uid] = res
    return paths, bad


if __name__ == "__main__":
    main()
