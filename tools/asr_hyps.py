#!/usr/bin/env python3
"""Chạy một model ASR đa ngôn ngữ trên manifest và ghi hyps.jsonl (SCHEMA.md mục 3) cho bộ lọc đồng thuận.

  python tools/asr_hyps.py --engine faster-whisper --model large-v3 --compute-type int8 \
      --manifest data/raw/vss/vss_cs_candidates.jsonl --out results/fw_large_v3/vss_cs/hyps.jsonl --limit 20
  python tools/asr_hyps.py --engine qwen3-asr --model Qwen/Qwen3-ASR-0.6B \
      --manifest data/raw/vss/vss_cs_candidates.jsonl --out results/qwen3_asr_0.6b/vss_cs/hyps.jsonl --limit 20

Chỉ GHI hyps; việc lọc làm ở tools/agreement_filter.py (và ft/curate_train.py --agreement-hyps), nên chạy model một
lần rồi thử nhiều ngưỡng được. Câu đã có trong --out thì bỏ qua (chạy tiếp được khi bị ngắt).
Cần cài thêm: pip install -r requirements-asr.txt (faster-whisper, qwen-asr).
Ngôn ngữ: faster-whisper mặc định ép "vi" (từ Anh vẫn ra chính tả Anh); Qwen3-ASR mặc định tự nhận
(ép "Vietnamese" bằng --language Vietnamese). Cả hai là lựa chọn chưa kiểm chứng: so trên vài câu rồi chốt.
"""
import argparse
import json
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
    args = ap.parse_args()

    rows = read_jsonl(args.manifest)
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
    t0 = time.time()
    run = load_engine(args)
    print(f"  nạp model {time.time() - t0:.1f} s")
    t0, audio_s = time.time(), 0.0
    with open(out, "a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            hyp = run(r["_audio_path"])
            audio_s += float(r["duration"])
            f.write(json.dumps({"id": r["id"], "hyp": hyp, "engine": args.engine, "model": args.model},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"  {i}/{len(todo)} câu, RTF {el / max(audio_s, 1e-6):.3f}")
    print(f"Đã ghi {out}")


if __name__ == "__main__":
    main()
