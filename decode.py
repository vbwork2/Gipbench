#!/usr/bin/env python3
"""Giải mã bench bằng model Zipformer-transducer ONNX (Gipformer hoặc bản FT export lại) qua sherpa-onnx.

  # E0 baseline
  python decode.py --bench data/bench.jsonl --model-dir models/gip15 --system e0_baseline
  # E1 hotwords từ catalog
  python decode.py --bench data/bench.jsonl --model-dir models/gip15 --system e1_hw2.0 \
      --hotwords-catalog configs/catalog.tsv --hotwords-score 2.0
  # model FT đã export
  python decode.py --bench data/bench.jsonl --model-dir exp/e3/onnx --system e3_ft

--model-dir cần: encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt (và bpe.model nếu dùng hotwords).
Không có model trên máy: tải bằng  huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt --local-dir models/gip15
Ghi ra results/<system>/hyps.jsonl và decode_summary.json (RTF, cấu hình).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.manifest import filter_splits, read_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402


def pick(d: Path, stem: str, quant: str) -> Path:
    pats = [f"{stem}.int8.onnx", f"{stem}-*.int8.onnx"] if quant == "int8" else [f"{stem}.onnx", f"{stem}-*[!8].onnx"]
    for p in pats:
        hits = sorted(x for x in d.glob(p) if (quant == "int8") == x.name.endswith(".int8.onnx"))
        if hits:
            return hits[0]
    sys.exit(f"Không tìm thấy {stem} ({quant}) trong {d}")


def write_hotwords(catalog: Catalog, out: Path, case: str, include_reserved: bool) -> int:
    terms = [" ".join(t) for t, v in catalog.terms.items() if include_reserved or not v.reserved_b3]
    with open(out, "w", encoding="utf-8") as f:
        for t in terms:
            f.write((t.upper() if case == "upper" else t) + "\n")
    return len(terms)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--system", required=True, help="tên hệ thống = tên thư mục kết quả")
    ap.add_argument("--results", default="results")
    ap.add_argument("--splits", nargs="*", help="mặc định: tất cả")
    ap.add_argument("--quantize", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--decoding-method", choices=["greedy_search", "modified_beam_search"], default="modified_beam_search")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--blank-penalty", type=float, default=0.0)
    ap.add_argument("--hotwords-file", help="file hotwords có sẵn (1 cụm/dòng, đúng kiểu chữ của model)")
    ap.add_argument("--hotwords-catalog", help="tạo hotwords từ catalog.tsv")
    ap.add_argument("--hotwords-exclude-b3", action="store_true",
                    help="bỏ thuật ngữ reserved_b3 khỏi danh sách (mặc định giữ: biasing được phép biết danh sách thuật ngữ lúc decode)")
    ap.add_argument("--hotwords-score", type=float, default=1.5)
    ap.add_argument("--case", choices=["upper", "lower"], default="upper", help="kiểu chữ của tokenizer (Gipformer: upper)")
    args = ap.parse_args()

    import numpy as np
    import sherpa_onnx
    import soundfile as sf

    md = Path(args.model_dir)
    out = Path(args.results) / args.system
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(encoder=str(pick(md, "encoder", args.quantize)), decoder=str(pick(md, "decoder", args.quantize)),
              joiner=str(pick(md, "joiner", args.quantize)), tokens=str(md / "tokens.txt"), num_threads=args.threads,
              decoding_method=args.decoding_method, max_active_paths=args.beam, blank_penalty=args.blank_penalty)
    hot = args.hotwords_file
    if args.hotwords_catalog:
        hot = str(out / "hotwords.txt")
        n = write_hotwords(Catalog.load(args.hotwords_catalog), Path(hot), args.case, not args.hotwords_exclude_b3)
        print(f"hotwords: {n} thuật ngữ -> {hot}")
    if hot:
        if args.decoding_method != "modified_beam_search":
            sys.exit("hotwords chỉ chạy với --decoding-method modified_beam_search")
        bpe_vocab = md / "bpe.vocab"
        if not bpe_vocab.exists():
            if not (md / "bpe.model").exists():
                sys.exit(f"Hotwords cần {md}/bpe.model (hoặc bpe.vocab). Copy bpe.model của Gipformer vào thư mục model.")
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=str(md / "bpe.model"))
            with open(bpe_vocab, "w", encoding="utf-8") as f:
                for i in range(sp.get_piece_size()):
                    f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
        kw.update(hotwords_file=hot, hotwords_score=args.hotwords_score, modeling_unit="bpe", bpe_vocab=str(bpe_vocab))
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(**kw)

    rows = filter_splits(read_jsonl(args.bench), args.splits)
    audio_sec = proc_sec = 0.0
    with open(out / "hyps.jsonl", "w", encoding="utf-8") as f:
        for i in range(0, len(rows), args.batch):
            chunk = rows[i:i + args.batch]
            streams = []
            for r in chunk:
                x, sr = sf.read(r["_audio_path"], dtype="float32", always_2d=True)
                x = np.ascontiguousarray(x.mean(axis=1))
                s = rec.create_stream()
                s.accept_waveform(sr, x)
                streams.append(s)
                audio_sec += len(x) / sr
            t0 = time.perf_counter()
            rec.decode_streams(streams)
            proc_sec += time.perf_counter() - t0
            for r, s in zip(chunk, streams):
                f.write(json.dumps({"id": r["id"], "hyp": s.result.text}, ensure_ascii=False) + "\n")
            print(f"\r{min(i + args.batch, len(rows))}/{len(rows)}", end="", flush=True)
    summ = {"system": args.system, "audio_sec": audio_sec, "proc_sec": proc_sec, "rtf": proc_sec / max(audio_sec, 1e-9),
            "threads": args.threads, "config": {k: v for k, v in vars(args).items()}, "model_files": {k: kw[k] for k in ("encoder", "decoder", "joiner")}}
    (out / "decode_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRTF {summ['rtf']:.3f} ({args.threads} luồng) -> {out}/hyps.jsonl")


if __name__ == "__main__":
    main()
