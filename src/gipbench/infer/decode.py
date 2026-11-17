#!/usr/bin/env python3
"""Giải mã bench bằng model Zipformer-transducer ONNX (Gipformer hoặc bản FT export lại) qua sherpa-onnx.

  # E0 baseline
  python -m gipbench.infer.decode --bench data/bench.jsonl --model-dir models/gip15 --system e0_baseline
  # E1 hotwords từ catalog
  python -m gipbench.infer.decode --bench data/bench.jsonl --model-dir models/gip15 --system e1_hw2.0 \
      --hotwords-catalog configs/catalog.tsv --hotwords-score 2.0
  # E1 hotwords + từ gây nhiễu (không có trong catalog; tạo bằng src/gipbench/model/systems/hotword_distractors.py), lấy mẫu 500 từ, có seed
  python -m gipbench.infer.decode --bench data/bench.jsonl --model-dir models/gip15 --system e1_hw1.5_dis500 \
      --hotwords-catalog configs/catalog.tsv --hotwords-distractors data/distractors.tsv \
      --hotwords-distractors-n 500 --hotwords-seed 0
  # model FT đã export
  python -m gipbench.infer.decode --bench data/bench.jsonl --model-dir exp/e3/onnx --system e3_ft

--model-dir cần: encoder*.onnx, decoder*.onnx, joiner*.onnx, tokens.txt (và bpe.model nếu dùng hotwords).
Không có model trên máy: tải bằng  huggingface-cli download g-group-ai-lab/gipformer1.5-68M-rnnt --local-dir models/gip15
Ghi ra results/<system>/hyps.jsonl và decode_summary.json (RTF, cấu hình).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

from gipbench.core.manifest import filter_splits, read_jsonl
from gipbench.core.textnorm import Catalog


def pick(d: Path, stem: str, quant: str) -> Path:
    pats = [f"{stem}.int8.onnx", f"{stem}-*.int8.onnx"] if quant == "int8" else [f"{stem}.onnx", f"{stem}-*[!8].onnx"]
    for p in pats:
        hits = sorted(x for x in d.glob(p) if (quant == "int8") == x.name.endswith(".int8.onnx"))
        if hits:
            return hits[0]
    sys.exit(f"Không tìm thấy {stem} ({quant}) trong {d}")


def catalog_terms(catalog: Catalog, include_reserved: bool) -> list:
    return [" ".join(t) for t, v in catalog.terms.items() if include_reserved or not v.reserved_b3]


def write_hotwords(catalog: Catalog, out: Path, case: str, include_reserved: bool) -> int:
    terms = catalog_terms(catalog, include_reserved)
    with open(out, "w", encoding="utf-8") as f:
        for t in terms:
            f.write((t.upper() if case == "upper" else t) + "\n")
    return len(terms)


def read_distractors(path) -> list:
    """Đọc file từ gây nhiễu (src/gipbench/model/systems/hotword_distractors.py): cột đầu của mỗi dòng, bỏ dòng trống và dòng '#'."""
    out = []
    for line in open(path, encoding="utf-8"):
        t = line.rstrip("\n").split("\t")[0].strip()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def mix_hotwords(real: list, distractors: list, n=None, seed: int = 0):
    """Trộn thuật ngữ thật với từ gây nhiễu (cùng kiểu chữ, cùng điểm vì chỉ là thêm dòng vào file hotwords).

    Từ gây nhiễu trùng nhau hoặc trùng thuật ngữ thật (không phân biệt hoa / thường) bị bỏ; lấy mẫu n từ có seed
    (n = None: tất cả). Danh sách đã sắp xếp trước khi lấy mẫu nên kết quả không phụ thuộc thứ tự dòng trong file.
    Trả về (danh sách = thật + gây nhiễu, {"n_real", "n_distractors", "n_distractor_pool"}).
    """
    if n is not None and n < 0:
        raise ValueError("--hotwords-distractors-n phải >= 0")
    taken = {t.casefold() for t in real}
    pool = {}
    for d in distractors:
        if d.casefold() not in taken:
            pool.setdefault(d.casefold(), d)
    pool = sorted(pool.values(), key=str.casefold)
    pick = pool if n is None or n >= len(pool) else sorted(random.Random(seed).sample(pool, n), key=str.casefold)
    return list(real) + pick, {"n_real": len(real), "n_distractors": len(pick), "n_distractor_pool": len(pool)}


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
    ap.add_argument("--hotwords-distractors", help="file từ gây nhiễu (src/gipbench/model/systems/hotword_distractors.py) trộn vào hotwords, cùng điểm "
                                                   "với thuật ngữ thật; cần --hotwords-catalog hoặc --hotwords-file")
    ap.add_argument("--hotwords-distractors-n", type=int, default=None, help="số từ gây nhiễu lấy mẫu (mặc định: tất cả)")
    ap.add_argument("--hotwords-seed", type=int, default=0, help="seed lấy mẫu từ gây nhiễu")
    ap.add_argument("--case", choices=["upper", "lower"], default="upper", help="kiểu chữ của tokenizer (Gipformer: upper)")
    ap.add_argument("--save-logprobs", action="store_true",
                    help="ghi thêm n_tokens, logprob_mean, logprob_min (từ ys_log_probs của sherpa-onnx) vào hyps.jsonl")
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
    hw_info = None
    if args.hotwords_distractors_n is not None and not args.hotwords_distractors:
        sys.exit("--hotwords-distractors-n cần --hotwords-distractors")
    if args.hotwords_distractors and not (args.hotwords_catalog or args.hotwords_file):
        sys.exit("--hotwords-distractors cần --hotwords-catalog hoặc --hotwords-file")
    if args.hotwords_catalog and not args.hotwords_distractors:
        hot = str(out / "hotwords.txt")
        n = write_hotwords(Catalog.load(args.hotwords_catalog), Path(hot), args.case, not args.hotwords_exclude_b3)
        print(f"hotwords: {n} thuật ngữ -> {hot}")
        hw_info = {"n_real": n, "n_distractors": 0}
    elif args.hotwords_distractors:
        if args.hotwords_catalog:  # thuật ngữ thật: đổi kiểu chữ như write_hotwords
            real = [t.upper() if args.case == "upper" else t
                    for t in catalog_terms(Catalog.load(args.hotwords_catalog), not args.hotwords_exclude_b3)]
        else:  # file có sẵn: giữ nguyên từng dòng (đúng kiểu chữ của model)
            real = [l.strip() for l in open(args.hotwords_file, encoding="utf-8") if l.strip()]
        dis = [t.upper() if args.case == "upper" else t for t in read_distractors(args.hotwords_distractors)]
        terms, hw_info = mix_hotwords(real, dis, args.hotwords_distractors_n, args.hotwords_seed)
        hot = str(out / "hotwords.txt")
        Path(hot).write_text("".join(t + "\n" for t in terms), encoding="utf-8")
        print(f"hotwords: {hw_info['n_real']} thuật ngữ thật + {hw_info['n_distractors']} từ gây nhiễu "
              f"(pool {hw_info['n_distractor_pool']}) -> {hot}")
    elif hot:
        hw_info = {"n_real": sum(1 for l in open(hot, encoding="utf-8") if l.strip()), "n_distractors": 0}
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
                rec_out = {"id": r["id"], "hyp": s.result.text}
                if args.save_logprobs:  # ys_log_probs của sherpa-onnx (rỗng nếu model / cách giải mã không trả)
                    lp = list(getattr(s.result, "ys_log_probs", []) or [])
                    rec_out.update(n_tokens=len(lp), logprob_mean=(sum(lp) / len(lp)) if lp else None,
                                   logprob_min=min(lp) if lp else None)
                f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            print(f"\r{min(i + args.batch, len(rows))}/{len(rows)}", end="", flush=True)
    summ = {"system": args.system, "audio_sec": audio_sec, "proc_sec": proc_sec, "rtf": proc_sec / max(audio_sec, 1e-9),
            "threads": args.threads, "config": {k: v for k, v in vars(args).items()}, "model_files": {k: kw[k] for k in ("encoder", "decoder", "joiner")}}
    if hw_info:  # số hotwords thật / số từ gây nhiễu (n_distractors = 0 khi không dùng --hotwords-distractors)
        summ["hotwords"] = hw_info
    (out / "decode_summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRTF {summ['rtf']:.3f} ({args.threads} luồng) -> {out}/hyps.jsonl")


if __name__ == "__main__":
    main()
