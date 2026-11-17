#!/usr/bin/env python3
"""Replay tự chưng cất (Spec mục 10): nhãn = hyp của model GỐC, chỉ giữ câu greedy và modified beam search giống hệt.

  python -m gipbench.dataset.pseudolabel.distill --manifest data/raw/vivos/train_vivos.jsonl --model-dir models/gip15 \
      --out data/distill/distill_vivos.jsonl

Các bước:
  1. gọi decode.py hai lần trên --manifest: --decoding-method greedy_search và modified_beam_search (beam --beam),
     kết quả ở <results>/<name>_greedy, <name>_mbs (hoặc đưa sẵn --greedy-hyps / --beam-hyps, không cần model);
  2. giữ câu có hai hyp GIỐNG HỆT nhau (sau strip) và không rỗng — proxy cho độ tin cậy;
  3. lọc thời lượng [--min-dur, --max-dur] và số từ / giây [--min-wps, --max-wps] (từ = token sau basic_tokens);
  4. nếu hyps có logprob_mean (decode.py --save-logprobs, lấy từ ys_log_probs của sherpa-onnx) và đặt
     --min-logprob-mean: bỏ câu có logprob trung bình thấp hơn ngưỡng (lấy theo bản greedy).
Ghi <out>: text = text_tagged = hyp viết thường, distill: true, dataset distill:<dataset gốc>, label_source
distill:<model>, split train; và <out>.report.md (tỉ lệ giữ, lý do loại).
"""
import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from gipbench.core.paths import REPO as G
from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.textnorm import basic_tokens


def load_hyps(p):
    return {h["id"]: h for h in (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())}


def decode(args, method, name):
    cmd = [sys.executable, "-m", "gipbench.infer.decode", "--bench", args.manifest, "--model-dir", args.model_dir,
           "--system", name, "--results", args.results, "--decoding-method", method, "--beam", str(args.beam),
           "--threads", str(args.threads), "--quantize", args.quantize, "--save-logprobs"]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    return str(Path(args.results) / name / "hyps.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, help="audio tiếng Việt (nhãn có sẵn, nếu có, bị bỏ qua)")
    ap.add_argument("--model-dir", help="model GỐC (Gipformer) cho decode.py")
    ap.add_argument("--model-name", default="gipformer1.5-68M-rnnt")
    ap.add_argument("--greedy-hyps", help="hyps greedy có sẵn (bỏ qua decode.py)")
    ap.add_argument("--beam-hyps", help="hyps modified beam search có sẵn")
    ap.add_argument("--results", default="results/distill")
    ap.add_argument("--name", help="tên thư mục kết quả decode (mặc định: tên manifest)")
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--quantize", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=20.0)
    ap.add_argument("--min-wps", type=float, default=1.0, help="số từ / giây tối thiểu (thấp: hyp bị cụt)")
    ap.add_argument("--max-wps", type=float, default=6.0, help="số từ / giây tối đa (cao: hyp bị lặp / ảo)")
    ap.add_argument("--min-logprob-mean", type=float, default=None, help="ngưỡng confidence (cần logprob trong hyps)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = read_jsonl(args.manifest)
    name = args.name or Path(args.manifest).stem
    if args.greedy_hyps and args.beam_hyps:
        gp, bp = args.greedy_hyps, args.beam_hyps
    elif args.model_dir:
        gp = decode(args, "greedy_search", f"{name}_greedy")
        bp = decode(args, "modified_beam_search", f"{name}_mbs")
    else:
        sys.exit("cần --model-dir, hoặc cả --greedy-hyps lẫn --beam-hyps")
    gh, bh = load_hyps(gp), load_hyps(bp)
    has_lp = any(h.get("logprob_mean") is not None for h in gh.values())
    if args.min_logprob_mean is not None and not has_lp:
        print("  CẢNH BÁO: hyps không có logprob_mean (sherpa-onnx không trả ys_log_probs?) -> bỏ qua --min-logprob-mean")
    why, kept = Counter(), []
    for r in rows:
        g, b = gh.get(r["id"]), bh.get(r["id"])
        if g is None or b is None:
            why["thiếu hyp"] += 1
            continue
        gt, bt = (g.get("hyp") or "").strip(), (b.get("hyp") or "").strip()
        dur = float(r["duration"])
        nw = len(basic_tokens(gt))
        if not gt:
            why["hyp rỗng"] += 1
        elif gt != bt:
            why["greedy khác beam"] += 1
        elif not args.min_dur <= dur <= args.max_dur:
            why[f"độ dài ngoài {args.min_dur:g}–{args.max_dur:g} s"] += 1
        elif not args.min_wps <= nw / dur <= args.max_wps:
            why[f"từ/giây ngoài {args.min_wps:g}–{args.max_wps:g}"] += 1
        elif (args.min_logprob_mean is not None and has_lp and g.get("logprob_mean") is not None
              and g["logprob_mean"] < args.min_logprob_mean):
            why[f"logprob_mean < {args.min_logprob_mean:g}"] += 1
        else:
            o = {k: v for k, v in r.items() if not k.startswith("_")}
            o.update(split="train", text=gt.lower(), text_tagged=gt.lower(), distill=True,
                     dataset=f"distill:{str(r.get('dataset') or '?').split(':')[0]}",
                     label_source=f"distill:{args.model_name}",
                     distill_meta={"logprob_mean": g.get("logprob_mean"), "wps": round(nw / dur, 3)})
            o.pop("text_tagged_orig", None)
            kept.append(o)
            why["giữ"] += 1
    write_jsonl(args.out, kept)
    n = len(rows)
    h_in = sum(float(r["duration"]) for r in rows) / 3600
    h_out = sum(float(r["duration"]) for r in kept) / 3600
    md = [f"# Chưng cất replay — {args.manifest}", "", f"Hyps: greedy `{gp}`, modified beam search `{bp}`. "
          f"Log-prob trong hyps: {'có' if has_lp else 'KHÔNG'}.", "",
          f"Giữ **{len(kept)} / {n}** câu ({100 * len(kept) / max(n, 1):.1f}%), {h_out:.2f} / {h_in:.2f} h.", "",
          "| Kết quả | Câu |", "| --- | --- |"] + [f"| {k} | {v} |" for k, v in why.most_common()]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Giữ {len(kept)}/{n} câu ({h_out:.2f}/{h_in:.2f} h) -> {args.out}; " + ", ".join(f"{k}: {v}" for k, v in why.items()))


if __name__ == "__main__":
    main()
