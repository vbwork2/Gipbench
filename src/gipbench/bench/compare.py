#!/usr/bin/env python3
"""Bảng so sánh nhiều hệ trên một benchmark, từ summary.json của gipbench.bench.evaluate (không chấm lại).

  python -m gipbench.bench.compare --benchmark general_v1_draft \
      --runs results/general_asr/general_v1_draft/gip15_mbs4_open results/general_asr/general_v1_draft/viegem_greedy \
      [--expect viegem_lm whisper_large_v3] [--out reports/bench/general_v1_draft_compare.md]

Mỗi hàng = (bộ dữ liệu, hệ). Cột: số câu, giờ, WER (space_token_v1) + CI95, CER, CS-WER, term recall, inference mode,
model + revision, RTF + thiết bị, cảnh báo trùng pretraining (từ configs/bench/datasets.yaml). Hệ trong --expect mà chưa
có kết quả: ghi "N/A (chưa chạy)". Chỉ so các hệ chấm trên CÙNG manifest_sha256 (khác thì dừng).
"""
import argparse
import json
import sys
from pathlib import Path

from gipbench.bench import registry as R

MODEL_KEY = {"sherpa_transducer": "gipformer15", "viegem": "viegem"}


def f(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="thư mục có summary.json (+ run_meta.json)")
    ap.add_argument("--expect", nargs="*", default=[], help="tên hệ dự kiến; chưa có thì ghi N/A (chưa chạy)")
    ap.add_argument("--profile", help="mặc định: hồ sơ chính của benchmark")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    bm = R.load_benchmark(args.benchmark)
    reg = R.load_datasets().get("datasets") or {}
    prof = args.profile or bm["profile"]
    runs, shas = [], set()
    for d in args.runs:
        s = json.loads((Path(d) / "summary.json").read_text(encoding="utf-8"))
        runs.append(s)
        shas.add(s["manifest_sha256"])
    if len(shas) > 1:
        sys.exit(f"các hệ được chấm trên manifest khác nhau: {sorted(x[:12] for x in shas)}")
    datasets = sorted({k.split(":", 1)[1] for s in runs for k in s["profiles"][prof] if k.startswith("dataset:")})
    L = [f"# So sánh trên {bm['benchmark_id']} {bm['version']} ({bm['evidence_level']}), hồ sơ `{prof}`", "",
         f"manifest_sha256 `{next(iter(shas))[:16]}`. WER = space_token_v1 (token theo khoảng trắng ≈ âm tiết), "
         "CI95 bootstrap theo cụm. RTF đo trên phần cứng khác nhau: không so tốc độ trực tiếp.", "",
         "| Bộ | Hệ | Mode | Câu | Giờ | WER | CI95 | CER | CS-WER | Term recall | Trùng pretraining | RTF (thiết bị) |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    names = [s["system"] for s in runs]
    for ds in datasets + ["overall"]:
        key = "overall" if ds == "overall" else f"dataset:{ds}"
        for s in runs:
            g = s["profiles"][prof].get(key)
            rm = s.get("run_meta") or {}
            mk = MODEL_KEY.get(rm.get("engine"))
            ov = ((reg.get(ds) or {}).get("pretraining_overlap") or {}).get(mk, "–") if ds != "overall" else ""
            rtf = f"{f(rm.get('rtf'), 4)} ({rm.get('device', '?')})" if rm else "–"
            if not g:
                L.append(f"| {ds} | {s['system']} | {s['inference_mode']} | N/A (chưa chạy) | | | | | | | {ov} | |")
                continue
            ci = g.get("wer_ci95")
            L.append(f"| {ds} | {s['system']} | {s['inference_mode']} | {g['utt']} | {g['hours']:.2f} | **{f(g['wer'])}** | "
                     f"{'[' + f(ci[0]) + ', ' + f(ci[1]) + ']' if ci else '–'} | {f(g['cer'])} | {f(g['cs_wer'])} | "
                     f"{f(g['term_recall'])} | {ov} | {rtf} |")
        for e in args.expect:
            if e not in names:
                L.append(f"| {ds} | {e} | – | N/A (chưa chạy) | | | | | | | | |")
    macro = {s["system"]: (s["profiles"][prof].get("macro_by_dataset") or {}) for s in runs}
    if any(macro.values()):
        L += ["", "Macro theo bộ (trung bình đều WER, không có CI; đọc kèm số từng bộ): " +
              ", ".join(f"{k} **{f(v.get('value'))}**" for k, v in macro.items() if v)]
    L += ["", "Thông tin hệ:", ""]
    for s in runs:
        rm = s.get("run_meta") or {}
        L.append(f"- `{s['system']}`: {rm.get('model_id', '?')} @ {str(rm.get('revision', '?'))[:12]}, giải mã "
                 f"{json.dumps(rm.get('decoding', {}), ensure_ascii=False)}, {rm.get('precision', '')}, "
                 f"{(rm.get('hardware') or {}).get('gpu') or 'CPU ' + str((rm.get('hardware') or {}).get('cpu_count')) + ' luồng'}, "
                 f"hyps_sha256 `{(s.get('hyps_sha256') or '')[:12]}`")
    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
