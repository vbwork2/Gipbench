#!/usr/bin/env python3
"""Lọc manifest CS có nhãn máy theo đồng thuận với hyp của các model ASR khác (chỉ ĐỌC hyps, không chạy model).

  python tools/agreement_filter.py --manifest data/raw/vss_full/vss_cs_pilot.jsonl \
      --hyps results/asr_hyps/qwen3_asr_0.6b/vss_cs_pilot.jsonl results/asr_hyps/fw_large_v3_int8/vss_cs_pilot.jsonl \
      --catalog data/raw/_proxy/catalog_proxy.tsv --min-agree 1 --out data/raw/vss_full/vss_cs_pilot_agreed.jsonl

Dùng cho vss_cs (SEA: tùy chọn, không dùng). Hyps do tools/asr_hyps.py ghi (định dạng SCHEMA.md mục 3).
Một span <en> của nhãn được xác nhận khi nó xuất hiện (sau normalize_hyp, strict) trong >= --min-agree file hyps.
Giữ câu khi tỉ lệ span được xác nhận >= --min-span-frac (mặc định 1.0: mọi span). Câu thiếu hyp ở một file thì
coi file đó là hyp rỗng. Ghi: <out> (câu giữ lại, thêm trường agreement) và <out>.report.md.

--relabel-spans (mặc định TẮT): câu không qua lọc được gán lại TỪNG span chưa xác nhận khi MỌI file hyps (>= 2) cho
cùng một chữ ở vị trí span đó sau chuẩn hoá VÀ WER phần tiếng Việt của nhãn so với từng hyp <= --relabel-max-vi-wer
(0.10). Không gán lại cả câu. Câu gán lại: relabeled = true, text_orig = text_tagged gốc, text / text_tagged mới;
sau đó chạy lại kiểm tra rò rỉ (--bench, reserved_b3 trong --catalog) và loại câu rò rỉ.
--write-unconfirmed-ids: ghi id các câu còn span chưa được xác nhận (đầu vào cho tầng ASR kế tiếp).
--listen-csv: --listen-n câu gán lại ngẫu nhiên (--seed) để người nghe kiểm tra.
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.agreement import agree, plain, relabel_spans  # noqa: E402
from lib.leak import LeakGuard  # noqa: E402
from lib.manifest import read_jsonl, write_jsonl  # noqa: E402
from lib.textnorm import Catalog  # noqa: E402

HF_URL = "https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech/resolve/main/"


def audio_rel(r, audio_root):
    p = Path(r.get("_audio_path") or r["audio"])
    if audio_root:
        try:
            return p.resolve().relative_to(Path(audio_root).resolve()).as_posix()
        except ValueError:
            pass
    s = p.as_posix()
    return s[s.index("audio/"):] if "audio/" in s else s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyps", nargs="+", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--min-agree", type=int, default=1, help="số file hyps tối thiểu phải chứa một span")
    ap.add_argument("--min-span-frac", type=float, default=1.0, help="tỉ lệ span tối thiểu phải được xác nhận")
    ap.add_argument("--only-with-hyps", action="store_true", help="chỉ xét câu có hyp ở mọi file (chạy thử một phần)")
    ap.add_argument("--relabel-spans", action="store_true", help="gán lại span chưa xác nhận (mặc định tắt)")
    ap.add_argument("--relabel-max-vi-wer", type=float, default=0.10)
    ap.add_argument("--bench", nargs="*", default=[], help="manifest bench để kiểm tra rò rỉ lại sau khi gán lại")
    ap.add_argument("--bench-splits", nargs="+", default=["B1", "B2", "B3"])
    ap.add_argument("--write-unconfirmed-ids", help="ghi id câu còn span chưa được xác nhận")
    ap.add_argument("--listen-csv", help="phiếu nghe: --listen-n câu gán lại ngẫu nhiên")
    ap.add_argument("--listen-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audio-root", help="thư mục gốc audio (đường dẫn tương đối trong repo HF), cho cột audio_url")
    ap.add_argument("--audio-url-prefix", default=HF_URL)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.relabel_spans and len(args.hyps) < 2:
        sys.exit("--relabel-spans cần ít nhất 2 file hyps (mọi ASR phải cho cùng chữ)")
    cat = Catalog.load(args.catalog)
    hyps = [{h["id"]: h.get("hyp", "") for h in read_jsonl(p)} for p in args.hyps]
    rows = read_jsonl(args.manifest)
    if args.only_with_hyps:
        rows = [r for r in rows if all(r["id"] in h for h in hyps)]
    guard = None
    if args.relabel_spans:
        if args.bench:
            guard = LeakGuard((r for b in args.bench for r in read_jsonl(b)), cat, args.bench_splits)
        else:
            print("  CẢNH BÁO: --relabel-spans không có --bench: chỉ kiểm tra lại reserved_b3, không kiểm text trùng bench")
            guard = LeakGuard([], cat)
    kept, unconfirmed, relabeled = [], [], []
    hours = Counter()
    span_tot, span_ok = Counter(), Counter()
    rl_reason, n_spans_changed, vi_wers, rule_rejected = Counter(), 0, [], []
    examples = {"giữ": [], "bỏ": []}
    for r in rows:
        hs = [h.get(r["id"]) for h in hyps]
        res = agree(r["text_tagged"], [h or "" for h in hs], cat, args.min_agree, args.min_span_frac)
        dur = float(r["duration"]) / 3600
        hours["tổng"] += dur
        for s, v in zip(res["spans"], res["confirmed"]):
            span_tot[s] += 1
            span_ok[s] += v >= args.min_agree
        if any(v < args.min_agree for v in res["confirmed"]):
            unconfirmed.append(r["id"])
        key = "giữ" if res["keep"] else "bỏ"
        if len(examples[key]) < 8:
            examples[key].append((r["id"], r["text_tagged"], [h or "" for h in hs], res["confirmed"]))
        if res["keep"]:
            r["agreement"] = {"confirmed": res["confirmed"], "n_hyps": len(hyps), "min_agree": args.min_agree}
            kept.append(r)
            hours["giữ: lọc"] += dur
            continue
        if not args.relabel_spans:
            continue
        rl = relabel_spans(r["text_tagged"], hs, cat, args.min_agree, args.relabel_max_vi_wer)
        if rl["vi_wer"]:
            vi_wers.append(max(rl["vi_wer"]))
        rule_rejected += [(r["id"],) + tuple(x[1:]) for x in rl.get("rejected", [])]
        if not rl["changed"]:
            rl_reason[rl["reason"]] += 1
            continue
        res2 = agree(rl["text_tagged"], [h or "" for h in hs], cat, args.min_agree, args.min_span_frac)
        if not res2["keep"]:
            rl_reason["gán lại xong vẫn chưa đủ span được xác nhận"] += 1
            continue
        new = dict(r, text_orig=r["text_tagged"], text_tagged=rl["text_tagged"], text=plain(rl["text_tagged"]))
        why = guard.reason(new, new["text"])
        if why:
            rl_reason[f"rò rỉ sau khi gán lại: {why}"] += 1
            continue
        new.update(relabeled=True, relabel={"changed": rl["changed"], "vi_wer": rl["vi_wer"],
                                            "max_vi_wer": args.relabel_max_vi_wer},
                   agreement={"confirmed": res2["confirmed"], "n_hyps": len(hyps), "min_agree": args.min_agree})
        rl_reason["gán lại"] += 1
        n_spans_changed += len(rl["changed"])
        kept.append(new)
        relabeled.append((new, hs))
        hours["giữ: gán lại"] += dur
    write_jsonl(args.out, kept)
    if args.write_unconfirmed_ids:
        Path(args.write_unconfirmed_ids).write_text("".join(i + "\n" for i in unconfirmed), encoding="utf-8")
    n_filter = sum(1 for r in kept if not r.get("relabeled"))
    pct = lambda a, b: 100 * a / max(b, 1e-9)  # noqa: E731
    md = [f"# Lọc đồng thuận — {args.manifest}", "",
          f"Hyps: {', '.join(f'`{p}`' for p in args.hyps)}. --min-agree {args.min_agree}, --min-span-frac {args.min_span_frac}"
          + (f", --relabel-spans (WER tiếng Việt <= {args.relabel_max_vi_wer})" if args.relabel_spans else "") + ".", "",
          f"Giữ **{len(kept)} / {len(rows)}** câu ({hours['giữ: lọc'] + hours['giữ: gán lại']:.2f} / {hours['tổng']:.2f} h, "
          f"{pct(len(kept), len(rows)):.1f}%).", "",
          f"- chỉ lọc: {n_filter} câu, {hours['giữ: lọc']:.2f} h ({pct(n_filter, len(rows)):.1f}% số câu, "
          f"{pct(hours['giữ: lọc'], hours['tổng']):.1f}% số giờ)"]
    if args.relabel_spans:
        md += [f"- thêm nhờ gán lại: {len(relabeled)} câu, {hours['giữ: gán lại']:.2f} h, {n_spans_changed} span được thay", "",
               "Câu không qua lọc, kết quả gán lại:", "", "| Kết quả | Số câu |", "| --- | --- |"]
        md += [f"| {k} | {v} |" for k, v in rl_reason.most_common()]
        if vi_wers:
            edges = [0.05, 0.10, 0.15, 0.20, 0.30, float("inf")]
            cnt = Counter(next(e for e in edges if w <= e) for w in vi_wers)
            md += ["", "WER tiếng Việt (max theo các hyp) của câu xét gán lại:", "", "| <= | Số câu |", "| --- | --- |"]
            md += [f"| {'∞' if e == float('inf') else e} | {cnt[e]} |" for e in edges]
    md += ["", f"Câu còn span chưa được xác nhận: {len(unconfirmed)}.", "",
           "## Tỉ lệ xác nhận theo thuật ngữ (20 thuật ngữ nhiều nhất)", "", "| Thuật ngữ | Số span | Được xác nhận |",
           "| --- | --- | --- |"]
    md += [f"| {t} | {c} | {span_ok[t]} |" for t, c in span_tot.most_common(20)]
    for key, ex in examples.items():
        md += ["", f"## Ví dụ câu {key} khi chỉ lọc", "", "| id | nhãn | hyps | số hyp xác nhận từng span |", "| --- | --- | --- | --- |"]
        md += [f"| {i} | {t} | {' ‖ '.join(h)} | {c} |" for i, t, h, c in ex]
    if args.relabel_spans:
        md += ["", f"## Span bị luật chặn (có dấu tiếng Việt / dài hơn span cũ > 1 token): {len(rule_rejected)}", "",
               "| id | span cũ | span ASR đề xuất | lý do |", "| --- | --- | --- | --- |"]
        md += [f"| {i} | {a} | {b} | {w} |" for i, a, b, w in rule_rejected]
    if relabeled:
        md += ["", "## Ví dụ câu gán lại", "", "| id | span cũ → mới | WER tiếng Việt |", "| --- | --- | --- |"]
        md += [f"| {r['id']} | {'; '.join(f'{a} → {b}' for _, a, b in r['relabel']['changed'])} | {r['relabel']['vi_wer']} |"
               for r, _ in relabeled[:15]]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    if args.listen_csv:
        pick = random.Random(args.seed).sample(relabeled, min(args.listen_n, len(relabeled)))
        Path(args.listen_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.listen_csv, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "audio_url", "audio_rel", "duration", "span_cu_moi", "text_orig", "text_tagged_moi"]
                       + [f"hyp_{Path(p).parent.name}" for p in args.hyps] + ["vi_wer", "decision", "note"])
            for r, hs in sorted(pick, key=lambda x: x[0]["id"]):
                rel = audio_rel(r, args.audio_root)
                w.writerow([r["id"], args.audio_url_prefix + rel, rel, r["duration"],
                            "; ".join(f"{a} -> {b}" for _, a, b in r["relabel"]["changed"]), r["text_orig"], r["text_tagged"]]
                           + [h or "" for h in hs] + [r["relabel"]["vi_wer"], "", ""])
        print(f"Phiếu nghe: {len(pick)} câu -> {args.listen_csv} (decision = ok / sai)")
    extra = f"; gán lại thêm {len(relabeled)} câu ({hours['giữ: gán lại']:.2f} h)" if args.relabel_spans else ""
    print(f"Giữ {len(kept)}/{len(rows)} câu ({hours['giữ: lọc'] + hours['giữ: gán lại']:.2f}/{hours['tổng']:.2f} h); "
          f"chỉ lọc {n_filter} câu ({hours['giữ: lọc']:.2f} h){extra} -> {args.out}; báo cáo {args.out}.report.md")


if __name__ == "__main__":
    main()
