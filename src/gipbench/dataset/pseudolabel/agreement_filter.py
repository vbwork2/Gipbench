#!/usr/bin/env python3
"""Lọc manifest CS có nhãn máy theo đồng thuận với hyp của các model ASR khác (chỉ ĐỌC hyps, không chạy model).

  python -m gipbench.dataset.pseudolabel.agreement_filter --manifest data/raw/vss_full/vss_cs_pilot.jsonl \
      --hyps results/asr_hyps/qwen3_asr_0.6b/vss_cs_pilot.jsonl results/asr_hyps/fw_large_v3_int8/vss_cs_pilot.jsonl \
      --catalog data/raw/_proxy/catalog_proxy.tsv --min-agree 1 --out data/raw/vss_full/vss_cs_pilot_agreed.jsonl

Dùng cho vss_cs (SEA: tùy chọn, không dùng). Hyps do src/gipbench/infer/asr_hyps.py ghi (định dạng SCHEMA.md mục 3).
Một span <en> của nhãn được xác nhận khi nó xuất hiện (sau normalize_hyp, strict) trong >= --min-agree file hyps.
Giữ câu khi tỉ lệ span được xác nhận >= --min-span-frac (mặc định 1.0: mọi span) VÀ WER phần tiếng Việt của nhãn,
lấy thấp nhất so với các ASR có hyp (lib.agreement.vi_wer, cùng cách tính với relabel_spans), <= --max-vi-wer (0.15).
Câu thiếu hyp ở một file thì coi file đó là hyp rỗng. Ghi: <out> (câu giữ lại, thêm trường agreement) và <out>.report.md.

--relabel-spans (mặc định TẮT): câu không qua lọc được gán lại TỪNG span chưa xác nhận khi MỌI file hyps (>= 2) cho
cùng một chữ ở vị trí span đó sau chuẩn hoá VÀ WER phần tiếng Việt của nhãn so với từng hyp <= --relabel-max-vi-wer
(0.10). Không gán lại cả câu. Câu gán lại: relabeled = true, text_orig = text_tagged gốc, text / text_tagged mới;
sau đó chạy lại kiểm tra rò rỉ (--bench, reserved_b3 trong --catalog) và loại câu rò rỉ.
--write-unconfirmed-ids: ghi id các câu còn span chưa được xác nhận (đầu vào cho tầng ASR kế tiếp).
--listen-csv: --listen-n câu gán lại ngẫu nhiên (--seed) để người nghe kiểm tra.

--mode hard (mặc định --mode confirm = mọi thứ ở trên, không đổi): KHAI THÁC CÂU KHÓ. Chế độ confirm chỉ giữ câu mà ASR khác
xác nhận nhãn giả của model nhỏ, nên model gần như đã đúng sẵn (thiên lệch xác nhận). Chế độ hard làm ngược lại: nhãn = text
của các "thầy" (>= 2 ASR mạnh, khác họ với Gipformer) đồng thuận với nhau; giữ câu khi text thầy chính (file --hyps ĐẦU TIÊN)
có span tiếng Anh (gắn <en> bằng converters.common.auto_tag, tôn trọng catalog, vi_abbrev, vi_loanwords, term_decisions), MỌI
thầy khác có đúng chuỗi token của span, mà "trò" (Gipformer 1.5, --student-hyps) không có (so n-gram, ghép bỏ dấu cách
trùng cũng tính là có: "set up" ~ "setup"). Điều kiện và tên tham số (lib.agreement.mine_hard):
  - thầy đồng thuận: WER từng cặp thầy <= --teacher-pair-wer (0.15), trên text chuẩn hoá strict, tính trên cả câu;
  - phần tiếng Việt (ngoài span) của trò so với nhãn thầy chính: WER <= --student-vi-wer (0.30), để loại audio hỏng / lệch đoạn;
  - bộ chặn như vss_cs_prefilter: reserved_b3, test_only, --block-text, speaker / source / text trùng --bench (nên truyền
    --bench), và --max-en-frac (0.5) tính trên nhãn thầy.
  - --easy-keep-frac F (0): giữ thêm phần F câu (theo hash seed + id, không phụ thuộc thứ tự) mà trò ĐÃ có đủ span, để tập
    ra không chỉ toàn câu khó. Câu hard: mining.mode = "hard"; câu easy được giữ: "easy".
Câu giữ: text_tagged / text = nhãn thầy chính (text_tagged gốc ở text_orig), label_source = teacher_consensus:<engine>+<engine>,
train_only = true, mining = {mode, teacher_spans (span trò thiếu), student_spans (đoạn token trò căn với từng span đó),
teacher_pair_wer (lớn nhất), student_vi_wer}. Báo cáo .md có phễu theo từng điều kiện, 30 cặp (thầy -> trò) hay gặp nhất,
phân bố domain, giờ. Câu thiếu hyp thầy hoặc hyp trò chỉ được đếm "chưa xét".

Chuỗi lệnh dùng chế độ hard (src/gipbench/dataset/pseudolabel/vss_cs_run.py KHÔNG đổi; hyp hai thầy lấy từ cache, không chạy lại ASR):
  1. lấy manifest các câu đã có hyp của CẢ HAI thầy nhưng chưa có hyp trò (ứng viên = đầu ra src/gipbench/dataset/pseudolabel/vss_cs_prefilter.py, vd
     data/raw/vss_full/vss_cs_pre_rebuild.jsonl). Cache fw large-v3 chỉ có ở câu Qwen chưa xác nhận hết span: đó chính là phần khó.
     Chạy lệnh ở bước 3 một lần với --student-hyps trỏ tới file CHƯA có và thêm --write-need-student <need.jsonl>;
  2. hyp trò = Gipformer 1.5 gốc trên cùng audio (CPU, int8, trong venv decode như E0, không chạy cùng job GPU / ASR khác):
       python -m gipbench.infer.decode --bench <need.jsonl> --model-dir models/gip15 --system vss_gip15 \
           --results results/distill --decoding-method greedy_search --threads 16 --quantize int8
     (-> results/distill/vss_gip15/hyps.jsonl);
  3. khai thác (thầy chính = file đầu: nên để faster-whisper trước vì chính tả Anh tốt hơn):
       python -m gipbench.dataset.pseudolabel.agreement_filter --mode hard --manifest data/raw/vss_full/vss_cs_pre_rebuild.jsonl \
           --hyps results/asr_hyps/fw_large_v3_int8 results/asr_hyps/qwen3_asr_0.6b \
           --student-hyps results/distill/vss_gip15/hyps.jsonl \
           --catalog data/raw/_proxy/catalog_proxy.tsv --bench data/raw/_proxy/bench_proxy.jsonl \
           --hash-index results/asr_hyps/vss_audio_sha256.tsv --audio-root data/dl/VietSuperSpeech \
           --easy-keep-frac 0.15 --seed 0 --out data/raw/vss_full/vss_cs_hard.jsonl
Hyp trò tra theo id (file của decode.py không có hash audio), hyp thầy tra theo id + hash nếu có --hash-index.
"""
import argparse
import csv
import random
import sys
from collections import Counter
from pathlib import Path

from gipbench.dataset.pseudolabel.agreement import HARD_STAGES, agree, mine_hard, plain, relabel_spans, vi_wer
from gipbench.infer.cache import HashIndex, HypLookup, audio_rel
from gipbench.core.leak import LeakGuard, add_block_arg
from gipbench.core.manifest import read_jsonl, write_jsonl
from gipbench.core.textnorm import Catalog, plain_text

HF_URL = "https://huggingface.co/datasets/thanhnew2001/VietSuperSpeech/resolve/main/"


def run_hard(args):
    """--mode hard: xem docstring đầu file và lib.agreement.mine_hard."""
    if len(args.hyps) < 2 or not args.student_hyps:
        sys.exit("--mode hard cần >= 2 file / thư mục --hyps của thầy và --student-hyps")
    if args.relabel_spans or args.tiered:
        sys.exit("--mode hard không dùng với --relabel-spans / --tiered")
    cat = Catalog.load(args.catalog)
    look = HypLookup(args.hyps, HashIndex(args.hash_index) if args.hash_index else None, args.audio_root)
    stu = HypLookup([args.student_hyps])  # file của decode.py không có hash audio: tra theo id
    names = [Path(p).name if Path(p).is_dir() else Path(p).parent.name for p in args.hyps]
    if args.bench:
        guard = LeakGuard((r for b in args.bench for r in read_jsonl(b)), cat, args.bench_splits, args.block_text)
    else:
        print("  CẢNH BÁO: --mode hard không có --bench: chỉ kiểm tra reserved_b3 / test_only / danh sách chặn, không kiểm "
              "speaker, source, text trùng bench")
        guard = LeakGuard([], cat, block_files=args.block_text)
    keys = [k for k, _ in HARD_STAGES]
    rows = read_jsonl(args.manifest)
    kept, pending, reasons, need = [], Counter(), Counter(), []
    fail_n, fail_h = Counter(), Counter()  # theo chỉ số điều kiện đầu tiên không đạt (len(keys) = qua hết)
    n_eval, hours = 0, Counter()
    pairs, doms, dom_h = Counter(), Counter(), Counter()
    for r in rows:
        hs, sh = look.get(r), stu.get(r)[0]
        if any(h is None for h in hs):
            pending["thiếu hyp thầy"] += 1
            continue
        dur = float(r["duration"]) / 3600
        own = r.get("text") or plain_text(r["text_tagged"])
        res = mine_hard(hs, sh, cat, args.teacher_pair_wer, args.student_vi_wer, args.max_en_frac,
                        leak=lambda t: guard.reason(r, t) or guard.reason(r, own))
        if sh is None:  # thiếu hyp trò: chưa xét; chỉ câu đã qua mọi điều kiện phía thầy mới đáng chạy decode
            pending["thiếu hyp trò"] += 1
            if res["status"] == "need_student":
                need.append(r)
            continue
        k = keys.index(res["reason"]) if res["reason"] else len(keys)
        n_eval += 1
        hours["tổng"] += dur
        fail_n[k] += 1
        fail_h[k] += dur
        if res["reason"]:
            reasons[res["reason"]] += 1
            continue
        keep = res["status"] == "hard" or random.Random(f"{args.seed}:{r['id']}").random() < args.easy_keep_frac
        hours[res["status"]] += dur
        reasons[res["status"]] += 1
        if res["status"] == "easy" and not keep:
            continue
        mining = {"mode": res["status"], "teacher_spans": res["teacher_spans"], "student_spans": res["student_spans"],
                  "teacher_pair_wer": res["teacher_pair_wer"], "student_vi_wer": res["student_vi_wer"]}
        kept.append(dict(r, text_orig=r["text_tagged"], text_tagged=res["text_tagged"], text=plain(res["text_tagged"]),
                         label_source="teacher_consensus:" + "+".join(names), train_only=True, mining=mining))
        hours["giữ: " + res["status"]] += dur
        if res["status"] == "hard":
            pairs.update(zip(res["teacher_spans"], res["student_spans"]))
        d = r.get("domain") or "other"
        doms[d] += 1
        dom_h[d] += dur
    write_jsonl(args.out, kept)
    if args.write_need_student:
        write_jsonl(args.write_need_student, need)
    n_hard, n_easy = sum(1 for r in kept if r["mining"]["mode"] == "hard"), sum(1 for r in kept if r["mining"]["mode"] == "easy")
    pct = lambda a, b: 100 * a / max(b, 1e-9)  # noqa: E731
    h_keep = hours["giữ: hard"] + hours["giữ: easy"]
    md = [f"# Khai thác câu khó — {args.manifest}", "",
          f"Thầy (thầy chính là file đầu): {', '.join(f'`{p}`' for p in args.hyps)}. Trò: `{args.student_hyps}`. "
          f"--teacher-pair-wer {args.teacher_pair_wer:g}, --student-vi-wer {args.student_vi_wer:g}, --max-en-frac "
          f"{args.max_en_frac:g}, --easy-keep-frac {args.easy_keep_frac:g}, --seed {args.seed}, bench: "
          f"{guard.n_bench} câu ({' '.join(args.bench_splits)}), {len(guard.reserved)} thuật ngữ reserved_b3, danh sách chặn "
          f"{len(guard.block)} câu.", "",
          f"Giữ **{len(kept)} / {n_eval}** câu ({h_keep:.2f} / {hours['tổng']:.2f} h, {pct(len(kept), n_eval):.1f}%): "
          f"{n_hard} câu khó ({hours['giữ: hard']:.2f} h) + {n_easy} câu dễ được giữ ngẫu nhiên ({hours['giữ: easy']:.2f} h).", "",
          "## Số câu qua từng điều kiện (phễu, xét theo thứ tự này)", "", "| Điều kiện | Câu qua | Giờ qua |", "| --- | --- | --- |",
          f"| tất cả câu có đủ hyp (mẫu số) | {n_eval} | {hours['tổng']:.2f} |"]
    for i, (_, label) in enumerate(HARD_STAGES):
        md.append(f"| {label} | {sum(v for j, v in fail_n.items() if j > i)} | {sum(v for j, v in fail_h.items() if j > i):.2f} |")
    md += [f"| trò thiếu >= 1 span (khó) | {reasons['hard']} | {hours['hard']:.2f} |",
           f"| trò có đủ mọi span (dễ, chỉ giữ {args.easy_keep_frac:g}) | {reasons['easy']} | {hours['easy']:.2f} |"]
    if pending:
        md += ["", "Chưa xét (thiếu hyp, không tính vào mẫu số): " + "; ".join(f"{k}: {v}" for k, v in pending.items())
               + f". Trong số câu thiếu hyp trò, {len(need)} câu đã qua mọi điều kiện phía thầy (cần decode trò"
               + (f", ghi ở `{args.write_need_student}`" if args.write_need_student else ", dùng --write-need-student") + ")."]
    md += ["", "## 30 cặp (thầy -> trò) hay gặp nhất (span trò thiếu; trống = trò không ra token nào ở đó)", "",
           "| Thầy | Trò | Số câu |", "| --- | --- | --- |"]
    md += [f"| {t} | {s or '(trống)'} | {c} |" for (t, s), c in pairs.most_common(30)]
    md += ["", "## Phân bố domain của câu giữ", "", "| Domain | Câu | Giờ |", "| --- | --- | --- |"]
    md += [f"| {d} | {c} | {dom_h[d]:.2f} |" for d, c in doms.most_common()]
    Path(args.out + ".report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    pend = f"; chưa xét {sum(pending.values())} câu (thiếu hyp; {len(need)} câu cần decode trò)" if pending else ""
    print(f"Hard: giữ {len(kept)}/{n_eval} câu ({h_keep:.2f}/{hours['tổng']:.2f} h): {n_hard} khó + {n_easy} dễ{pend} "
          f"-> {args.out}; báo cáo {args.out}.report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--hyps", nargs="+", required=True)
    ap.add_argument("--catalog")
    ap.add_argument("--min-agree", type=int, default=1, help="số file hyps tối thiểu phải chứa một span")
    ap.add_argument("--min-span-frac", type=float, default=1.0, help="tỉ lệ span tối thiểu phải được xác nhận")
    ap.add_argument("--only-with-hyps", action="store_true", help="chỉ xét câu có hyp ở mọi file (chạy thử một phần)")
    ap.add_argument("--max-vi-wer", type=float, default=0.15,
                    help="nhánh chỉ lọc: WER tiếng Việt thấp nhất so với các ASR có hyp phải <= ngưỡng (inf = tắt)")
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
    ap.add_argument("--hash-index", help="chỉ mục audio_rel -> sha256 (src/gipbench/infer/cache.py): tra hyps theo id + hash audio")
    ap.add_argument("--tiered", action="store_true",
                    help="file hyps đầu là tầng 1 (bắt buộc); câu chưa có hyp tầng 1, hoặc còn span chưa xác nhận mà "
                         "thiếu hyp tầng sau, được đếm riêng là 'chưa chạy' thay vì 'bỏ'")
    ap.add_argument("--mode", choices=["confirm", "hard"], default="confirm",
                    help="confirm (mặc định): giữ câu mà ASR khác xác nhận nhãn giả; hard: khai thác câu khó (xem đầu file)")
    ap.add_argument("--student-hyps", help="[hard] hyp của trò (Gipformer 1.5 trên cùng audio; decode.py -> hyps.jsonl), tra theo id")
    ap.add_argument("--teacher-pair-wer", type=float, default=0.15, help="[hard] WER cặp thầy tối đa (cả câu)")
    ap.add_argument("--student-vi-wer", type=float, default=0.30, help="[hard] WER tiếng Việt (ngoài span) của trò tối đa")
    ap.add_argument("--easy-keep-frac", type=float, default=0.0,
                    help="[hard] giữ thêm phần này (theo seed + id) số câu mà trò ĐÃ có đủ span")
    ap.add_argument("--max-en-frac", type=float, default=0.5, help="[hard] tỉ lệ token trong <en> tối đa của nhãn thầy")
    ap.add_argument("--write-need-student", help="[hard] ghi manifest các câu đủ hyp thầy nhưng chưa có hyp trò (đầu vào decode.py)")
    add_block_arg(ap)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.mode == "hard":
        return run_hard(args)
    if args.relabel_spans and len(args.hyps) < 2:
        sys.exit("--relabel-spans cần ít nhất 2 file hyps (mọi ASR phải cho cùng chữ)")
    cat = Catalog.load(args.catalog)
    look = HypLookup(args.hyps, HashIndex(args.hash_index) if args.hash_index else None, args.audio_root)
    rows = read_jsonl(args.manifest)
    if args.only_with_hyps:
        rows = [r for r in rows if all(h is not None for h in look.get(r))]
    pending = Counter()
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
    vw_all, n_vi_drop = [], 0  # WER tiếng Việt (min theo ASR) của câu đã qua lọc span
    examples = {"giữ": [], "bỏ": []}
    for r in rows:
        hs = look.get(r)
        if args.tiered and hs[0] is None:
            pending["chưa có hyp tầng 1"] += 1
            continue
        res = agree(r["text_tagged"], [h or "" for h in hs], cat, args.min_agree, args.min_span_frac)
        if any(v < args.min_agree for v in res["confirmed"]):
            unconfirmed.append(r["id"])
        if args.tiered and not res["keep"] and any(h is None for h in hs[1:]):
            pending["chờ tầng 2 (còn span chưa xác nhận, thiếu hyp tầng sau)"] += 1
            continue
        dur = float(r["duration"]) / 3600
        hours["tổng"] += dur
        for s, v in zip(res["spans"], res["confirmed"]):
            span_tot[s] += 1
            span_ok[s] += v >= args.min_agree
        key = "giữ" if res["keep"] else "bỏ"
        if len(examples[key]) < 8:
            examples[key].append((r["id"], r["text_tagged"], [h or "" for h in hs], res["confirmed"]))
        if res["keep"]:
            avail = [h for h in hs if (h or "").strip()]
            vw = min((vi_wer(r["text_tagged"], h, cat) for h in avail), default=float("inf"))
            vw_all.append(vw)
            if vw > args.max_vi_wer:
                hours["bỏ: WER tiếng Việt"] += dur
                n_vi_drop += 1
                continue
            r["agreement"] = {"confirmed": res["confirmed"], "n_hyps": len(args.hyps), "min_agree": args.min_agree,
                              "vi_wer": round(vw, 4)}
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
                   agreement={"confirmed": res2["confirmed"], "n_hyps": len(args.hyps), "min_agree": args.min_agree})
        rl_reason["gán lại"] += 1
        n_spans_changed += len(rl["changed"])
        kept.append(new)
        relabeled.append((new, hs))
        hours["giữ: gán lại"] += dur
    write_jsonl(args.out, kept)
    if args.write_unconfirmed_ids:
        Path(args.write_unconfirmed_ids).write_text("".join(i + "\n" for i in unconfirmed), encoding="utf-8")
    n_filter = sum(1 for r in kept if not r.get("relabeled"))
    n_eval = len(rows) - sum(pending.values())
    pct = lambda a, b: 100 * a / max(b, 1e-9)  # noqa: E731
    md = [f"# Lọc đồng thuận — {args.manifest}", "",
          f"Hyps: {', '.join(f'`{p}`' for p in args.hyps)}. --min-agree {args.min_agree}, --min-span-frac {args.min_span_frac}"
          f", --max-vi-wer {args.max_vi_wer:g}"
          + (f", --relabel-spans (WER tiếng Việt <= {args.relabel_max_vi_wer})" if args.relabel_spans else "")
          + (", tra cache theo id + hash audio" if args.hash_index else "") + ".", "",
          f"Giữ **{len(kept)} / {n_eval}** câu ({hours['giữ: lọc'] + hours['giữ: gán lại']:.2f} / {hours['tổng']:.2f} h, "
          f"{pct(len(kept), n_eval):.1f}%).", "",
          f"- chỉ lọc: {n_filter} câu, {hours['giữ: lọc']:.2f} h ({pct(n_filter, n_eval):.1f}% số câu, "
          f"{pct(hours['giữ: lọc'], hours['tổng']):.1f}% số giờ)"]
    if vw_all:
        edges = [0.05, 0.10, 0.15, 0.20, 0.30, float("inf")]
        cnt = Counter(next(e for e in edges if w <= e) for w in vw_all)
        mid = sum(1 for w in vw_all if 0.10 < w <= 0.15)
        md += [f"- qua lọc span nhưng bị loại vì WER tiếng Việt > {args.max_vi_wer:g}: {n_vi_drop} câu, "
               f"{hours['bỏ: WER tiếng Việt']:.2f} h", "",
               f"WER tiếng Việt (thấp nhất theo các ASR có hyp) của {len(vw_all)} câu qua lọc span "
               f"({mid} câu trong khoảng 0.10–0.15):", "", "| <= | Số câu |", "| --- | --- |"]
        md += [f"| {'∞' if e == float('inf') else e} | {cnt[e]} |" for e in edges]
        md += [""]
    if args.relabel_spans:
        md += [f"- thêm nhờ gán lại: {len(relabeled)} câu, {hours['giữ: gán lại']:.2f} h, {n_spans_changed} span được thay", "",
               "Câu không qua lọc, kết quả gán lại:", "", "| Kết quả | Số câu |", "| --- | --- |"]
        md += [f"| {k} | {v} |" for k, v in rl_reason.most_common()]
        if vi_wers:
            edges = [0.05, 0.10, 0.15, 0.20, 0.30, float("inf")]
            cnt = Counter(next(e for e in edges if w <= e) for w in vi_wers)
            md += ["", "WER tiếng Việt (max theo các hyp) của câu xét gán lại:", "", "| <= | Số câu |", "| --- | --- |"]
            md += [f"| {'∞' if e == float('inf') else e} | {cnt[e]} |" for e in edges]
    if pending:
        md += ["", f"Chưa xét (thiếu hyp trong cache, không tính vào mẫu số): "
               + "; ".join(f"{k}: {v}" for k, v in pending.items()) + "."]
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
                       + [f"hyp_{Path(p).name if Path(p).is_dir() else Path(p).parent.name}" for p in args.hyps]
                       + ["vi_wer", "decision", "note"])
            for r, hs in sorted(pick, key=lambda x: x[0]["id"]):
                rel = audio_rel(r, args.audio_root)
                w.writerow([r["id"], args.audio_url_prefix + rel, rel, r["duration"],
                            "; ".join(f"{a} -> {b}" for _, a, b in r["relabel"]["changed"]), r["text_orig"], r["text_tagged"]]
                           + [h or "" for h in hs] + [r["relabel"]["vi_wer"], "", ""])
        print(f"Phiếu nghe: {len(pick)} câu -> {args.listen_csv} (decision = ok / sai)")
    extra = f"; gán lại thêm {len(relabeled)} câu ({hours['giữ: gán lại']:.2f} h)" if args.relabel_spans else ""
    pend = f"; chưa xét {sum(pending.values())} câu (thiếu hyp)" if pending else ""
    print(f"Giữ {len(kept)}/{n_eval} câu ({hours['giữ: lọc'] + hours['giữ: gán lại']:.2f}/{hours['tổng']:.2f} h); "
          f"chỉ lọc {n_filter} câu ({hours['giữ: lọc']:.2f} h){extra}{pend} -> {args.out}; báo cáo {args.out}.report.md")


if __name__ == "__main__":
    main()
