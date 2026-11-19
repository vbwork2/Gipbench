#!/usr/bin/env python3
"""ĐỀ XUẤT danh sách thuật ngữ giữ riêng (reserved_terms_v1) để người dùng duyệt. KHÔNG tự chốt, không ghi đè file chốt.

  python -m gipbench.dataset.release.propose_reserved_terms [--fraction 0.5] [--out configs/dataset/reserved_terms_v1.proposal.txt]

Thuật ngữ giữ riêng = thuật ngữ tiếng Anh của bench (span <en> của bench khoá) mà dự án CAM KẾT không đưa vào bất kỳ text FT
nào (train, TTS, LM, hotwords). Nhờ đó bench có lát "FT-unseen" thật: tăng term recall ở lát này không thể do ghi nhớ.
Quy tắc đề xuất (đổi được bằng tham số):
  1. ứng viên = thuật ngữ bench KHÔNG xuất hiện trong text của bất kỳ pool / dev hiện có (giữ riêng chúng không làm mất dữ liệu
     train nào; thuật ngữ đã có trong train mà giữ riêng thì phải loại câu train đó);
  2. chọn --fraction ứng viên theo hash ổn định (--salt) trong từng domain, phần còn lại được phép dùng để FT / TTS.
Đánh đổi bạn quyết: giữ riêng nhiều thì lát FT-unseen lớn và chắc chắn hơn, nhưng cấm dạy model chính các thuật ngữ tech / finance
mục tiêu (TTS không được sinh câu chứa chúng). Ra: file đề xuất (dòng "thuật ngữ<TAB>bench; domain; số span", dòng # là chú
thích, đọc được bằng --reserved-terms) + bảng thống kê. Bạn xem, sửa tay, rồi đổi tên thành reserved_terms_v1.txt.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.decontam import find_terms_in_texts, row_text
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.release.build_train_catalog import span_terms


def stable_frac(term: str, salt: str) -> float:
    return int(hashlib.sha256(f"{salt}|{term}".encode("utf-8")).hexdigest()[:8], 16) / 2 ** 32


def default_pool_refs():
    import yaml
    cfg = yaml.safe_load(R.releases_config().read_text(encoding="utf-8"))["releases"]
    return [r for r in cfg if r.split("/")[0] in ("pool", "dev")]


def propose(bench_refs, pool_refs, fraction, salt):
    terms = {}  # thuật ngữ -> {"bench": Counter, "domain": Counter, "spans": int}
    for ref in bench_refs:
        rows = read_jsonl(str(R.resolve(ref).manifest))
        for k, doms in span_terms(rows).items():
            t = terms.setdefault(k, {"bench": Counter(), "domain": Counter()})
            t["bench"][ref] += sum(doms.values())
            t["domain"].update(doms)
    texts = []
    for ref in pool_refs:
        try:
            texts += [row_text(r) for r in read_jsonl(str(R.resolve(ref, verify=False).manifest))]
        except R.ReleaseError:
            print(f"BỎ {ref}: chưa có release (không kiểm được thuật ngữ đã có trong pool này)")
    seen = set(find_terms_in_texts(list(terms), texts))
    cand = {k: v for k, v in terms.items() if k not in seen}
    by_dom = defaultdict(list)
    for k, v in cand.items():
        dom = [d for d, _ in v["domain"].most_common() if d and d != "other"]
        by_dom[dom[0] if dom else "(không rõ)"].append(k)
    picked = {k for d, ks in by_dom.items() for k in ks if stable_frac(k, salt) < fraction}
    stats = {"bench_terms": len(terms), "seen_in_pools": len(seen), "candidates": len(cand), "reserved": len(picked),
             "by_domain": {d: {"candidates": len(ks), "reserved": sum(k in picked for k in ks)} for d, ks in sorted(by_dom.items())}}
    return terms, picked, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", action="append", help="bench release (mặc định: mọi bench khoá)")
    ap.add_argument("--pool", action="append", help="pool / dev release để kiểm thuật ngữ đã có (mặc định: mọi pool / dev trong releases.yaml)")
    ap.add_argument("--fraction", type=float, default=0.5, help="tỉ lệ ứng viên được giữ riêng trong mỗi domain")
    ap.add_argument("--salt", default="reserved-terms-v1")
    ap.add_argument("--out", default="configs/dataset/reserved_terms_v1.proposal.txt")
    args = ap.parse_args(argv)
    bench = args.bench or R.locked_bench_refs()
    pools = args.pool or default_pool_refs()
    terms, picked, st = propose(bench, pools, args.fraction, args.salt)
    L = ["# ĐỀ XUẤT reserved_terms_v1 (CHƯA CHỐT). Xem docs: tool dataset.release.propose_reserved_terms.",
         f"# bench: {', '.join(bench)}", f"# pool / dev đã kiểm: {', '.join(pools)}",
         f"# fraction {args.fraction}, salt {args.salt}; {json.dumps(st, ensure_ascii=False)}",
         "# Sửa tay được (thêm / bớt dòng), rồi đổi tên thành reserved_terms_v1.txt. Cột 2+ chỉ là chú thích."]
    for k in sorted(picked):
        t = terms[k]
        L.append(f"{k}\t{'+'.join(t['bench'])}; {t['domain'].most_common(1)[0][0] or '?'}; {sum(t['bench'].values())} span")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(st, ensure_ascii=False, indent=1))
    print(f"đã ghi đề xuất {out} ({len(picked)} thuật ngữ). Chưa phải bản chốt.")
    return st


if __name__ == "__main__":
    main()
