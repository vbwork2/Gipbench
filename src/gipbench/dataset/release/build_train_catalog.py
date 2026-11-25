#!/usr/bin/env python3
"""Dựng catalog thuật ngữ TỪ TEXT TRAIN, không chứa span của bench (audit 17/11/2026, R05, IP-05).

  python -m gipbench.dataset.release.build_train_catalog \
      --source pool/vimedcss_train@full:text_tagged_orig --source pool/vss_cs@e3_v1:text_tagged \
      --reserved-terms configs/dataset/reserved_terms_v1.txt --out data/catalogs/train_v1/catalog.tsv

Vì sao: catalog/v0-draft có 1 618 span lấy từ chính bench; curate (--max-per-term), select_ckpt (term_acc) và make_text
(lexicon TTS) dùng nó nên cải thiện trên bench là ghi nhớ. Catalog này chỉ gồm thuật ngữ nằm trong span <en> của text TRAIN,
trừ:
  * mọi thuật ngữ là span <en> của các bench khoá (releases.yaml flags.locked; hoặc --bench REF ...);
  * mọi thuật ngữ trong --reserved-terms (BẮT BUỘC, không rỗng: chốt TRƯỚC khi dựng dữ liệu, xem docs).
Thuật ngữ chuẩn hoá bằng core.decontam.dup_key (cùng khoá với bộ kiểm rò rỉ). category = domain phổ biến nhất của các câu chứa nó
("other" / trống thì để trống); reserved_b3 = 0 mọi dòng (thuật ngữ giữ riêng đã bị loại hẳn, không phải đánh dấu).
Ra: <out> (catalog.tsv) + <out>.meta.json (nguồn + sha256, số thuật ngữ từng bước, danh sách bench đã loại). Đăng ký thành release
`catalog/train-v1` (flags.from_bench: false) bằng configs/dataset/releases.yaml + dataset.release.register; --preview chỉ in số liệu.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from gipbench.core import release as R
from gipbench.core.decontam import dup_key, read_lines
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import read_jsonl
from gipbench.core.textnorm import parse_tagged

HEADER = "term\tcategory\tvariants\tphonetic\treserved_b3"
_SPEC = re.compile(r"(.+?):([A-Za-z_][A-Za-z0-9_]*)")


def span_terms(rows, field="text_tagged"):
    """{thuật ngữ (dup_key): Counter(domain của câu chứa nó)} cho các span <en> của `field`."""
    out = defaultdict(Counter)
    for r in rows:
        for seg, is_en in parse_tagged(r.get(field) or ""):
            if is_en:
                k = dup_key(seg)
                if k:
                    out[k][str(r.get("domain") or "")] += 1
    return out


def load_source(spec):
    """`<release ref hoặc đường dẫn manifest>[:trường]` -> (rows, trường). Chữ ổ đĩa Windows (`d:\\...`) không bị tách."""
    m = _SPEC.fullmatch(spec)
    path, field = (m.group(1), m.group(2)) if m and len(m.group(1)) > 1 else (spec, "text_tagged")
    if path.split("/")[0] in R.KINDS:
        return read_jsonl(str(R.resolve(path).manifest)), field
    return read_jsonl(path), field


def build(sources, reserved_path, bench_refs=None, min_count=1):
    reserved = sorted({t for t in (dup_key(x) for x in read_lines(reserved_path)) if t})
    if not reserved:
        raise SystemExit(f"LỖI --reserved-terms {reserved_path} rỗng: phải chốt danh sách thuật ngữ giữ riêng trước khi dựng catalog")
    cnt = defaultdict(Counter)
    meta_src = []
    for spec in sources:
        rows, field = load_source(spec)
        st = span_terms(rows, field)
        for k, c in st.items():
            cnt[k].update(c)
        meta_src.append({"source": spec, "field": field, "rows": len(rows), "terms": len(st)})
    all_terms = {k: c for k, c in cnt.items() if sum(c.values()) >= min_count}
    refs = R.locked_bench_refs() if bench_refs is None else list(bench_refs)
    bench_terms, bench_meta = set(), []
    for ref in refs:
        st = span_terms(read_jsonl(str(R.resolve(ref).manifest)))
        bench_terms |= set(st)
        bench_meta.append({"bench": ref, "span_terms": len(st), "also_in_train": len(set(st) & set(all_terms))})
    rs = set(reserved)
    kept = {k: c for k, c in all_terms.items() if k not in bench_terms and k not in rs}
    meta = {"sources": meta_src, "min_count": min_count, "reserved_terms": str(reserved_path), "n_reserved": len(rs),
            "reserved_sha256": R.sha256(reserved_path), "benches_excluded": bench_meta,
            "n_train_terms": len(all_terms), "n_removed_bench": len(set(all_terms) & bench_terms),
            "n_removed_reserved": len(set(all_terms) & rs), "n_catalog": len(kept)}
    return kept, meta


def write_catalog(kept, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for k in sorted(kept):
        dom = [d for d, _ in kept[k].most_common() if d and d != "other"]
        lines.append(f"{k}\t{dom[0] if dom else ''}\t\t\t0")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", required=True, metavar="NGUỒN[:TRƯỜNG]",
                    help="pool release (vd pool/vimedcss_train@full) hoặc manifest; :trường chứa tag <en> (mặc định text_tagged)")
    ap.add_argument("--reserved-terms", required=True)
    ap.add_argument("--bench", action="append", help="bench release loại thuật ngữ (mặc định: mọi bench khoá)")
    ap.add_argument("--min-count", type=int, default=1, help="thuật ngữ xuất hiện ít hơn N lần trong train bị bỏ")
    ap.add_argument("--out", help="catalog.tsv (bắt buộc trừ --preview)")
    ap.add_argument("--preview", action="store_true", help="chỉ in số liệu, không ghi file")
    args = ap.parse_args(argv)
    if not args.preview and not args.out:
        sys.exit("cần --out (hoặc --preview)")
    kept, meta = build(args.source, args.reserved_terms, args.bench, args.min_count)
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"}, ensure_ascii=False, indent=1))
    if args.preview:
        return meta
    write_catalog(kept, args.out)
    meta["catalog_sha256"] = sha256_file(args.out)
    Path(str(args.out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"đã ghi {args.out} ({len(kept)} thuật ngữ) + {args.out}.meta.json")
    return meta


if __name__ == "__main__":
    main()
