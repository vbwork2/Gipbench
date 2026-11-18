"""Text train cho RNN LM token-level của Gipformer 1.5 (Phần 4, hàng "Gipformer beam 4 + LM").

  python scripts/diagnostics/gip15_lm_text.py --out results/phase4/lm/gip15_rnnlm

Nguồn: chỉ các pool train đã qua cổng rò rỉ (0 critical với mọi bench khoá). Text = model_case_text (bỏ tag, IN HOA như
tokenizer Gipformer, chính tả Anh theo catalog/train-v1, không phải catalog từ bench). Bỏ câu trùng, tách valid 5% theo
hash ổn định của câu. Ra lm_train.txt / lm_valid.txt / meta.json (nguồn, số câu, sha256). Sau đó kiểm bằng
leak_audit --text-files lm_train.txt lm_valid.txt --reserved-terms configs/dataset/reserved_terms_v1.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gipbench.core import decontam, release  # noqa: E402
from gipbench.core.textnorm import Catalog, model_case_text  # noqa: E402

POOLS = ("pool/vimedcss_train@v2", "pool/vivos_train@v2", "pool/cv_other@v2", "pool/vss_cs@e3_v1")
FIELDS = ("text_tagged_orig", "text_tagged", "text")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", nargs="+", default=list(POOLS))
    ap.add_argument("--catalog", default="catalog/train-v1")
    ap.add_argument("--valid-frac", type=float, default=0.05)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rel = ROOT / "data" / "releases"
    cat_path = rel / args.catalog / "catalog.tsv"
    cat = Catalog.load(str(cat_path))
    # Pool train được phép trùng chữ với bench đọc (VIVOS train / test chung câu nhắc: `lexical_overlap_read_corpus`, chỉ
    # review vì audio khác). Với LM thì chữ chính là thứ được học, nên bỏ mọi câu trùng hẳn hoặc gần trùng với bench khoá.
    bench_refs = release.locked_bench_refs()
    bench_rows = [json.loads(line) for ref in bench_refs
                  for line in (rel / ref / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    bench_keys = {decontam.dup_key(decontam.row_text(r)) for r in bench_rows}

    seen: set[str] = set()
    rows, per_pool = [], {}
    for ref in args.pool:
        man = rel / ref / "manifest.jsonl"
        n_in = n_new = n_bench = 0
        for line in man.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            src = next((row[f] for f in FIELDS if row.get(f)), "")
            n_in += 1
            text = model_case_text(src, "upper", cat).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            if decontam.dup_key(src) in bench_keys or decontam.dup_key(text) in bench_keys:
                n_bench += 1
                continue
            n_new += 1
            rows.append({"id": f"{ref}:{row['id']}", "text": text, "ref": ref})
        per_pool[ref] = {"rows": n_in, "unique_kept": n_new, "dropped_bench_exact": n_bench,
                         "manifest_sha256": sha256(man)}
    near = {d["test_id"] for d in decontam.near_duplicates(rows, bench_rows)}
    for r in rows:
        if r["id"] in near:
            per_pool[r["ref"]]["unique_kept"] -= 1
            per_pool[r["ref"]]["dropped_bench_near"] = per_pool[r["ref"]].get("dropped_bench_near", 0) + 1
    train, valid = [], []
    for r in rows:
        if r["id"] in near:
            continue
        h = int(hashlib.sha256(r["text"].encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
        (valid if h < args.valid_frac else train).append(r["text"])

    args.out.mkdir(parents=True, exist_ok=True)
    tr, va = args.out / "lm_train.txt", args.out / "lm_valid.txt"
    tr.write_text("\n".join(train) + "\n", encoding="utf-8")
    va.write_text("\n".join(valid) + "\n", encoding="utf-8")
    meta = {
        "pools": per_pool,
        "excluded_benches": bench_refs,
        "catalog": {"ref": args.catalog, "sha256": sha256(cat_path)},
        "fields": FIELDS,
        "case": "upper",
        "valid_frac": args.valid_frac,
        "train": {"sentences": len(train), "words": sum(len(t.split()) for t in train), "sha256": sha256(tr)},
        "valid": {"sentences": len(valid), "words": sum(len(t.split()) for t in valid), "sha256": sha256(va)},
    }
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("train", "valid")}, indent=1))


if __name__ == "__main__":
    main()
