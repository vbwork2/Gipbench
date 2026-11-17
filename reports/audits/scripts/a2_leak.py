"""Audit 2: contamination matrix train/dev pools x benchmarks (metadata + text level). Read-only; prints CSV + notes."""
import json, sys, re, collections, time
sys.path.insert(0, "src")
from gipbench.core.decontam import dup_key, near_duplicates, test_terms_from_rows, find_terms_in_texts
from gipbench.core.textnorm import parse_tagged

OUT = sys.argv[1] if len(sys.argv) > 1 else None


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


POOLS = {
    "vimedcss_train": "data/releases/pool/vimedcss_train@full/manifest.jsonl",
    "vss_cs": "data/releases/pool/vss_cs@e3_v1/manifest.jsonl",
    "vivos_train": "data/releases/pool/vivos_train@e3_v1/manifest.jsonl",
    "cv27_train": "data/releases/pool/cv_train@v1/manifest.jsonl",
    "distill_vss": "data/releases/pool/distill@e3_v1/manifest.jsonl",
    "confusable": "data/releases/pool/confusable_train@e3_v1/manifest.jsonl",
    "dev_cs": "data/releases/dev/dev_cs@e3_v1/manifest.jsonl",
    "dev_vi": "data/releases/dev/dev_vi@e3_v1/manifest.jsonl",
    "e3v1_train_curated": "data/e3_v1/train_curated.jsonl",
    "e3v1b_train_curated": "data/e3_v1b/train_curated.jsonl",
}
BENCH = {
    "G0": "data/benchmarks/general_v1/manifest.jsonl",
    "C0": "data/benchmarks/code_switch_v1/manifest.jsonl",
    "C1": "data/benchmarks/cs_ggroup_v1/manifest.jsonl",
    "E0": "data/benchmarks/external_v1/manifest.jsonl",
    "v0draft": "data/releases/bench/v0-draft/manifest.jsonl",
}
YT = re.compile(r"(?:youtube:|yt_)([A-Za-z0-9_-]{11})")


def srckey(r):
    for f in ("source", "speaker"):
        m = YT.search(str(r.get(f) or ""))
        if m:
            return "yt:" + m.group(1)
    s = str(r.get("source") or "")
    return s or None


def text_of(r):
    return r.get("text_tagged") or r.get("text") or ""


pools = {k: rd(v) for k, v in POOLS.items()}
bench = {k: rd(v) for k, v in BENCH.items()}
for k, v in pools.items():
    hrs = sum(float(r.get("duration") or 0) for r in v) / 3600
    print(f"POOL {k}: {len(v)} rows {hrs:.2f} h datasets={collections.Counter(str(r.get('dataset')).split(':')[0] for r in v).most_common(8)}")

rows_out = ["pool,bench,bench_dataset,test_rows,id,speaker,source_raw,source_norm(video),text_exact,text_exact_ge4tok,near_dup_0.8"]
t0 = time.time()
for pk, P in pools.items():
    pid = {r["id"] for r in P}
    pspk = {r.get("speaker") for r in P} - {None, ""}
    psrc = {r.get("source") for r in P} - {None, ""}
    psk = {srckey(r) for r in P} - {None}
    ptx = {dup_key(text_of(r)) for r in P} - {""}
    for bk, Bm in bench.items():
        byds = collections.defaultdict(list)
        for r in Bm:
            byds[str(r.get("dataset"))].append(r)
        for ds, T in sorted(byds.items()):
            n_id = sum(r["id"] in pid for r in T)
            n_spk = sum(r.get("speaker") in pspk for r in T)
            n_src = sum(r.get("source") in psrc for r in T)
            n_sk = sum(srckey(r) in psk for r in T if srckey(r))
            ks = [dup_key(text_of(r)) for r in T]
            n_tx = sum(k in ptx for k in ks)
            n_tx4 = sum(k in ptx and len(k.split()) >= 4 for k in ks)
            nd = ""
            if (n_tx or n_sk or n_spk) or bk in ("C0", "C1") or ds in ("vimedcss_test",):
                nd = len(near_duplicates(T, P, 0.8))
            if any([n_id, n_spk, n_src, n_sk, n_tx, nd]):
                rows_out.append(f"{pk},{bk},{ds},{len(T)},{n_id},{n_spk},{n_src},{n_sk},{n_tx},{n_tx4},{nd}")
    print(f"  done {pk} {time.time()-t0:.0f}s", flush=True)
print("\n".join(rows_out))
if OUT:
    open(OUT, "w", encoding="utf-8").write("\n".join(rows_out) + "\n")

# ---- video-level detail ViMedCSS
def vids(rows):
    return {srckey(r) for r in rows if srckey(r) and srckey(r).startswith("yt:")}

tr_v = vids(pools["vimedcss_train"])
for name, rows in [("C0 hard", bench["C0"]), ("E0 vimedcss_test", [r for r in bench["E0"] if r["dataset"] == "vimedcss_test"]),
                   ("dev_cs ViMedCSS", [r for r in pools["dev_cs"] if str(r.get("dataset")).startswith("ViMedCSS")])]:
    v = vids(rows)
    aff = [r for r in rows if srckey(r) in tr_v]
    hrs = sum(float(r.get("duration") or 0) for r in aff) / 3600
    print(f"VIDEO {name}: {len(v)} videos, {len(v & tr_v)} also in ViMedCSS train; rows affected {len(aff)}/{len(rows)} ({100*len(aff)/max(1,len(rows)):.1f}%), {hrs:.2f} h")
hv = vids(bench["C0"]); tv = vids([r for r in bench["E0"] if r["dataset"] == "vimedcss_test"]); dv = vids(pools["dev_cs"])
print("VIDEO hard∩test", len(hv & tv), "hard∩dev", len(hv & dv), "test∩dev", len(tv & dv))

# ---- term exposure: bench <en> terms vs training text (actual E3 v1 curated)
for bk in ("C0", "C1", "v0draft"):
    terms = test_terms_from_rows(bench[bk])
    for pk in ("e3v1_train_curated", "vimedcss_train", "vss_cs"):
        hit = find_terms_in_texts(terms, (text_of(r) for r in pools[pk]))
        n5 = sum(1 for c in hit.values() if c >= 5)
        print(f"TERMS {bk}: {len(terms)} distinct <en> terms; in {pk}: {len(hit)} ({100*len(hit)/max(1,len(terms)):.1f}%), >=5 rows: {n5}")
# span-level share of C1 spans whose term appears in E3 train
for bk in ("C0", "C1"):
    terms = test_terms_from_rows(bench[bk])
    hit = find_terms_in_texts(terms, (text_of(r) for r in pools["e3v1_train_curated"]))
    tot = seen = 0
    for r in bench[bk]:
        for seg, en in parse_tagged(r.get("text_tagged") or ""):
            if en:
                k = dup_key(seg)
                if k:
                    tot += 1
                    seen += k in hit
    print(f"SPANS {bk}: {tot} spans, {seen} ({100*seen/max(1,tot):.1f}%) are terms present in E3v1 train text")
