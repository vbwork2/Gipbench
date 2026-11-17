"""Audit 1: locked benchmark manifests -> counts, hash reproduction, field coverage, cross-track overlap."""
import json, sys, hashlib, collections, re, unicodedata
from pathlib import Path
sys.path.insert(0, "src")
from gipbench.bench import registry as R
from gipbench.core.textnorm import basic_tokens, plain_text

B = ["general_v1", "code_switch_v1", "cs_ggroup_v1", "external_v1"]
rows = {}
for b in B:
    bm = R.load_benchmark(b)
    rs = [json.loads(l) for l in open(R.expand(bm["manifest"]), encoding="utf-8")]
    rows[b] = rs
    got = R.manifest_sha256(rs)
    print(f"== {b}: {len(rs)} rows, hash_ok={got == bm['manifest_sha256']}")
    by = collections.defaultdict(list)
    for r in rs:
        by[r["dataset"]].append(r)
    for d, rr in sorted(by.items()):
        dur = [r.get("duration") for r in rr]
        nd = sum(1 for x in dur if x)
        hrs = sum(x for x in dur if x) / 3600
        sha = sum(1 for r in rr if r.get("audio_sha256"))
        spk = {r.get("speaker") for r in rr}
        spk_is_id = sum(1 for r in rr if r.get("speaker") in (None, "", r["id"]))
        src = {r.get("source") for r in rr}
        tx = collections.Counter(" ".join(basic_tokens(plain_text(r.get("text_tagged") or ""))) for r in rr)
        rep = sum(c for c in tx.values() if c > 1)
        intdur = sum(1 for x in dur if x and float(x).is_integer())
        print(f"  {d:24s} n={len(rr):5d} dur={nd:5d} h={hrs:6.2f} sha={sha:5d} spk={len(spk):5d} spk=id:{spk_is_id:5d}"
              f" src={len(src):5d} uniq_text={len(tx):5d} rows_with_repeated_text={rep:5d} int_dur={intdur}")

sha = {b: {r["audio_sha256"]: r for r in rs if r.get("audio_sha256")} for b, rs in rows.items()}
for i, a in enumerate(B):
    for b in B[i + 1:]:
        inter = set(sha[a]) & set(sha[b])
        if inter:
            pairs = collections.Counter((sha[a][s]["dataset"], sha[b][s]["dataset"]) for s in inter)
            print(f"AUDIO overlap {a} x {b}: {len(inter)} {dict(pairs)}")


def nt(r):
    return " ".join(basic_tokens(plain_text(r.get("text_tagged") or "")))


txt = {b: collections.defaultdict(list) for b in B}
for b, rs in rows.items():
    for r in rs:
        txt[b][nt(r)].append(r["dataset"])
for i, a in enumerate(B):
    for b in B[i + 1:]:
        inter = {t for t in set(txt[a]) & set(txt[b]) if len(t.split()) >= 4}
        if inter:
            pairs = collections.Counter((txt[a][t][0], txt[b][t][0]) for t in inter)
            print(f"TEXT(>=4 tok) overlap {a} x {b}: {len(inter)} {dict(pairs.most_common(8))}")
e = collections.defaultdict(set)
for r in rows["external_v1"]:
    t = nt(r)
    if len(t.split()) >= 4:
        e[t].add(r["dataset"])
c = collections.Counter(tuple(sorted(v)) for v in e.values() if len(v) > 1)
print("E0 cross-dataset text dup:", dict(c))
cv27 = {r["upstream_id"] for r in rows["general_v1"] if r["dataset"] == "commonvoice_vi_27"}
cv17 = {r["upstream_id"] for r in rows["external_v1"] if r["dataset"] == "cv17_vi"}
print("CV17 vs CV27 upstream_id overlap:", len(cv27 & cv17), "of", len(cv17))
for b in B:
    keys = collections.Counter(k for r in rows[b] for k in r)
    print(b, "fields:", dict(keys))
