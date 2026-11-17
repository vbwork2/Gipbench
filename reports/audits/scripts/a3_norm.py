"""Audit 3: normalization sensitivity on real G0/C1 results (numbers, ngàn/nghìn, tư/bốn), insertion attribution,
CS-WER boundary effects, and alternative metrics (MER-like, English-only, PIER-like). Read-only."""
import json, sys, re, collections
sys.path.insert(0, "src")
from gipbench.core.normprofiles import canon_v1
from gipbench.core.textnorm import basic_tokens, normalize_ref, normalize_hyp, Catalog, plain_text, parse_tagged
from gipbench.core.align import align, attribute

R = "results/general_asr"


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def hyps(p):
    return {h["id"]: h.get("hyp", "") for h in rd(p)}


G0 = rd("data/benchmarks/general_v1/manifest.jsonl")
C1 = rd("data/benchmarks/cs_ggroup_v1/manifest.jsonl")
C0 = rd("data/benchmarks/code_switch_v1/manifest.jsonl")
E0 = rd("data/benchmarks/external_v1/manifest.jsonl")
DIG = re.compile(r"\d")
for name, rows in (("G0", G0), ("C1", C1), ("C0", C0), ("E0", E0)):
    by = collections.Counter(r["dataset"] for r in rows if DIG.search(plain_text(r["text_tagged"])))
    print(f"{name}: rows with digits in reference: {sum(by.values())}/{len(rows)}", dict(by))

cat = Catalog()
VAR = [("bốn", "tư"), ("nghìn", "ngàn"), ("linh", "lẻ"), ("năm", "lăm"), ("một", "mốt")]


def errs(rows, H, pre=canon_v1, mapping=None):
    tot_e = tot_n = 0
    for r in rows:
        ref = [t.tok for t in normalize_ref(pre(r["text_tagged"]), cat)[0]]
        hy = normalize_hyp(pre(H.get(r["id"], "")), cat)
        if mapping:
            ref = [mapping.get(t, t) for t in ref]
            hy = [mapping.get(t, t) for t in hy]
        ops = align(ref, hy)
        tot_e += sum(o[0] != "C" for o in ops)
        tot_n += len(ref)
    return 100 * tot_e / tot_n


MAP = {"tư": "bốn", "ngàn": "nghìn", "lẻ": "linh"}
for sysn in ("gip15_mbs4_open", "viegem_greedy"):
    H = hyps(f"{R}/general_v1/{sysn}/hyps.jsonl")
    dig_rows = [r for r in G0 if DIG.search(r["text_tagged"])]
    a = errs(dig_rows, H)
    b = errs(dig_rows, H, mapping=MAP)
    print(f"G0 {sysn}: digit-rows WER strict_v1 {a:.2f} -> with number-variant merge {b:.2f}  (n_rows={len(dig_rows)})")
    byds = collections.defaultdict(list)
    for r in G0:
        byds[r["dataset"]].append(r)
    for ds, rr in sorted(byds.items()):
        print(f"   {ds:20s} WER {errs(rr, H):.2f} -> {errs(rr, H, mapping=MAP):.2f}")
    # hyp token counts of variants
    c = collections.Counter(t for h in H.values() for t in basic_tokens(h) if t in ("tư", "ngàn", "lẻ", "bốn", "nghìn", "linh"))
    print("   hyp variant counts:", dict(c))
    print("   empty hyps:", sum(1 for h in H.values() if not h.strip()))
    dg = sum(1 for h in H.values() if DIG.search(h))
    print("   hyps containing digits:", dg)

# ---- CS-WER decomposition on C1 (strict, no catalog)
def decomp(rows, H):
    S = collections.Counter()
    ins_en_from_vi_side = 0
    n_en = 0
    tok_en_only_ref = tok_en_only_err = 0
    bnd_err = 0
    for r in rows:
        rt, spans = normalize_ref(canon_v1(r["text_tagged"]), cat)
        ref = [t.tok for t in rt]
        sp = [t.span for t in rt]
        hy = normalize_hyp(canon_v1(H.get(r["id"], "")), cat)
        ops = align(ref, hy)
        e, s_, d_, i_ = attribute(ops, sp)
        n_en += sum(x >= 0 for x in sp)
        last = None
        for k, (t, ri, hj) in enumerate(ops):
            if t != "I":
                if t != "C":
                    S[("en" if sp[ri] >= 0 else "vi") + "_" + t] += 1
                last = ri
            else:
                left = sp[last] if last is not None else -1
                nxt = next((ops[j][1] for j in range(k + 1, len(ops)) if ops[j][1] is not None), None)
                right = sp[nxt] if nxt is not None else -1
                if left >= 0:
                    S["en_I_left"] += 1
                elif right >= 0:
                    S["en_I_rightonly"] += 1  # insertion between a VI token (left) and EN token (right): charged to EN
                else:
                    S["vi_I"] += 1
    return n_en, S


for sysn in ("gip15_mbs4_open", "viegem_greedy"):
    H = hyps(f"{R}/cs_ggroup_v1/{sysn}/hyps.jsonl")
    n_en, S = decomp(C1, H)
    en_err = sum(v for k, v in S.items() if k.startswith("en_"))
    print(f"C1 {sysn}: n_en={n_en} EN errors={en_err} -> CS-WER {100*en_err/n_en:.2f}; breakdown {dict(S)}")
    print(f"   CS-WER if right-only insertions charged to VI: {100*(en_err-S['en_I_rightonly'])/n_en:.2f}; "
          f"if ALL insertions excluded (S+D only): {100*(S['en_S']+S['en_D'])/n_en:.2f}")

# ---- VI-only utterances inside G0 (no EN spans) vs C1 subset: retention view
gg = [r for r in G0 if r["dataset"].startswith("ggroup")]
c1ids = {r["id"].replace("c1_", "g0_", 1) for r in C1}
print("C1 id sample", C1[0]["id"], "G0 gg sample", gg[0]["id"])
c1sha = {r["audio_sha256"] for r in C1}
vi_only = [r for r in gg if r["audio_sha256"] not in c1sha]
print(f"G-Group rows without EN (Vietnamese-only control inside G0): {len(vi_only)} / {len(gg)}")
for sysn in ("gip15_mbs4_open", "viegem_greedy"):
    H = hyps(f"{R}/general_v1/{sysn}/hyps.jsonl")
    cs_rows = [r for r in gg if r["audio_sha256"] in c1sha]
    print(f"   {sysn}: WER G-Group VI-only {errs(vi_only, H):.2f} vs G-Group CS rows {errs(cs_rows, H):.2f}")
