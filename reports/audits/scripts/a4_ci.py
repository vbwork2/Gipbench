"""Audit 4: CI sensitivity to bootstrap unit (utterance vs estimated speaker cluster vs FLEURS sentence)."""
import json, sys, collections
sys.path.insert(0, "src")
from gipbench.core.evalstats import paired_bootstrap, cluster_bootstrap


def rd(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


v0 = rd("data/releases/bench/v0-draft/manifest.jsonl")
clu = {r["id"]: r.get("speaker_cluster") for r in v0 if r.get("speaker_cluster")}
print("v0 rows with speaker_cluster:", len(clu), collections.Counter(c.split(":")[0] for c in clu.values()))
R = "results/general_asr"
for bench, metric in (("cs_ggroup_v1", "cs_wer"), ("general_v1", "wer")):
    a = rd(f"{R}/{bench}/gip15_mbs4_open/utt_scores.jsonl")
    b = rd(f"{R}/{bench}/viegem_greedy/utt_scores.jsonl")
    if bench == "general_v1":
        a = [r for r in a if r["dataset"].startswith("ggroup")]
        b = [r for r in b if r["dataset"].startswith("ggroup")]
    def cl(r):
        raw = r["id"].split("_", 1)[1]  # c1_gg_edu_112998 -> gg_edu_112998
        return clu.get(raw)
    mapped = sum(1 for r in a if cl(r))
    print(f"\n{bench} G-Group rows: {len(a)}, mapped to a speaker_cluster: {mapped}, clusters: {len({cl(r) for r in a if cl(r)})}")
    for unit_name, f in (("utt", lambda r: r["id"]), ("speaker_cluster", lambda r: cl(r) or r["id"])):
        aa = [dict(r, u=f(r)) for r in a]
        bb = [dict(r, u=f(r)) for r in b]
        p = paired_bootstrap(aa, bb, metric, "u", 1000, 0)
        c = cluster_bootstrap(aa, [metric], "u", 1000, 0)[metric]
        print(f"  unit={unit_name:16s} units={p['units']:5d} gip {metric}={p['a']:.2f} CI{tuple(round(x,2) for x in c)} "
              f"width={c[1]-c[0]:.2f} | Δ(VieGem-Gip)={p['delta']:+.2f} CI{tuple(round(x,2) for x in p['ci95'])} width={p['ci95'][1]-p['ci95'][0]:.2f}")

# per-dataset WER CI width utt vs cluster for ggroup_pubadmin in G0 (only Gipformer)
# FLEURS: there are no results yet on E0 -> just report repetition structure
E0 = rd("data/benchmarks/external_v1/manifest.jsonl")
fl = [r for r in E0 if r["dataset"] == "fleurs_vi"]
c = collections.Counter(r["source"] for r in fl)
print("\nFLEURS: utterances", len(fl), "unique sentence ids", len(c), "max repeats", max(c.values()),
      "dist", collections.Counter(c.values()))
