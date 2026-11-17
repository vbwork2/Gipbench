"""Score locked G0 by dataset, recording setting and channel with paired WER intervals."""

import json
from collections import Counter

import yaml

from gipbench.bench import evaluate, registry
from gipbench.core.manifest import read_jsonl


def main():
    results = registry.P.path("results_root")
    bm = registry.load_benchmark("general_v1")
    rows = read_jsonl(registry.expand(bm["manifest"]))
    if error := registry.check_frozen(bm, rows):
        raise SystemExit(error)
    out = results / "phase4" / "g0_recording"
    out.mkdir(parents=True, exist_ok=True)
    counts = Counter((row["dataset"], row["slice"], row["channel"]) for row in rows)
    (out / "recording_counts.json").write_text(json.dumps([
        {"dataset": dataset, "slice": setting, "channel": channel, "utterances": n}
        for (dataset, setting, channel), n in sorted(counts.items())
    ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Diagnostic grouping leaves the locked manifest and the official score unchanged.
    config = out / "general_v1_recording.yaml"
    config.write_text(yaml.safe_dump({
        **{key: value for key, value in bm.items() if not key.startswith("_")},
        "group_by": ["dataset", "slice", "channel"],
    }, allow_unicode=True), encoding="utf-8")
    source = results / "general_asr" / "general_v1"
    gipformer = source / "gip15_mbs4_open" / "hyps.jsonl"
    for system in ("gip15_mbs4_open", "viegem_greedy"):
        hyps = source / system / "hyps.jsonl"
        hrows, _ = evaluate.load_hyps(hyps)
        check = evaluate.check_hyps([row["id"] for row in rows], [row["id"] for row in hrows],
                                    [row.get("hyp", "") for row in hrows])
        if not check["ok"]:
            raise SystemExit(f"Hypotheses do not match locked G0: {hyps}")
        paired = [] if hyps == gipformer else ["--baseline-hyps", str(gipformer)]
        evaluate.main(["--benchmark", str(config), "--hyps", str(hyps),
                       "--out", str(out / system), *paired])


if __name__ == "__main__":
    main()
