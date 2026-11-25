"""Exercise the official CLI on a fixed engineering corpus."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml
from lhotse import CutSet, Recording, SupervisionSegment
from vnasr.text import normalize_text

from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.release import sha256
from gipbench.core.textnorm import plain_text
from gipbench.model.train.vietnote_icefall import strict_check, train_cmd
from gipbench.model.vietnote import LOCK, load_checkpoint


def finish(out, cfg, card, base, icefall, export, checkpoints=None, bundle_path=None):
    import torch
    from gipbench.bench.evaluate import score_rows, summarize
    from gipbench.core.normprofiles import get_profile
    from gipbench.core.release import resolve
    from gipbench.core.textnorm import Catalog
    from vietnote_torch_check import load_model, offline_decode
    from vietnote_parity import compare, decode

    pilot = checkpoints is None
    if pilot:
        checkpoints = out / "exp"
        bundle_path = out / "bundle.json"
        bundle = json.loads(bundle_path.read_text())
        if any(sha256(out / name) != digest for name, digest in bundle["files"].items()):
            raise ValueError("Pilot bundle has changed")
    rows = read_jsonl(out / "valid.jsonl")
    for row in rows:
        if sha256(row["_audio_path"]) != row["audio_sha256"]:
            raise ValueError(f"Stale development audio: {row['id']}")
    catalog = Catalog.load(str(resolve("catalog/train-v1").catalog))
    model, sp, _ = load_model(base, offline=True)
    torch.set_num_threads(2)
    scores = []
    for epoch in range(cfg["train"]["num_epochs"] + 1):
        checkpoint = checkpoints / f"epoch-{epoch}.pt" if epoch else base / "pytorch/pretrained.pt"
        state = load_checkpoint(checkpoint)
        model.load_state_dict(state["model"], strict=True)
        hyps_path = out / f"epoch-{epoch}-dev.jsonl"
        offline_decode(model, sp, rows, hyps_path)
        hyps = {r["id"]: r["hyp"] for r in read_jsonl(hyps_path)}
        if set(hyps) != {r["id"] for r in rows}:
            raise ValueError("Incomplete pilot hypotheses")
        recs = score_rows(rows, hyps, catalog, get_profile("gipbench_strict_v1"), "text_tagged", "speaker", [])
        scores.append(dict(epoch=epoch, checkpoint_sha256=sha256(checkpoint), steps=state.get("batch_idx_train", 0),
                           hyps_sha256=sha256(hyps_path), **summarize(recs, 0, 0, with_ci=False)))
        del state
    del model
    selected = min(scores[1:], key=lambda s: (s["wer"], s["epoch"]))
    retention = cfg["retention_stop_vi_wer_delta_pp"] if pilot else cfg["research"]["acceptance"]["vi_wer_delta_max_pp"]
    selection = dict(scores=scores, epoch=selected["epoch"], checkpoint_sha256=selected["checkpoint_sha256"],
                     retention_passed=selected["vi_only_wer"] <= scores[0]["vi_only_wer"] + retention)
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    checkpoint = checkpoints / f"epoch-{selected['epoch']}.pt"
    if not export.exists():
        archive = icefall.parent / f"{card['icefall_commit']}.tar.gz"
        subprocess.run([sys.executable, str(Path(__file__).with_name("vietnote_reexport.py")), "--checkpoint",
                        str(checkpoint), "--icefall", str(archive), "--out", str(export)], check=True)
    meta = json.loads((export / "export.json").read_text())
    if (meta["checkpoint_sha256"] != selected["checkpoint_sha256"]
            or sha256(export / "model.pt") != meta["model_state_file_sha256"]
            or any(sha256(export / name) != digest for name, digest in meta["files"].items())):
        raise ValueError("Selected checkpoint and exported files differ")
    spec = yaml.safe_load(Path("configs/model/experiments/vietnote_validation.yaml").read_text())["parity"]
    if pilot:
        spec = dict(spec, long_rows_min=1)
    tensor_path = export / "pilot_tensors.json"
    if not tensor_path.exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("vietnote_fp32_tensors.py")),
                        "--icefall", str(icefall), "--export", str(export), "--manifest", str(out / "valid.jsonl"),
                        "--out", str(tensor_path), "--pilot-model", str(export / "model.pt"),
                        "--pairs", str(1 if pilot else 12)], check=True)
    tensors = json.loads(tensor_path.read_text())
    if (tensors["export_metadata_sha256"] != sha256(export / "export.json")
            or tensors["manifest_sha256"] != manifest_sha256(rows)
            or tensors["checkpoint_input_sha256"] != meta["model_state_file_sha256"]):
        raise ValueError("Stale pilot tensor comparison")
    results = {}
    for mode in ("offline", "streaming"):
        fp32, int8 = [decode(mode, p, export / mode, rows, out / f"{mode}-{p}.jsonl") for p in ("fp32", "int8")]
        behavior = compare(fp32, int8, spec, "int8", catalog, rows)
        results[mode] = dict(tensors=tensors["measurements"][mode], fp32_vs_int8=behavior,
                             passed=tensors["passed"] and behavior["passed"])
    result = dict(selection=selection, parity=results, bundle_sha256=sha256(bundle_path),
                  scope="engineering_pilot" if pilot else "single_seed_engineering_checks",
                  export_sha256=sha256(export / "export.json"),
                  passed=selection["retention_passed"] and all(v["passed"] for v in results.values()))
    (out / "result.json").write_text(json.dumps(result, indent=2))


def locked_rows(path, digest):
    rows = read_jsonl(path)
    if manifest_sha256(rows) != digest:
        raise ValueError(f"Stale pilot source: {path}")
    return sorted(rows, key=lambda row: (row["duration"], row["id"]))


def prepare(out, cfg, validation):
    train = locked_rows(cfg["train_manifest"], cfg["train_manifest_sha256"])[:cfg["train_rows"]]
    sources = {r["source"] for r in train}
    audio = {r["audio_sha256"] for r in train}
    dev = []
    for item in validation["parity"]["sources"]:
        rows = [r for r in locked_rows(item["path"], item["sha256"])
                if r["source"] not in sources and r["audio_sha256"] not in audio
                and r["label_source"].startswith("human:")]
        count = cfg["dev_rows_per_kind"]
        if len(rows) < count:
            raise ValueError("Insufficient source-disjoint development rows")
        dev.extend(rows[:count - 1] + rows[-1:])
    for name, rows in (("train", train), ("valid", dev)):
        cuts = []
        for row in rows:
            path = Path(row["_audio_path"]).resolve()
            if sha256(path) != row["audio_sha256"]:
                raise ValueError(f"Changed recording: {path}")
            rec = Recording.from_file(path, recording_id=row["id"]).resample(16000)
            cut = rec.to_cut()
            cut.supervisions = [SupervisionSegment(id=row["id"], recording_id=rec.id,
                                start=0, duration=cut.duration, text=normalize_text(plain_text(row["text_tagged"])))]
            cuts.append(cut)
            row["audio"] = os.path.relpath(path, out)
        CutSet.from_cuts(cuts).to_file(out / f"{name}.jsonl.gz")
        write_jsonl(out / f"{name}.jsonl", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--finish", action="store_true", help="Evaluate and export a completed pilot")
    parser.add_argument("--export", type=Path, help="Verify or create this export directory")
    parser.add_argument("--experiment", type=Path, help="Evaluate a completed registered V03 training run")
    parser.add_argument("--cuts", type=Path)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    proto = Path("configs/model/experiments/vietnote_cli_pilot.yaml")
    cfg = yaml.safe_load(proto.read_text())
    card = json.loads(LOCK.read_text())
    base = Path("models/vietnote78").resolve()
    icefall = base / "upstream/icefall" / f"icefall-{card['icefall_commit']}"
    recipe = Path("data/work/vietnote/recipe").resolve()
    out = args.out.resolve()
    export = args.export or out / "export"
    if args.experiment:
        from gipbench.model.recipe import experiment
        from gipbench.model.readiness import registry, resolve_manifest, verify_bundle
        if (args.experiment.resolve() != Path(registry()["experiment_config"]).resolve()
                or not args.cuts or not args.checkpoints or args.seed is None):
            parser.error("Post-training evaluation requires the registered experiment, bundle, seed and checkpoints")
        cfg = experiment.load(args.experiment)
        verify_bundle(cfg, args.seed, args.cuts)
        out.mkdir(parents=True, exist_ok=False)
        dev = []
        for kind, ref in cfg["dev"].items():
            for row in read_jsonl(resolve_manifest(ref).manifest):
                dev.append(dict(row, id=f"dev_{kind}-{row['id']}", audio=os.path.relpath(row["_audio_path"], out)))
        write_jsonl(out / "valid.jsonl", dev)
        finish(out, cfg, card, base, icefall, export, args.checkpoints, args.cuts / "bundle.json")
        return
    if args.finish:
        if json.loads((out / "bundle.json").read_text())["protocol_sha256"] != sha256(proto):
            raise ValueError("Pilot protocol changed")
        finish(out, cfg, card, base, icefall, export)
        return
    out.mkdir(parents=True, exist_ok=False)
    prepare(out, cfg, yaml.safe_load(proto.with_name("vietnote_validation.yaml").read_text()))
    strict_check(card, recipe, icefall, base)
    cmd = train_cmd(cfg, card, base, recipe, out, out / "exp", cfg["seed"], 1, 0)
    files = {p.name: sha256(p) for p in out.glob("*.jsonl*")}
    (out / "bundle.json").write_text(json.dumps(dict(protocol_sha256=sha256(proto), files=files, command=cmd), indent=2))
    with (out / "cli.log").open("x") as log:
        subprocess.run(cmd, cwd=recipe, check=True, stdout=log, stderr=subprocess.STDOUT)
    finish(out, cfg, card, base, icefall, export)


if __name__ == "__main__":
    main()
