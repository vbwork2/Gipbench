"""Generate a bounded research-only ZeroTTS pilot from the frozen training text."""

import argparse
import csv
import hashlib
import json
import os
import shutil
from importlib.metadata import version
from pathlib import Path

import numpy as np
import yaml

from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.paths import REPO
from gipbench.core.textnorm import plain_text

CONFIG = REPO / "configs/research/vietnote_tts.yaml"


def selected_rows(config):
    checked = {}
    for key in ("script", "negative_control"):
        path = REPO / config[key]
        if R.sha256(path) != config[f"{key}_sha256"]:
            raise ValueError(f"The frozen {key} changed")
        checked[key] = read_jsonl(path)
    ids = {row["id"] for row in checked["negative_control"]}
    rows = [row for row in checked["script"] if row["id"] in ids]
    order = lambda row: hashlib.sha256(f"{config['seed']}:{row['id']}".encode()).hexdigest()
    selected = [row for medical in (False, True)
                for row in sorted((r for r in rows if (r["domain"] == "medical") == medical), key=order)
                [:config["samples_per_stratum"]]]
    if len(selected) != config["samples_per_stratum"] * 2:
        raise ValueError("The frozen control does not cover both strata")
    return selected, {row["id"]: row for row in checked["negative_control"]}


def score_pilot(directory, config, primary="zerotts"):
    from gipbench.bench import evaluate, transcribe
    from gipbench.dataset.synth import verify

    catalog = R.resolve(config["catalog"]).catalog
    script = {row["id"]: row for row in selected_rows(config)[0]}
    systems = {"gipformer": "gip15_mbs4_open", "vietnote": "vietnote78_offline_int8"}
    datasets = {primary: "pilot.jsonl", "espeak": "espeak_matched.jsonl"}
    scores = {}
    review = []
    for engine, filename in datasets.items():
        manifest = directory / filename
        rows = read_jsonl(manifest)
        if set(script) != {row["id"] for row in rows} or len(rows) != len(script):
            raise ValueError(f"Unexpected sample IDs in {manifest}")
        for row in rows:
            if (R.sha256(row["_audio_path"]) != row["audio_sha256"]
                    or row["text_tagged"] != script[row["id"]]["text_tagged"]):
                raise ValueError(f"Pilot text or audio changed for {row['id']}")
        digest = manifest_sha256(rows)
        hypotheses = []
        for name, system in systems.items():
            out = directory / "asr" / engine / name
            model = REPO / "configs/bench/asr_models" / f"{system}.yaml"
            hyps = out / "hyps.jsonl"
            identity = out / "run_meta.json"
            meta = json.loads(identity.read_text(encoding="utf-8")) if identity.is_file() else {}
            model_files = meta.get("extra", {}).get("model_files", {})
            if not (hyps.is_file() and meta.get("manifest_sha256") == digest
                    and meta.get("model_config_sha256") == R.sha256(model)
                    and meta.get("hyps_sha256") == R.sha256(hyps) and model_files
                    and all(Path(file["path"]).is_file() and R.sha256(file["path"]) == file["sha256"]
                            for file in model_files.values())):
                transcribe.main(["--model", str(model), "--manifest", str(manifest),
                                 "--out", str(out), "--no-resume", "--batch", "1"])
            hypotheses.append(f"{name}={hyps}")
            evaluate.main(["--manifest", str(manifest), "--catalog", str(catalog), "--hyps", str(hyps),
                           "--out", str(out / "score"), "--seed", "0", "--iters", "1000"])
            result = json.loads((out / "score/summary.json").read_text(encoding="utf-8"))
            score = result["profiles"]["gipbench_strict_v1"]["overall"]
            scores[f"{engine}/{name}"] = {key: score[key] for key in ("wer", "cs_wer", "term_recall")}
        triage = directory / f"{engine}_triage.jsonl"
        triage_input = directory / f"{engine}_triage_input.jsonl"
        voice = config["engines"]["zerotts"]["voice"]
        write_jsonl(triage_input, (dict(row, tts=row.get("tts") or {
            "backend": engine, "voice": voice, "tts_text": plain_text(row["text_tagged"]),
            "readings": script[row["id"]]["readings"], "category": script[row["id"]]["category"],
        }) for row in rows))
        verify.main(["--manifest", str(triage_input), "--hyps", *hypotheses, "--catalog", str(catalog),
                     "--out", str(triage), "--min-agree", "1"])
        kept = {row["id"] for row in read_jsonl(triage)}
        review.extend((engine, row, row["id"] in kept) for row in rows)
    (directory / "asr_metrics.json").write_text(json.dumps(scores, indent=2) + "\n", encoding="utf-8")
    review.sort(key=lambda item: hashlib.sha256(item[1]["audio_sha256"].encode()).hexdigest())
    listening = directory / "blind_audio"
    listening.mkdir(exist_ok=False)
    dims = config["human_review"]["dimensions"]
    with ((directory / "blind_review.csv").open("x", encoding="utf-8-sig", newline="") as sheet,
          (directory / "blind_key.csv").open("x", encoding="utf-8-sig", newline="") as mapping):
        writer, key = csv.writer(sheet), csv.writer(mapping)
        writer.writerow(["id", "audio", "text_tagged", *dims, "decision"])
        key.writerow(["id", "engine", "script_id", "asr_triage_kept"])
        for number, (engine, row, kept) in enumerate(review, 1):
            identifier = f"sample-{number:02d}"
            path = listening / f"{identifier}.wav"
            shutil.copyfile(row["_audio_path"], path)
            writer.writerow([identifier, f"blind_audio/{path.name}", row["text_tagged"], *[""] * (len(dims) + 1)])
            key.writerow([identifier, engine, row["id"], kept])


def load_engine(engine, cfg, revision):
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(cfg["model"], revision=revision or cfg.get("revision"), token=False)
    if not info.sha or not info.card_data or str(info.card_data.license).lower() != cfg["weights_license"].lower():
        raise ValueError("Missing or changed TTS model license")
    cache = REPO / "models/vietnote78/tts_cache"
    if engine == "vieneu":
        from vieneu import Vieneu

        if version("vieneu") != cfg["runtime"].split("==", 1)[1].split(",", 1)[0]:
            raise ValueError(f"Install the registered runtime {cfg['runtime']}")
        if shutil.disk_usage(cache.parent).free < 4 * 2**30:
            raise OSError("Insufficient disk space for the pinned ONNX pilot")
        root = Path(snapshot_download(cfg["model"], revision=info.sha, cache_dir=cache, token=False,
                                      allow_patterns=["onnx_update/*", "README.md"], max_workers=2))
        tts = Vieneu(mode="v3turbo", backend="onnx", device="cpu", backbone_repo=str(root),
                     onnx_dir=str(root / "onnx_update"), threads=4)
        return info.sha, tts.sample_rate, tts.infer

    from zerotts import ZeroTTS, load_voice, resolve_model_dir

    if f"zerotts=={version('zerotts')}" != cfg["runtime"]:
        raise ValueError(f"Install the registered runtime {cfg['runtime']}")
    root = resolve_model_dir(cfg["model"], revision=info.sha, cache_dir=str(cache))
    voice = load_voice(root / "voices", cfg["voice"])
    tts = ZeroTTS(root, intra_op_num_threads=4)
    return info.sha, tts.sample_rate, lambda text: tts.synthesize(text, voice=voice, max_frames=250)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision", help="Use the recorded revision to reproduce a previous pilot")
    parser.add_argument("--engine", default="zerotts", choices=["zerotts", "vieneu"])
    parser.add_argument("--score", action="store_true", help="Score existing audio and prepare blind human review")
    args = parser.parse_args()
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if args.score:
        score_pilot(args.out, config, args.engine)
        return
    import soundfile as sf

    cfg = config["engines"][args.engine]
    R.forbid_bench_catalog(config["catalog"], "Neural TTS")
    catalog = R.resolve(config["catalog"]).catalog
    rows, negative = selected_rows(config)
    revision, sample_rate, synthesize = load_engine(args.engine, cfg, args.revision)
    args.out.mkdir(parents=True, exist_ok=False)
    audio = args.out / "audio"
    audio.mkdir()
    voice = cfg.get("voice", "default_preset")
    generated = []
    for row in rows:
        np.random.seed(int(hashlib.sha256(f"{config['seed']}:{row['id']}".encode()).hexdigest()[:8], 16))
        wave = synthesize(plain_text(row["text_tagged"]))
        if not wave.size or not np.isfinite(wave).all() or float(np.max(np.abs(wave))) < 0.001:
            raise ValueError(f"Invalid generated audio for {row['id']}")
        dest = audio / f"{row['id']}.wav"
        sf.write(dest, wave, sample_rate, subtype="PCM_16")
        generated.append({"id": row["id"], "audio": f"audio/{dest.name}", "text_tagged": row["text_tagged"],
                          "source": f"custom_tts:{row['id']}", "speaker": f"tts:{args.engine}:{voice}",
                          "dataset": f"custom_{args.engine}_pilot", "split": "train", "duration": wave.size / sample_rate,
                          "domain": row["domain"], "audio_sha256": R.sha256(dest), "synthetic": True,
                          "train_only": True, "label_source": "synthetic:conditioning_text",
                          "quality_status": "pending_human_review", "weights_license": cfg["weights_license"]})
    write_jsonl(args.out / "pilot.jsonl", generated)
    control = [{**negative[row["id"]], "audio": os.path.relpath(negative[row["id"]]["_audio_path"], args.out)}
               for row in generated]
    write_jsonl(args.out / "espeak_matched.jsonl", control)
    review = [{**row, "audio": row["audio"]} for row in [*generated, *control]]
    review.sort(key=lambda row: hashlib.sha256(row["audio_sha256"].encode()).hexdigest())
    dimensions = config["human_review"]["dimensions"]
    with (args.out / "listen.csv").open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "audio", "text_tagged", *dimensions, "decision"])
        for number, row in enumerate(review, 1):
            writer.writerow([f"sample-{number:02d}", row["audio"], row["text_tagged"], *[""] * (len(dimensions) + 1)])
    evidence = {"model": cfg["model"], "revision": revision, "voice": voice, "runtime": cfg["runtime"],
                "weights_license": cfg["weights_license"], "catalog_sha256": R.sha256(catalog),
                "script_sha256": config["script_sha256"], "rows": len(generated),
                "hours": sum(row["duration"] for row in generated) / 3600, "human_approved_rows": 0,
                "status": "pending_asr_and_human_review", "manifest_sha256": R.sha256(args.out / "pilot.jsonl")}
    (args.out / "result.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
