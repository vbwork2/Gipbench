"""Run VietNote tiny-set learning and separate FP32 CUDA batch probes."""

import argparse
import csv
import json
import os
import random
import platform
from pathlib import Path

import torch
import yaml
from lhotse import Fbank, FbankConfig
from torch.nn.utils.rnn import pad_sequence

from gipbench.bench.evaluate import score_rows, scorer_code_sha256, summarize
from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.normprofiles import get_profile
from gipbench.core.textnorm import Catalog, plain_text
from gipbench.infer.adapters.base import load_audio
from gipbench.model.vietnote import LOCK
from vietnote_torch_check import load_model

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/model/experiments/vietnote_validation.yaml"
MEMORY_PROTOCOL = PROTOCOL.with_name("vietnote_memory.yaml")
CATALOG = "catalog/train-v1"


def reviewed_rows(spec):
    release = R.resolve(spec["pool"])
    review = ROOT / spec["review"]
    historical = ROOT / "data/work/recovery/p3/default/tiny.jsonl"
    if (R.sha256(release.manifest) != spec["pool_sha256"] or R.sha256(review) != spec["review_sha256"]
            or R.sha256(historical) != spec["historical_tiny_sha256"]):
        raise ValueError("The registered pool, review, or historical tiny-set manifest changed")
    pool = {row["id"]: row for row in read_jsonl(release.manifest)}
    expected = {row["id"]: row for row in read_jsonl(historical)}
    rows = []
    with review.open(encoding="utf-8-sig", newline="") as handle:
        for decision in csv.DictReader(handle):
            if decision["decision"] not in {"ok", "fix"} or decision["id"] not in expected:
                continue
            row = dict(pool[decision["id"]])
            text = decision["text_tagged_fixed"] if decision["decision"] == "fix" else decision["text_tagged"]
            old = expected[row["id"]]
            if text != old["text_tagged"] or "<en>" not in text or row["audio_sha256"] != old["audio_sha256"]:
                raise ValueError(f"Review or audio identity changed for {row['id']}")
            row.update(text_tagged=text, label_source=old["label_source"], train_only=True)
            rows.append(row)
    if len(rows) != spec["rows"] or {row["id"] for row in rows} != set(expected):
        raise ValueError("Incomplete or duplicated tiny-set review")
    return sorted(rows, key=lambda row: row["id"])


def prepare(rows):
    from vnasr.text import normalize_text

    extractor = Fbank(FbankConfig(num_mel_bins=80))
    features = {}
    for row in rows:
        path = Path(row["_audio_path"]).resolve()
        if R.sha256(path) != row["audio_sha256"]:
            raise ValueError(f"Audio digest mismatch for {row['id']}")
        wave, sample_rate = load_audio(path, target_sr=16000)
        features[row["id"]] = (torch.as_tensor(extractor.extract(wave, sample_rate)),
                               normalize_text(plain_text(row["text_tagged"])))
    return features


def make_batch(rows, features):
    frames, texts = zip(*(features[row["id"]] for row in rows))
    return {"inputs": pad_sequence(frames, batch_first=True),
            "supervisions": {"num_frames": torch.tensor([len(frame) for frame in frames]), "text": list(texts)}}


def batches(rows, limit, rng):
    # Group similarly sized examples so the limit reflects the padded batch.
    ordered = sorted(rows, key=lambda row: float(row["duration"]))
    groups, batch = [], []
    for row in ordered:
        if batch and (len(batch) + 1) * float(row["duration"]) > limit:
            groups.append(batch)
            batch = []
        batch.append(row)
    if batch:
        groups.append(batch)
    rng.shuffle(groups)
    return groups


def run_step(model, sp, params, optimizer, scheduler, batch):
    from finetune import compute_loss, get_adjusted_batch_count, set_batch_count

    set_batch_count(model, get_adjusted_batch_count(params))
    params.batch_idx_train += 1
    loss, stats = compute_loss(params, model, sp, batch, is_training=True)
    if not bool(torch.isfinite(loss)):
        raise ValueError("Non-finite loss")
    loss.backward()
    norms = torch._foreach_norm([p.grad for p in model.parameters() if p.grad is not None])
    grad_norm = torch.linalg.vector_norm(torch.stack(norms)).item()
    if not 0 < grad_norm < float("inf"):
        raise ValueError("Missing, zero or non-finite gradient")
    scheduler.step_batch(params.batch_idx_train)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"step": params.batch_idx_train, "grad_l2": grad_norm, "loss": float(loss.detach()),
            "frames": float(stats["frames"]), "simple_loss": stats["simple_loss"], "pruned_loss": stats["pruned_loss"]}


def setup(base, lr, limit, seed):
    from finetune import Eden, ScaledAdam, get_parameter_groups_with_lrs

    torch.manual_seed(seed)
    random.seed(seed)
    model, sp, params = load_model(base, offline=False)
    model = model.cuda().train()
    params.update(base_lr=lr, max_duration=limit, batch_idx_train=0)
    optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=lr, include_names=True),
                           lr=lr, clipping_scale=2.0)
    return model, sp, params, optimizer, Eden(optimizer, lr_batches=100000, lr_epochs=100)


def measure(model, sp, rows, features, out, catalog):
    from beam_search import greedy_search_batch

    model.eval()
    hyps = {}
    with torch.inference_mode():
        for batch in batches(rows, 32, random.Random(0)):
            tensors = make_batch(batch, features)
            enc, lengths = model.forward_encoder(tensors["inputs"].cuda(), tensors["supervisions"]["num_frames"].cuda())
            for row, tokens in zip(batch, greedy_search_batch(model, enc, lengths), strict=True):
                hyps[row["id"]] = sp.decode(tokens)
    write_jsonl(out, ({"id": row["id"], "hyp": hyps[row["id"]]} for row in rows))
    recs = score_rows(rows, hyps, catalog, get_profile("gipbench_strict_v1"), "text_tagged", "speaker", [])
    report = summarize(recs, 0, 0, with_ci=False)
    model.train()
    return {**{key: report[key] for key in ("utt", "wer", "cs_wer", "term_recall")},
            **{key: sum(rec[key] for rec in recs) for key in ("S", "D", "I")}}


def tiny(base, rows, settings, out):
    features = prepare(rows)
    catalog_path = R.resolve(CATALOG).catalog
    catalog = Catalog.load(str(catalog_path))
    limit = settings["batch_seconds"]
    seed = settings["seed"]
    model, sp, params, opt, sched = setup(base, settings["lr"], limit, seed)
    probe_names = ("encoder_embed.conv.0.weight", "decoder.embedding.weight", "joiner.output_linear.weight")
    probes = {key: tensor.detach().cpu().clone() for key, tensor in model.named_parameters() if key in probe_names}
    if not any(key.startswith("encoder") for key in probes):
        name, tensor = next((key, tensor) for key, tensor in model.named_parameters() if key.startswith("encoder_embed."))
        probes[name] = tensor.detach().cpu().clone()
    initial = measure(model, sp, rows, features, out / "before.jsonl", catalog)
    torch.cuda.reset_peak_memory_stats()
    metrics = [initial]
    with (out / "steps.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(settings["passes"]):
            for batch in batches(rows, limit, random.Random(seed + epoch)):
                result = run_step(model, sp, params, opt, sched, make_batch(batch, features))
                log.write(json.dumps({**result, "ids": [row["id"] for row in batch],
                                      "audio_seconds": sum(row["duration"] for row in batch),
                                      "learning_rates": [g["lr"] for g in opt.param_groups]}) + "\n")
                log.flush()
            metrics.append(measure(model, sp, rows, features, out / f"pass-{epoch + 1}.jsonl", catalog))
            print(f"Pass {epoch + 1}, {params.batch_idx_train} steps: {metrics[-1]}", flush=True)
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    updates = {name: float((tensor.detach().cpu() - probes[name]).abs().max())
               for name, tensor in model.named_parameters() if name in probes}
    checkpoint = out / "tiny.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    del opt, sched
    reloaded = model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"], strict=True)
    final = metrics[-1]
    passed = (
        all(value > 0 for value in updates.values()) and len(updates) == 3
        and final["wer"] <= initial["wer"] * (1 - settings["relative_wer_reduction_min"])
        and final["cs_wer"] <= initial["cs_wer"] * (1 - settings["relative_cs_wer_reduction_min"])
        and final["term_recall"] >= initial["term_recall"] - settings["recall_drop_max_pp"]
        and not reloaded.missing_keys and not reloaded.unexpected_keys
    )
    return {"passed": passed, "steps": params.batch_idx_train, "score_by_pass": metrics, "max_updates": updates,
            "gradients_sha256": R.sha256(out / "steps.jsonl"), "checkpoint_sha256": R.sha256(checkpoint),
            "reloaded": True, "catalog_sha256": R.sha256(catalog_path), "scorer_code_sha256": scorer_code_sha256(),
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved}


def memory(base, cfg, selected_limit):
    settings = yaml.safe_load(MEMORY_PROTOCOL.read_text(encoding="utf-8"))
    release = R.resolve(settings["pool"])
    source = sorted(read_jsonl(release.manifest), key=lambda row: float(row["duration"]), reverse=True)
    long_release = R.resolve(settings["long_pool"])
    long = sorted((row for row in read_jsonl(long_release.manifest)
                   if float(row["duration"]) >= settings["long_clip_min_seconds"]),
                  key=lambda row: float(row["duration"]), reverse=True)
    count = settings["steps_per_case"]
    planned = []
    for limit in settings["cases"]:
        minimum = limit * settings["min_batch_fraction"]
        groups = []
        for batch_size in range(1, limit + 1):
            rows = [row for row in source if minimum <= float(row["duration"]) * batch_size <= limit][:batch_size * count]
            if len(rows) == batch_size * count:
                groups = [rows[i::count] for i in range(count)]
                break
        planned.append(("pooled", limit, groups))
    if selected_limit not in settings["cases"]:
        raise ValueError("Selected batch duration is outside the registered probe cases")
    qualifying = [row for row in long if float(row["duration"]) <= selected_limit
                  and int(selected_limit / float(row["duration"])) * float(row["duration"])
                  >= settings["min_batch_fraction"] * selected_limit]
    long_groups = []
    for index in range(count):
        if not qualifying:
            break
        first = qualifying[index % len(qualifying)]
        size = int(selected_limit / float(first["duration"]))
        fillers = [row for row in source if float(row["duration"]) <= float(first["duration"])]
        long_groups.append([first, *fillers[index * size:index * size + size - 1]])
    planned.append(("long", selected_limit, long_groups))
    results = []
    for kind, limit, groups in planned:
        if not groups:
            results.append({"kind": kind, "limit": limit, "passed": False, "reason": "Insufficient qualifying training clips"})
            continue
        selected = list({row["id"]: row for group in groups for row in group}.values())
        features = prepare(selected)
        props = torch.cuda.get_device_properties(0)
        free, _ = torch.cuda.mem_get_info()
        torch.cuda.reset_peak_memory_stats()
        log = {"kind": kind, "limit": limit, "gpu_total_bytes": props.total_memory, "gpu_free_before_bytes": free,
               "precision": "fp32", "steps": [], "longest_clip": max(float(row["duration"]) for row in selected)}
        try:
            state = setup(base, cfg["tiny"]["lr"], limit, cfg["tiny"]["seed"])
            for group in groups:
                stats = run_step(*state, make_batch(group, features))
                log["steps"].append({**stats, "audio_seconds": sum(float(row["duration"]) for row in group),
                                     "padded_seconds": len(group) * max(float(row["duration"]) for row in group),
                                     "ids": [row["id"] for row in group]})
            log["passed"] = torch.cuda.max_memory_reserved() <= free - settings["min_free_fraction"] * props.total_memory
        except torch.cuda.OutOfMemoryError as exc:
            log.update(passed=False, reason=str(exc))
        finally:
            log.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved())
            state = None
            torch.cuda.empty_cache()
        results.append(log)
        print(json.dumps(log), flush=True)
    environment = {"platform": platform.system(), "kernel": platform.release(), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "colab": bool(os.environ.get("COLAB_RELEASE_TAG"))}
    if Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir():
        environment["backend"] = "kaggle"
    return {"cases": results, "selected_limit": selected_limit,
            "passed": all(case["passed"] for case in results if case["limit"] == selected_limit),
            "environment": environment,
            "pool_manifest_sha256": R.sha256(release.manifest),
            "long_pool_manifest_sha256": R.sha256(long_release.manifest),
            "memory_config_sha256": R.sha256(MEMORY_PROTOCOL)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["tiny", "memory"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-seconds", type=int, help="Selected training batch duration for the memory probe")
    parser.add_argument("--lr-protocol", type=Path, help="Preregistered learning-rate arm for the tiny-set test")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if args.lr_protocol:
        if args.mode != "tiny":
            parser.error("--lr-protocol applies only to tiny")
        arm = yaml.safe_load(args.lr_protocol.read_text(encoding="utf-8"))
        if not 0 < arm["lr"] < cfg["tiny"]["lr"]:
            parser.error("The follow-up tests a lower, positive learning rate")
        cfg["tiny"]["lr"] = arm["lr"]
    base = ROOT / "models/vietnote78"
    args.out.mkdir(parents=True, exist_ok=False)
    if args.mode == "tiny":
        rows = reviewed_rows(cfg["tiny"])
        write_jsonl(args.out / "train.jsonl",
                    ({**row, "audio": os.path.relpath(row["_audio_path"], args.out)} for row in rows))
        result = tiny(base, rows, cfg["tiny"], args.out)
        result.update(train_rows=len(rows), train_hours=sum(float(row["duration"]) for row in rows) / 3600,
                      train_manifest_sha256=manifest_sha256(rows))
    else:
        if args.batch_seconds is None:
            parser.error("Memory validation requires --batch-seconds matching the training configuration")
        result = memory(base, cfg, args.batch_seconds)
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    result.update(config_sha256=R.sha256(PROTOCOL), model_revision=card["revision"],
                  checkpoint_input_sha256=R.sha256(base / "pytorch/pretrained.pt"), torch_version=torch.__version__,
                  gpu=torch.cuda.get_device_name())
    if args.lr_protocol:
        result.update(lr=cfg["tiny"]["lr"], lr_protocol_sha256=R.sha256(args.lr_protocol))
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
