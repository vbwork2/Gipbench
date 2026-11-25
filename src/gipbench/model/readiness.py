"""Resolve VietNote experiment evidence and validate a V03 training bundle."""

import hashlib
import json
import math
import re
import shutil
from pathlib import Path

import yaml

from gipbench.core import leakgate as LG
from gipbench.core import release as R
from gipbench.core.decontam import is_video_unit
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.normprofiles import code_sha256
from gipbench.core.paths import REPO
from gipbench.model.recipe import experiment as X
from gipbench.model.vietnote import LOCK

REGISTRY = REPO / "configs/research/vietnote.yaml"
GROUP_KINDS = {"replay_real": "vi", "cs_real": "cs"}


def read_yaml(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def registry():
    return read_yaml(REGISTRY)


def resolve_manifest(ref):
    rel = R.resolve(ref)
    if rel.manifest is None or "manifest.jsonl" not in rel.meta.get("files", {}):
        raise R.ReleaseError(f"{ref} lacks a locked manifest")
    return rel


def bindings(config, seed):
    return {
        "seed": seed,
        "config_sha256": hashlib.sha256(json.dumps({
            "sources": [R.sha256(p) for p in config["_sources"]],
            "overrides": config["_overrides"],
            "weights": X.groups(config),
        }, sort_keys=True).encode()).hexdigest(),
        "release_sha256": {
            ref: resolve_manifest(ref).meta["files"]["manifest.jsonl"]["sha256"] for ref in
            dict.fromkeys([*X.input_refs(config), *config["dev"].values(), *LG.guard_refs(config), *R.locked_bench_refs()])
        },
        "source_registry_sha256": R.sha256(REPO / registry()["sources"]),
        "catalog_sha256": R.sha256(R.resolve(config["catalog"]).catalog),
    }


def eligible_release(release, sources, kind):
    flags = release.meta.get("flags") or {}
    approval = flags.get("vietnote_v03") or {}
    source = sources.get(approval.get("source"), {})
    rights = source.get("recording_rights") or {}
    evidence = rights.get("evidence")
    review = source.get("release_reviews", {}).get(release.ref, {})
    if not review.get("path") or release.manifest is None:
        return False
    path = X.abspath(review["path"])
    if not path.is_file() or R.sha256(path) != review.get("sha256"):
        return False
    audit = read_json(path)
    if (audit.get("release") != release.ref or audit.get("manifest_sha256") != R.sha256(release.manifest)
            or audit.get("source_revision") != source.get("revision") or not audit.get("reviewer")):
        return False
    decisions = audit.get("decisions", {})
    rows = read_jsonl(release.manifest)
    if not rows or len(rows) != len(decisions):
        return False
    for row in rows:
        decision = decisions.get(row["id"], {})
        if (decision.get("decision") != "accept" or decision.get("language") != kind
                or decision.get("audio_sha256") != row.get("audio_sha256")
                or not eligible_row(row, kind)
                or decision.get("speaker") != row["speaker"] or decision.get("domain") != row.get("domain")
                or not row.get("domain") or decision.get("speech_type") != row.get("speech_type")
                or row.get("speech_type") not in {"natural", "read"}):
            return False
    return bool(evidence and rights.get("training_allowed") is True and rights.get("license")
                and R.sha256(X.abspath(evidence)) == rights.get("sha256") and approval.get("kind") == kind
                and source.get("revision") and approval.get("source_revision") == source.get("revision")
                and flags.get("audio_screened") is True and flags.get("acoustic_checked") is True
                and approval.get("transcripts_verified") is True and approval.get("speakers_verified") is True)


def eligible_row(row, kind):
    text = row.get("text_tagged") or ""
    spans = re.findall(r"<en>([^<>]+)</en>", text)
    outside = re.sub(r"<en>[^<>]+</en>", "", text)
    return bool(row.get("audio_sha256") and row.get("speaker") and row.get("source") and row.get("split") == "train"
                and not row.get("synthetic") and not row.get("augmentation")
                and row.get("language_category") in ({"vi"} if kind == "vi" else {"mixed", "cs"})
                and any(c.isalpha() for c in outside) and "<" not in outside and ">" not in outside
                and bool(spans) == (kind == "cs"))


def inventory(config, sources):
    available = {kind: 0.0 for kind in GROUP_KINDS.values()}
    eligible = dict(available)
    sources_hours = {}
    problems, seen, seen_eligible = [], set(), set()
    for group in X.groups(config):
        kind = GROUP_KINDS.get(group["group"])
        if kind is None or group["weight"] <= 0:
            problems.append(f"Unexpected or disabled V03 mixture group: {group['group']}")
            continue
        for ref in group["refs"]:
            try:
                release = resolve_manifest(ref)
                approved = eligible_release(release, sources, kind)
                rows = read_jsonl(release.manifest)
            except (R.ReleaseError, OSError, ValueError, TypeError) as exc:
                problems.append(f"{ref}: {exc}")
                continue
            if not approved:
                problems.append(f"{ref}: missing V03 rights or data-quality evidence")
            approval = (release.meta.get("flags") or {}).get("vietnote_v03") or {}
            summary = {"kind": kind, "available_hours": 0.0, "eligible_hours": 0.0,
                       "rows": len(rows), "source": approval.get("source")}
            for row in rows:
                audio = Path(row.get("_audio_path", ""))
                sec = float(row.get("duration", 0))
                if not audio.is_file() or not math.isfinite(sec) or sec <= 0:
                    continue
                identity = row.get("audio_sha256") or str(audio.resolve())
                if identity not in seen:
                    seen.add(identity)
                    summary["available_hours"] += sec / 3600
                if approved and eligible_row(row, kind) and identity not in seen_eligible and R.sha256(audio) == identity:
                    seen_eligible.add(identity)
                    summary["eligible_hours"] += sec / 3600
            sources_hours[ref] = summary
            available[kind] += summary["available_hours"]
            eligible[kind] += summary["eligible_hours"]
    return {"available": available, "eligible": eligible, "releases": sources_hours, "problems": problems}


def independent_cs(config, spec):
    ref = config["dev"]["cs"]
    rel = resolve_manifest(ref)
    flags = rel.meta.get("flags") or {}
    rows = read_jsonl(rel.manifest)
    domains = {str(row.get("domain") or "").strip().lower() for row in rows} - {"", "unk", "unknown", "other"}
    speakers = {row.get("speaker_cluster") for row in rows if row.get("speaker_cluster")
                and not is_video_unit(row["speaker_cluster"])}
    exclusion = {*X.input_refs(config), *R.locked_bench_refs(), *LG.guard_refs(config)}
    audit = flags.get("independent_cs_audit") or {}
    required_hashes = {name: resolve_manifest(name).meta["files"]["manifest.jsonl"]["sha256"] for name in exclusion}
    if (ref != spec["ref"] or flags.get("independent") is not True or flags.get("interim")
            or any((audit.get("exclusion_sha256") or {}).get(name) != digest for name, digest in required_hashes.items())
            or audit.get("audio_verified") is not True
            or audit.get("review_manifest_sha256") != manifest_sha256(rows)
            or len(rows) < spec["min_rows"] or len(domains) < spec["min_domains"]
            or len(speakers) < spec["min_verified_speakers"]
            or any(not str(row.get("label_source", "")).startswith("human:") for row in rows)):
        raise ValueError(f"{ref}: independent CS provenance or registered thresholds are incomplete")
    return {"rows": len(rows), "domains": len(domains), "verified_speakers": len(speakers)}


def check_memory(result, protocol, train, expected_gpu, environment=None, available_vram=None):
    limit = train["max_duration"]
    precision = "fp16" if train["fp16"] else "fp32"
    if result.get("selected_limit") != limit or precision != protocol["precision"]:
        raise ValueError("Missing selected batch configuration or precision")
    if not result.get("gpu") or (expected_gpu is not None and expected_gpu != result["gpu"]):
        raise ValueError("Batch evidence used a different GPU")
    if not result.get("environment") or (environment is not None and result["environment"] != environment):
        raise ValueError("Batch evidence used a different runtime environment")
    for kind in ("pooled", "long"):
        cases = [c for c in result["cases"] if c["kind"] == kind and c["limit"] == limit]
        if len(cases) != 1:
            raise ValueError(f"Missing {kind} batch probe at {limit} seconds")
        case, = cases
        steps = case.get("steps", [])
        if (case.get("passed") is not True or case.get("precision") != precision
                or len(steps) != protocol["steps_per_case"]
                or not (0 < case["peak_allocated_bytes"] <= case["peak_reserved_bytes"]
                        <= case["gpu_free_before_bytes"] - protocol["min_free_fraction"] * case["gpu_total_bytes"])
                or not (0 < case["gpu_free_before_bytes"] <= case["gpu_total_bytes"])
                or any(not (limit * protocol["min_batch_fraction"] <= step["padded_seconds"] <= limit
                            and 0 < step["audio_seconds"] <= step["padded_seconds"] + 1e-6) for step in steps)
                or (kind == "long" and case["longest_clip"] < protocol["long_clip_min_seconds"])):
            raise ValueError(f"Unsupported {kind} batch at {limit} seconds")
        if available_vram is not None:
            free, total = available_vram
            if total != case["gpu_total_bytes"] or free < case["peak_reserved_bytes"] + protocol["min_free_fraction"] * total:
                raise ValueError(f"Current CUDA memory cannot support the verified {kind} batch")


def model_validation(config, research, expected_gpu=None, environment=None, available_vram=None):
    base = X.abspath(config["base_model"])
    lock = read_json(LOCK)
    checkpoint = R.sha256(base / "pytorch/pretrained.pt")
    spec = next(f for f in lock["model_files"] if f["path"] == "pytorch/pretrained.pt")
    if checkpoint != spec["sha256"]:
        raise ValueError("Checkpoint differs from the VietNote revision lock")
    evidence = {key: REPO / val for key, val in research["evidence"].items()}
    if environment and environment.get("backend") == "kaggle":
        evidence["memory"] = evidence["memory_kaggle"]
    elif environment and environment.get("colab"):
        evidence["memory"] = evidence["memory_colab"]
    protocol = R.sha256(REPO / research["validation_config"])
    scorer = code_sha256([REPO / "src/gipbench" / path for path in research["scorer_files"]])
    identity = {"model_revision": lock["revision"], "checkpoint_input_sha256": checkpoint}
    details, problems = {}, {}
    for name in ("parity", "tensors", "tiny", "tiny_followup", "memory"):
        try:
            result = read_json(evidence[name])
            details[name] = result
            if any(result.get(key) != val for key, val in identity.items()):
                raise ValueError("stale model or checkpoint identity")
            if name != "tensors" and result.get("config_sha256") != protocol:
                raise ValueError("stale validation protocol")
            if name != "memory" and result.get("passed") is not True:
                raise ValueError("registered criteria failed")
            if name == "parity":
                comparisons = {"offline_fp32", "offline_int8", "streaming_fp32", "streaming_int8"}
                if set(result.get("comparisons", {})) != comparisons:
                    raise ValueError("incomplete export comparison")
                if result.get("scorer_code_sha256") != scorer or result.get("catalog_sha256") != R.sha256(R.resolve(config["catalog"]).catalog):
                    raise ValueError("stale scorer")
                for filename, digest in result["outputs"].items():
                    if R.sha256(evidence[name].parent / filename) != digest:
                        raise ValueError(f"stale parity output {filename}")
            elif name == "tensors":
                if result["manifest_sha256"] != details.get("parity", {}).get("dev_manifest_sha256"):
                    raise ValueError("FP32 tensor and transcript evidence used different subsets")
            elif name in {"tiny", "tiny_followup"}:
                if name == "tiny_followup" and result.get("lr_protocol_sha256") != R.sha256(REPO / research["tiny_followup_config"]):
                    raise ValueError("stale tiny-set follow-up protocol")
                for filename, field in (("steps.jsonl", "gradients_sha256"), ("tiny.pt", "checkpoint_sha256")):
                    if R.sha256(evidence[name].parent / filename) != result[field]:
                        raise ValueError(f"stale V02 artifact {filename}")
                train = read_jsonl(evidence[name].parent / "train.jsonl")
                if result["train_manifest_sha256"] != manifest_sha256(train):
                    raise ValueError("stale tiny-set manifest")
                if result.get("scorer_code_sha256") != scorer or result.get("catalog_sha256") != R.sha256(R.resolve(config["catalog"]).catalog):
                    raise ValueError("stale scorer")
            elif name == "memory":
                memory_path = REPO / research["memory_config"]
                memory = read_yaml(memory_path)
                if (result.get("memory_config_sha256") != R.sha256(memory_path)
                        or result.get("pool_manifest_sha256") != R.sha256(resolve_manifest(memory["pool"]).manifest)
                        or result.get("long_pool_manifest_sha256") != R.sha256(resolve_manifest(memory["long_pool"]).manifest)):
                    raise ValueError("stale memory protocol")
                check_memory(result, memory, config["train"], expected_gpu, environment, available_vram)
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems[name] = str(exc)
    if {"parity", "tensors"} <= details.keys():
        try:
            exporter = evidence["checkpoint_export"]
            meta = read_json(exporter)
            for key in ("parity", "tensors"):
                if details[key]["export_metadata_sha256"] != R.sha256(exporter):
                    raise ValueError("export evidence is stale")
            for folder, files in ((exporter.parent, meta["files"]), (base, details["parity"]["official_files"])):
                for filename, digest in files.items():
                    if R.sha256(folder / filename) != digest:
                        raise ValueError(f"stale ONNX artifact {filename}")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            problems["export"] = str(exc)
    if "tiny" not in problems or "tiny_followup" not in problems:
        problems.pop("tiny", None)
        problems.pop("tiny_followup", None)
    return details, problems


def data_ready(config, research):
    sources = read_yaml(REPO / research["sources"])["sources"]
    corpus = inventory(config, sources)
    targets = config["research"]["target_unique_hours"]
    issues = [*corpus["problems"],
              *[f"{kind}: {corpus['eligible'][kind]:.3f}/{hours:g} eligible unique hours"
                for kind, hours in targets.items() if corpus["eligible"][kind] < hours]]
    try:
        independent_cs(config, research["independent_cs"])
    except (R.ReleaseError, OSError, ValueError) as exc:
        issues.append(str(exc))
    return corpus, issues


def cutset_hours(config, cuts):
    from lhotse import CutSet
    from vnasr.text import normalize_text
    from gipbench.core.textnorm import plain_text

    expected = {}
    for group in X.groups(config):
        kind = GROUP_KINDS[group["group"]]
        for ref in group["refs"]:
            for row in read_jsonl(resolve_manifest(ref).manifest):
                identifier = f"{group['group']}-{row['id']}"
                if identifier in expected:
                    raise ValueError(f"Duplicate train recording ID: {identifier}")
                if eligible_row(row, kind):
                    expected[identifier] = (kind, row)
    valid = {f"dev_{kind}-{row['id']}": (kind, row) for kind, ref in config["dev"].items()
             for row in read_jsonl(resolve_manifest(ref).manifest)}
    seen, seen_dev = set(), set()
    hours = {"vi": 0.0, "cs": 0.0}
    for filename, rows, identities in (("train.jsonl.gz", expected, seen), ("valid.jsonl.gz", valid, seen_dev)):
        kinds = set()
        recording_ids = set()
        for cut in CutSet.from_file(cuts / filename):
            kind, row = rows[cut.recording_id]
            recording_ids.add(cut.recording_id)
            path = Path(row["_audio_path"]).resolve()
            digest = R.sha256(path)
            recording = cut.recording
            if (digest in identities or digest != row.get("audio_sha256") or cut.start != 0
                    or abs(cut.duration - recording.duration) > 0.001
                    or abs(cut.duration - float(row["duration"])) > 0.05 or recording.sampling_rate != 16000
                    or any(type(transform).__name__ != "Resample" for transform in (recording.transforms or []))
                    or len(recording.sources) != 1 or recording.sources[0].type != "file"
                    or Path(recording.sources[0].source).resolve() != path or len(cut.supervisions) != 1
                    or cut.supervisions[0].text != normalize_text(plain_text(row["text_tagged"]))):
                raise ValueError(f"Cut does not match verified source audio and label: {cut.id}")
            identities.add(digest)
            kinds.add(kind)
            if filename == "train.jsonl.gz":
                hours[kind] += cut.duration / 3600
        if kinds != {"vi", "cs"}:
            raise ValueError(f"{filename} must contain both VI and CS")
        if filename == "valid.jsonl.gz":
            require_dev_coverage(valid, recording_ids)
    if seen & seen_dev:
        raise ValueError("Audio overlap between train and dev CutSets")
    check_mixture(config, hours)
    return hours


def require_dev_coverage(expected, observed):
    missing = set(expected) - observed
    if missing:
        raise ValueError(f"Development CutSet missing {len(missing)} rows: {sorted(missing)[:5]}")


def check_mixture(config, hours):
    groups = X.groups(config)
    total_weight = sum(g["weight"] for g in groups)
    total_hours = sum(hours.values())
    tolerance = config["features"]["max_duration"] / 3600
    for group in groups:
        kind = GROUP_KINDS[group["group"]]
        if abs(hours[kind] - total_hours * group["weight"] / total_weight) > tolerance:
            raise ValueError(f"CutSet hours differ from registered mixture for {kind}")


def build_bundle(config, seed, staged, out):
    from lhotse import CutSet

    research = registry()
    _, issues = data_ready(config, research)
    if issues:
        raise ValueError("\n".join(issues))
    provenance = bindings(config, seed)
    leak = LG.gate_config(config)
    if (leak["mode"] != "fail" or leak["audio_sha"] != "hash" or not leak["require_audio_sha"]
            or not leak["near"] or not leak["check_dev"] or not leak["reserved_terms"] or leak["exemptions"]):
        raise ValueError("V03 requires the strict contamination gate without exemptions")
    refs = list(dict.fromkeys([*X.input_refs(config), *config["dev"].values()]))
    report = LG.run_gate(config["name"], LG.guard_refs(config),
                         [(ref, resolve_manifest(ref).manifest) for ref in refs], mode="fail", audio_sha="hash",
                         require_audio_sha=True, near=True, dev_refs=list(config["dev"].values()),
                         reserved_terms=X.abspath(leak["reserved_terms"]))
    if not report["passed"]:
        raise ValueError(f"Contamination gate failed with {report['critical']} critical findings")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(staged / "train_mix_cuts.jsonl.gz", out / "train.jsonl.gz")
    dev = [cut for kind in sorted(config["dev"]) for cut in CutSet.from_file(staged / f"dev_{kind}_cuts_raw.jsonl.gz")]
    CutSet.from_cuts(dev).to_file(out / "valid.jsonl.gz")
    hours = cutset_hours(config, out)
    if any(hours[kind] < minimum for kind, minimum in config["research"]["target_unique_hours"].items()):
        raise ValueError(f"Insufficient unique hours after CutSet filtering: {hours}")
    LG.write_report(report, out)
    provenance.update(hours=hours, cutsets={name: R.sha256(out / name) for name in ("train.jsonl.gz", "valid.jsonl.gz")},
                      contamination_sha256=R.sha256(out / "leak_report.json"),
                      reserved_terms_sha256=R.sha256(X.abspath(leak["reserved_terms"])))
    (out / "bundle.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")


def verify_bundle(config, seed, cuts):
    current = bindings(config, seed)
    bundle = read_json(cuts / "bundle.json")
    if any(bundle.get(key) != value for key, value in current.items()):
        raise ValueError("Stale bundle config, releases, catalog or source registry")
    digests = {name: R.sha256(cuts / name) for name in ("train.jsonl.gz", "valid.jsonl.gz")}
    if bundle.get("cutsets") != digests or R.sha256(cuts / "leak_report.json") != bundle["contamination_sha256"]:
        raise ValueError("Stale CutSets or contamination report")
    reserved = LG.gate_config(config)["reserved_terms"]
    if R.sha256(X.abspath(reserved)) != bundle["reserved_terms_sha256"]:
        raise ValueError("Stale reserved-term list")
    if read_json(cuts / "leak_report.json").get("passed") is not True:
        raise ValueError("Contamination audit did not pass")
    hours = cutset_hours(config, cuts)
    if hours != bundle["hours"] or any(hours[k] < v for k, v in config["research"]["target_unique_hours"].items()):
        raise ValueError("Insufficient verified unique training hours")
    return bundle


def require_training(config, seed, cuts, gpu):
    import os
    import platform
    import torch
    research = registry()
    if Path(config["_path"]).resolve() != (REPO / research["experiment_config"]).resolve():
        raise ValueError("Select the registered V03 training configuration")
    _, issues = data_ready(config, research)
    environment = {"platform": platform.system(), "kernel": platform.release(), "torch": str(torch.__version__),
                   "cuda": torch.version.cuda, "colab": bool(os.environ.get("COLAB_RELEASE_TAG"))}
    if Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir():
        environment["backend"] = "kaggle"
    _, model_issues = model_validation(config, research, expected_gpu=gpu, environment=environment,
                                       available_vram=torch.cuda.mem_get_info())
    issues.extend(f"{name}: {reason}" for name, reason in model_issues.items())
    if issues:
        raise ValueError("\n".join(issues))
    return verify_bundle(config, seed, cuts)
