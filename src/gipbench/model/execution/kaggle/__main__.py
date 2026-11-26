"""Prepare, submit, and reconcile private VietNote experiments with the official Kaggle CLI."""

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.paths import REPO, path
from gipbench.model.execution.kaggle.bundle import (
    AUTHORIZED_LICENSE, FLEURS_RELEASE, FLEURS_SHA256, TINY_SHA256, check_approval, diag_ft_inputs,
    expl_ft_inputs, fleurs_memory_inputs, fleurs_pilot, portable_name, prepare_v02, tiny_diag_inputs, write_zip,
)

EXPERIMENTS = ("smoke", "synthetic", "icefall", "icefall-pilot", "icefall-memory", "tiny-diag", "diag-ft", "expl-ft", "V02")
FLEURS_EXPERIMENTS = {"icefall-pilot", "icefall-memory"}
DIAG_EXPERIMENTS = {"tiny-diag", "diag-ft", "expl-ft"}
PUBLIC_BUNDLE_EXPERIMENTS = FLEURS_EXPERIMENTS | DIAG_EXPERIMENTS
ARM_LRS = {"A": 0.0045, "B": 0.0015}
PILOT_EXPORT_REQUIREMENT = ("onnxruntime==1.32.0 "
                            "--hash=sha256:02ea12a28c069ea8510093d95a7f6ff8126bfd9fc5c0d3381c03b80fb2b424ba")
TERMINAL = {"remote_complete", "failed", "interrupted", "completed", "rejected"}
SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,49}")
CONFIGURE = "Create an API token at https://www.kaggle.com/settings/api and save it to ~/.kaggle/access_token."


def registry():
    return path("data_root") / "work/vietnote/kaggle"


def write_json(file, value):
    file.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=file.parent, suffix=".tmp")
    pending = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        pending.replace(file)
    finally:
        pending.unlink(missing_ok=True)


def read_json(file):
    return json.loads(file.read_text(encoding="utf-8"))


@contextmanager
def exclusive(file):
    with file.open("x"):
        pass
    try:
        yield
    finally:
        file.unlink()


def redact(value):
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_KEY", "HF_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"\bKGAT_[A-Za-z0-9_-]+|(?i:Bearer)\s+[A-Za-z0-9_.-]+", "[REDACTED]", value)
    return re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[REDACTED]", value)


class KaggleError(RuntimeError):
    pass


def official(*args, timeout=120, sdk_script=None):
    env = dict(os.environ, PYTHONUTF8="1")
    local = REPO / ".venv/kaggle-cli"
    binary = shutil.which("kaggle")
    cmd = [binary] if binary else [sys.executable, "-m", "kaggle"]
    if sdk_script:
        cmd = [sys.executable, "-c", sdk_script]
    if (not binary or sdk_script) and (local / "kaggle").is_dir():
        env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(local), env.get("PYTHONPATH"))))
    config = Path(env.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")
    if not config.is_dir() and "KAGGLE_CONFIG_DIR" not in env and (local / "kaggle").is_dir():
        env["KAGGLE_CONFIG_DIR"] = str(local)
    try:
        result = subprocess.run([*cmd, *args], env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise KaggleError(f"Kaggle {args[0]} timed out. Reconcile the recorded identity before retrying.") from exc
    output = redact(result.stdout + result.stderr)
    if result.returncode:
        raise KaggleError(output.strip()[-2000:])
    return output.strip()


def doctor():
    report = {"cli": None, "authentication": "unverified"}
    try:
        text = official("--version")
        match = re.search(r"Kaggle CLI (\S+)", text)
        if not match:
            raise KaggleError("The official Kaggle CLI version could not be determined")
        report["cli"] = match[1]
        official("datasets", "list", "--mine", "--page-size", "1", "--csv")
        report["authentication"] = "verified"
    except (OSError, KaggleError) as exc:
        if "Authentication required" in str(exc):
            report.update(authentication="missing", action=CONFIGURE)
        else:
            report["error"] = str(exc)
    write_json(registry() / "doctor.json", report)
    return report


def plan(experiment, approval=None, runtime=None):
    state = read_json(registry() / "doctor.json") if (registry() / "doctor.json").is_file() else {}
    blockers = []
    if state.get("authentication") != "verified":
        blockers.append("Kaggle authentication is required.")
    if experiment == "V02":
        try:
            check_approval(approval, {"train_manifest_sha256": TINY_SHA256})
        except (OSError, ValueError, KeyError) as exc:
            blockers.append(str(exc))
        if runtime is None:
            blockers.append("A hash-pinned dependency lock matching the Kaggle GPU smoke is required.")
    return {"experiment": experiment, "blockers": blockers,
            "stage": "low_lr" if experiment == "V02" else "gpu_environment"}


def notebook_metadata(state):
    dataset = state.get("dataset")
    if (dataset and state.get("dataset_version") and state["experiment"] not in PUBLIC_BUNDLE_EXPERIMENTS
            and not state.get("bundle", {}).get("members_manifest_sha256")):
        dataset = f"{dataset}/{state['dataset_version']}"
    return {
        "id": state["kernel"], "title": state["slug"], "code_file": "job.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_internet": state["internet"],
        "dataset_sources": [dataset] if dataset else [],
        "competition_sources": [], "kernel_sources": [], "model_sources": [],
    }


def prepare(args):
    if not SLUG.fullmatch(args.owner):
        raise ValueError("Supply a Kaggle account slug as --owner")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    root = registry()
    root.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("runtime.py")
    arm = getattr(args, "arm", None)
    duration_policy = getattr(args, "duration_policy", None)
    if args.experiment == "V02" and arm is None:
        arm = "B"
    if args.experiment == "tiny-diag" and arm is None:
        arm = "A"
    if args.experiment == "diag-ft" and duration_policy is None:
        duration_policy = "control_20s"
    identity = [args.experiment, args.owner, commit, sha256_file(template), args.internet]
    if args.experiment == "V02" and arm != "B":
        identity.extend(["arm", arm, ARM_LRS[arm]])
    if args.experiment == "tiny-diag":
        identity.extend(["arm", arm, ARM_LRS[arm]])
    if args.experiment == "diag-ft":
        identity.extend(["duration_policy", duration_policy])
    extra = ""
    pilot = None
    memory = None
    diag_bundle = None
    if args.experiment == "icefall" or args.experiment in PUBLIC_BUNDLE_EXPERIMENTS:
        if not args.internet:
            raise ValueError("The Icefall probe downloads pinned public dependencies")
        helper = template.with_name("icefall_probe.py")
        requirements = helper.with_name("icefall-t4-requirements.txt")
        identity.extend([sha256_file(helper), sha256_file(requirements)])
        extra = helper.read_text(encoding="utf-8")
        if args.experiment == "icefall-pilot":
            runner = template.with_name("icefall_pilot.py")
            pilot = fleurs_pilot()
            identity.extend([sha256_file(runner), hashlib.sha256(json.dumps(pilot).encode()).hexdigest(),
                             PILOT_EXPORT_REQUIREMENT])
            extra += "\n" + runner.read_text(encoding="utf-8")
        elif args.experiment == "icefall-memory":
            runner = template.with_name("icefall_memory.py")
            memory = fleurs_memory_inputs(commit)
            identity.extend([sha256_file(runner), memory[1]["members_manifest_sha256"]])
            extra += "\n" + runner.read_text(encoding="utf-8")
        elif args.experiment in DIAG_EXPERIMENTS:
            runner = template.with_name("icefall_diag.py")
            if args.experiment == "tiny-diag":
                diag_bundle = tiny_diag_inputs(commit)
            elif args.experiment == "diag-ft":
                diag_bundle = diag_ft_inputs(commit)
            else:
                diag_bundle = expl_ft_inputs(commit)
            identity.extend([sha256_file(runner), diag_bundle[1]["members_manifest_sha256"]])
            extra += "\n" + runner.read_text(encoding="utf-8")
    if args.runtime:
        identity.extend(sha256_file(args.runtime / name) for name in ("lock.json", "requirements.txt"))
        identity.extend((file.name, sha256_file(file)) for file in sorted((args.runtime / "wheels").glob("*.whl")))
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
    slug = f"gipbench-{args.experiment.lower()}-{key}"
    run = root / slug
    state_file = run / "run.json"
    if state_file.exists():
        return read_json(state_file)
    run.mkdir(exist_ok=True)
    with exclusive(run / "prepare.lock"):
        if state_file.exists():
            return read_json(state_file)
        state = {
            "run": slug, "slug": slug, "experiment": args.experiment, "commit": commit,
            "kernel": f"{args.owner}/{slug}", "internet": args.internet, "state": "prepared",
        }
        if args.experiment in {"V02", "tiny-diag"}:
            state["arm"] = arm
            state["lr"] = ARM_LRS[arm]
        if args.experiment == "diag-ft":
            state["duration_policy"] = duration_policy
        if args.experiment == "V02" or args.experiment in PUBLIC_BUNDLE_EXPERIMENTS:
            dataset_dir = run / "dataset"
            dataset_dir.mkdir(exist_ok=True)
            if pilot:
                archive = dataset_dir / "bundle.zip"
                pilot_payload = json.dumps(pilot).encode()
                write_zip(archive, {"pilot.json": pilot_payload})
                bundle = {"sha256": sha256_file(archive), "bytes": archive.stat().st_size,
                          "release_sha256": pilot["release_sha256"], "license": pilot["license"]}
                state["pilot_payload_sha256"] = hashlib.sha256(pilot_payload).hexdigest()
                license_name = "CC-BY-4.0"
            elif memory:
                archive = dataset_dir / "bundle.zip"
                files, evidence = memory
                write_zip(archive, files)
                bundle = {**evidence, "sha256": sha256_file(archive), "bytes": archive.stat().st_size}
                license_name = "CC-BY-4.0"
            elif diag_bundle:
                archive = dataset_dir / "bundle.zip"
                files, evidence = diag_bundle
                write_zip(archive, files)
                bundle = {**evidence, "sha256": sha256_file(archive), "bytes": archive.stat().st_size}
                license_name = "CC-BY-4.0"
            else:
                bundle = prepare_v02(dataset_dir, commit, args.runtime)
                license_name = "other"
            dataset_slug = f"gipbench-{args.experiment.lower()}-{bundle['sha256'][:20]}"
            state.update(bundle=bundle, dataset=f"{args.owner}/{dataset_slug}")
            dataset_metadata = {
                "id": state["dataset"], "title": dataset_slug, "licenses": [{"name": license_name}],
            }
            if args.experiment in FLEURS_EXPERIMENTS:
                dataset_metadata["description"] = (
                    "Google FLEURS vi_vn train (CC BY 4.0), selected from GipBench's audited v2 release. "
                    "https://huggingface.co/datasets/google/fleurs"
                    + (" Audio converted to 16-bit FLAC." if pilot else "")
                )
            elif args.experiment == "expl-ft":
                dataset_metadata["description"] = (
                    "Authorized CC-BY-4.0 / CC0-1.0 exploratory speech bundle "
                    "(DataHive VI conversations, Google FLEURS vi_vn, Mozilla Common Voice vi, OpenSLR LibriSpeech). "
                    "Audio converted to 16-bit mono 16 kHz FLAC."
                )
            elif args.experiment in DIAG_EXPERIMENTS:
                dataset_metadata["description"] = (
                    "Authorized CC-BY-4.0 / CC0-1.0 speech subset from audited GipBench releases "
                    "(Google FLEURS vi_vn, Mozilla Common Voice vi, OpenSLR LibriSpeech train-clean-100). "
                    "Audio converted to 16-bit mono 16 kHz FLAC."
                )
            write_json(dataset_dir / "dataset-metadata.json", dataset_metadata)
            write_json(dataset_dir / "bundle-identity.json", bundle)
        notebook = run / "notebook"
        write_json(notebook / "kernel-metadata.json", notebook_metadata(state))
        config = {k: state[k] for k in ("run", "experiment", "commit", "internet", "bundle",
                                        "pilot_payload_sha256", "arm", "lr", "duration_policy")
                  if k in state}
        if args.experiment == "icefall" or args.experiment in PUBLIC_BUNDLE_EXPERIMENTS:
            config["requirements"] = requirements.read_text(encoding="utf-8")
            if pilot:
                config["requirements"] = config["requirements"].rstrip() + "\n" + PILOT_EXPORT_REQUIREMENT + "\n"
        script = template.read_text(encoding="utf-8") + "\n" + extra + f"\nmain(json.loads({json.dumps(json.dumps(config))}))\n"
        write_json(notebook / "job.ipynb", {
            "nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"}},
            "cells": [{"id": "gipbench", "cell_type": "code", "execution_count": None,
                       "metadata": {}, "outputs": [], "source": script.splitlines(keepends=True)}],
        })
        state["notebook_sha256"] = sha256_file(notebook / "job.ipynb")
        state["runtime_lock"] = bool(args.runtime)
        write_json(state_file, state)
        return state


def dataset_status(identifier):
    raw = official("datasets", "status", identifier, "--format", "json")
    match = re.search(r"\{[^{}]+\}", raw)
    if not match:
        raise KaggleError("Kaggle did not return dataset version metadata")
    return json.loads(match[0])


def verify_private_dataset(identifier, version):
    snippet = """
import sys
from kaggle import api
from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest

request = ApiGetDatasetRequest()
request.owner_slug, request.dataset_slug = sys.argv[1].split("/")
with api.build_kaggle_client() as client:
    dataset = client.datasets.dataset_api_client.get_dataset(request)
if dataset.is_private is not True or dataset.current_version_number != int(sys.argv[2]):
    raise SystemExit("Dataset visibility or current version does not match the private run")
"""
    official(identifier, str(version), sdk_script=snippet)


def ensure_dataset(run, state):
    dataset = state["dataset"]
    try:
        status = dataset_status(dataset)
    except KaggleError as exc:
        if state.get("dataset_create_attempted"):
            raise
        if not re.search(r"\b404\b", str(exc)):
            if not re.search(r"\b403\b", str(exc)):
                raise
            owned = official("datasets", "list", "--mine", "--search", dataset.split("/", 1)[1], "--csv")
            if dataset.lower() in owned.lower():
                raise
        state["dataset_create_attempted"] = True
        write_json(run / "run.json", state)
        official("datasets", "create", "-p", str(run / "dataset"), "--keep-tabular", timeout=900)
        status = {}
    for delay in (3, 5, 8, 13, 21, 34):
        if status.get("status") == "ready":
            break
        if status.get("status") in {"failed", "error"}:
            raise KaggleError(f"Dataset creation failed for {dataset}")
        time.sleep(delay)
        status = dataset_status(dataset)
    version = status.get("current_version_number")
    if status.get("status") != "ready" or not version:
        raise KaggleError("Kaggle dataset version is not ready")
    verify_private_dataset(dataset, version)
    with tempfile.TemporaryDirectory(dir=run) as tmp:
        official("datasets", "download", f"{dataset}/{version}", "-f", "bundle-identity.json", "-p", tmp)
        if read_json(Path(tmp) / "bundle-identity.json") != state["bundle"]:
            raise ValueError("Remote dataset identity differs from the prepared bundle")
    return version


def submit(run, state, args):
    if state["state"] != "prepared":
        raise ValueError("This run has already been dispatched. Use status to reconcile it.")
    if not args.timeout_seconds or args.timeout_seconds < 60:
        raise ValueError("Provide the approved Kaggle session budget in --timeout-seconds (at least 60)")
    if doctor()["authentication"] != "verified":
        raise ValueError(CONFIGURE)
    notebook = run / "notebook"
    if (sha256_file(notebook / "job.ipynb") != state["notebook_sha256"]
            or read_json(notebook / "kernel-metadata.json") != notebook_metadata(state)):
        raise ValueError("Prepared notebook or private metadata was modified")
    if state.get("bundle"):
        if state["experiment"] == "V02":
            state["transfer_approval_sha256"] = check_approval(args.approval, state["bundle"])
            if not state["runtime_lock"]:
                raise ValueError("Prepare V02 with the approved dependency lock after the GPU smoke")
        elif state["experiment"] in FLEURS_EXPERIMENTS:
            if state["bundle"].get("license") != "CC-BY-4.0":
                raise ValueError("No transfer policy for this dataset bundle")
        elif state["experiment"] in DIAG_EXPERIMENTS:
            if state["bundle"].get("license") != AUTHORIZED_LICENSE:
                raise ValueError("No transfer policy for this diagnostic bundle")
        else:
            raise ValueError("No transfer policy for this dataset bundle")
        if sha256_file(run / "dataset/bundle.zip") != state["bundle"]["sha256"]:
            raise ValueError("Prepared bundle was modified")
        state["dataset_version"] = ensure_dataset(run, state)
        write_json(notebook / "kernel-metadata.json", notebook_metadata(state))
    state.update(state="submitting", timeout_seconds=args.timeout_seconds, accelerator=args.accelerator)
    write_json(run / "run.json", state)
    options = ["--accelerator", args.accelerator] if args.accelerator else []
    receipt = official("kernels", "push", "-p", str(notebook), "--timeout", str(args.timeout_seconds), *options)
    version = re.search(r"Kernel version (\d+) successfully", receipt)
    if not version:
        raise KaggleError("Submission response omitted the version. Use status to reconcile it.")
    state.update(state="submitted", version=int(version[1]), url=f"https://www.kaggle.com/code/{state['kernel']}")
    write_json(run / "run.json", state)
    return state


def status(run, state):
    if state["state"] in {"prepared", "completed", "rejected"}:
        return state
    # A fresh, content-derived slug is pushed at most once from this registry.
    version = state.get("version", 1)
    response = official("kernels", "status", f"{state['kernel']}/{version}")
    match = re.search(r'has status "(?:KernelWorkerStatus\.)?(\w+)"', response)
    states = {"queued": "submitted", "running": "running", "complete": "remote_complete", "error": "failed",
              "cancel_requested": "running", "cancel_acknowledged": "interrupted", "new_script": "submitted"}
    if not match or match[1].lower() not in states:
        raise KaggleError("Kaggle returned an unknown notebook status")
    state.update(state=states[match[1].lower()], version=version,
                 url=f"https://www.kaggle.com/code/{state['kernel']}")
    write_json(run / "run.json", state)
    return state


def verify_result(result, state):
    metadata = read_json(result / "result-manifest.json")
    expected = {"run": state["run"], "experiment": state["experiment"], "commit": state["commit"],
                "bundle_sha256": (state.get("bundle") or {}).get("sha256")}
    if any(metadata.get(key) != value for key, value in expected.items()) or not metadata.get("files"):
        raise ValueError("The remote result belongs to a different execution")
    for name, digest in metadata["files"].items():
        file = result / portable_name(name)
        if not file.resolve().is_relative_to(result.resolve()) or not file.is_file() or sha256_file(file) != digest:
            raise ValueError(f"Missing or modified result artifact: {name}")
    if "environment.json" not in metadata["files"]:
        raise ValueError("The result is missing verified GPU evidence")
    environment = read_json(result / "environment.json")
    if (environment.get("cuda_probe_passed") is not True or not environment.get("gpu")
            or not 0 < environment.get("gpu_free_bytes", 0) <= environment.get("gpu_total_bytes", 0)):
        raise ValueError("The output does not prove a successful CUDA probe")
    if state["experiment"] == "V02" and not {"result.json", "steps.jsonl", "tiny.pt", "train.jsonl"} <= metadata["files"].keys():
        raise ValueError("The lower-LR result is incomplete")
    if state["experiment"] == "synthetic":
        if "probe.json" not in metadata["files"] or state.get("bundle") or environment["gpu"] != "Tesla T4":
            raise ValueError("Missing synthetic T4 evidence")
        probe = read_json(result / "probe.json")
        if (probe["source"] != "synthetic_chirp" or probe["features"] != [100, 80] or probe["device"] != "cuda:0"
                or not all(math.isfinite(probe[k]) and probe[k] > 0 for k in ("loss", "gradient_norm"))):
            raise ValueError("The synthetic feature or RNN-T check failed")
    if state["experiment"] == "icefall":
        required = {"epoch-1.pt", "icefall.json", "runtime-packages.json", "requirements.txt", "train.log"}
        if state.get("bundle") or environment["gpu"] != "Tesla T4" or not required <= metadata["files"].keys():
            raise ValueError("Incomplete synthetic Icefall artifacts")
        probe = read_json(result / "icefall.json")
        if (probe["source"] != "synthetic_chirp" or probe["train_cuts"] != 1 or probe["steps"] != 1
                or not all(probe[key] is True for key in ("strict_reload", "optimizer_reload", "scheduler_reload"))
                or not probe["updated_parameter"]
                or probe["checkpoint_sha256"] != metadata["files"]["epoch-1.pt"]
                or not math.isfinite(probe["train_loss"])):
            raise ValueError("Invalid one-step or checkpoint evidence")
    if state["experiment"] == "icefall-pilot":
        files = metadata["files"]
        required = {"pilot.json", "epoch-2.pt", "runtime-packages.json", "requirements.txt",
                    "train.log", "resume.log", "uninterrupted.log", "export.log", "train.jsonl", "valid.jsonl",
                    *(f"{part}.onnx" for part in ("encoder", "decoder", "joiner")), "tokens.txt"}
        if environment["gpu"] != "Tesla T4" or not required <= files.keys():
            raise ValueError("Incomplete FLEURS engineering pilot artifacts")
        pilot = read_json(result / "pilot.json")
        if (pilot["release"] != "pool/vietnote_fleurs_vi@v2"
                or pilot["release_sha256"] != state["bundle"]["release_sha256"] or pilot["train_cuts"] != 4
                or pilot["valid_cuts"] != 32 or pilot["checkpoint_sha256"] != files["epoch-2.pt"]
                or pilot["pilot_payload_sha256"] != state["pilot_payload_sha256"]
                or pilot["steps_resumed"] <= pilot["steps_before_resume"]
                or not all(math.isfinite(value) for value in pilot["validation_losses"])):
            raise ValueError("Invalid FLEURS engineering pilot evidence")
    if state["experiment"] == "icefall-memory":
        required = {"engineering-memory.json", "batches.json", "train.jsonl", "runtime-packages.json", "requirements.txt"}
        if environment["gpu"] != "Tesla T4" or not required <= metadata["files"].keys():
            raise ValueError("Incomplete FP32 memory probe artifacts")
        probe = read_json(result / "engineering-memory.json")
        if (probe["release"] != FLEURS_RELEASE or probe["release_sha256"] != FLEURS_SHA256
                or probe["scope"] != "engineering_only" or probe["v03_gate_passed"] is not False
                or probe["limit"] != 200 or probe["precision"] != "fp32" or probe["longest_clip"] < 25
                or probe["planned_steps"] != state["bundle"]["planned_steps"]
                or not 0 <= probe["peak_allocated_bytes"] <= probe["peak_reserved_bytes"] <= probe["gpu_total_bytes"]
                or (probe["passed"] and len(probe["steps"]) != probe["planned_steps"])
                or not all(math.isfinite(step["loss"]) and math.isfinite(step["grad_l2"]) and step["grad_l2"] > 0
                           and 160 <= step["padded_seconds"] <= 200 for step in probe["steps"])):
            raise ValueError("Invalid FP32 memory probe evidence")
    if state["experiment"] == "tiny-diag":
        required = {"result.json", "steps.jsonl", "train.jsonl", "runtime-packages.json", "requirements.txt"}
        if environment["gpu"] != "Tesla T4" or not required <= metadata["files"].keys():
            raise ValueError("Incomplete tiny-diag artifacts")
        diag = read_json(result / "result.json")
        if (diag.get("experiment") != "tiny-diag" or diag.get("not_v02") is not True
                or diag.get("arm") != state.get("arm") or diag.get("train_rows") != state["bundle"]["train_rows"]
                or diag.get("train_manifest_sha256") != state["bundle"]["train_manifest_sha256"]
                or diag.get("steps") != 489):
            raise ValueError("Invalid tiny-diag evidence")
    if state["experiment"] in {"diag-ft", "expl-ft"}:
        report_name = f"{state['experiment']}.json"
        required = {report_name, "duration-filter.json", "selected.pt", "train.log",
                    "train.jsonl", "valid.jsonl", "runtime-packages.json", "requirements.txt"}
        if environment["gpu"] != "Tesla T4" or not required <= metadata["files"].keys():
            raise ValueError(f"Incomplete {state['experiment']} artifacts")
        diag = read_json(result / report_name)
        if (diag.get("experiment") != state["experiment"]
                or (state["experiment"] == "diag-ft" and diag.get("duration_policy") != state.get("duration_policy"))
                or diag.get("train_manifest_sha256") != state["bundle"]["train_manifest_sha256"]
                or diag.get("valid_manifest_sha256") != state["bundle"]["valid_manifest_sha256"]
                or diag.get("selected_checkpoint_sha256") != metadata["files"]["selected.pt"]):
            raise ValueError(f"Invalid {state['experiment']} evidence")
    return metadata


def sync(run, state):
    if status(run, state)["state"] not in {"remote_complete", "completed"}:
        raise ValueError("The recorded Kaggle version has not completed")
    result = run / "output"
    if not result.exists():
        with tempfile.TemporaryDirectory(dir=run) as tmp:
            identifier = f"{state['kernel']}/{state['version']}"
            official("kernels", "output", identifier, "-p", tmp, "--page-size", "200", timeout=900)
            found = list(Path(tmp).rglob("result-manifest.json"))
            if len(found) != 1:
                raise ValueError("Kaggle output must contain exactly one result manifest")
            verify_result(found[0].parent, state)
            found[0].parent.rename(result)
    verify_result(result, state)
    state.update(state="completed", environment=read_json(result / "environment.json"),
                 result_manifest_sha256=sha256_file(result / "result-manifest.json"))
    if state["experiment"] == "V02":
        measurement = read_json(result / "result.json")
        if (measurement["train_manifest_sha256"] != state["bundle"]["train_manifest_sha256"]
                or measurement.get("train_rows") != state["bundle"]["train_rows"]
                or measurement.get("steps") != 489):
            raise ValueError("The V02 result used a different training manifest")
        state["acceptance_passed"] = measurement["passed"]
        arm_dir = "tiny_kaggle_lr0045" if state.get("arm") == "A" else "tiny_low_lr"
        record = registry().parent / "v02" / arm_dir
        record.mkdir(parents=True, exist_ok=True)
        if not (record / "result.json").exists():
            if ((record / "train.jsonl").exists()
                    and manifest_sha256(read_jsonl(record / "train.jsonl")) != measurement["train_manifest_sha256"]):
                raise ValueError("Existing V02 output directory has a different training manifest")
            with tempfile.TemporaryDirectory(dir=record) as tmp:
                for name in ("train.jsonl", "steps.jsonl", "tiny.pt", "result.json"):
                    target = record / name
                    if target.exists():
                        if name != "train.jsonl" and sha256_file(target) != sha256_file(result / name):
                            raise ValueError(f"Existing V02 artifact has a different digest: {name}")
                        continue
                    staged = Path(tmp) / name
                    try:
                        os.link(result / name, staged)
                    except OSError:
                        shutil.copyfile(result / name, staged)
                    staged.rename(target)
    write_json(run / "run.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("doctor", "report"):
        commands.add_parser(command)
    for command in ("plan", "prepare"):
        current = commands.add_parser(command)
        current.add_argument("--experiment", choices=EXPERIMENTS, required=True)
        current.add_argument("--runtime", type=Path)
        if command == "prepare":
            current.add_argument("--owner", required=True)
            current.add_argument("--internet", action="store_true")
            current.add_argument("--arm", choices=("A", "B"))
            current.add_argument("--duration-policy", choices=("control_20s", "extended_45s"))
        else:
            current.add_argument("--approval", type=Path)
    for command in ("submit", "status", "logs", "sync", "watch"):
        current = commands.add_parser(command)
        current.add_argument("--run", required=True)
        if command == "submit":
            current.add_argument("--approval", type=Path)
            current.add_argument("--timeout-seconds", type=int, required=True)
            current.add_argument("--accelerator")
        if command == "watch":
            current.add_argument("--wait-seconds", type=int, default=300)
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            value = doctor()
        elif args.command == "report":
            value = [read_json(file) for file in sorted(registry().glob("*/run.json"))]
        elif args.command == "plan":
            value = plan(args.experiment, args.approval, args.runtime)
        elif args.command == "prepare":
            value = prepare(args)
        else:
            if not SLUG.fullmatch(args.run):
                raise ValueError("Invalid registry run ID")
            run = registry() / args.run
            state = read_json(run / "run.json")
            if args.command == "submit":
                with exclusive(run / "submit.lock"):
                    value = submit(run, read_json(run / "run.json"), args)
            elif args.command == "status":
                value = status(run, state)
            elif args.command == "sync":
                value = sync(run, state)
            elif args.command == "logs":
                if not state.get("version"):
                    raise ValueError("Reconcile the notebook version with status before reading logs")
                value = {"logs": official("kernels", "logs", f"{state['kernel']}/{state['version']}")}
            else:
                deadline = time.monotonic() + max(0, args.wait_seconds)
                delay = 5
                value = status(run, state)
                while value["state"] not in TERMINAL | {"prepared"} and time.monotonic() < deadline:
                    time.sleep(min(delay, max(0, deadline - time.monotonic())))
                    value = status(run, state)
                    delay = min(60, delay * 2)
                if value["state"] == "remote_complete":
                    value = sync(run, state)
        print(json.dumps(value, ensure_ascii=True, indent=2))
        return int(args.command == "doctor" and value.get("authentication") != "verified")
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": redact(str(exc))}, ensure_ascii=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
