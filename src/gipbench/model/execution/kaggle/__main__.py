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
from gipbench.model.execution.kaggle.bundle import TINY_SHA256, check_approval, portable_name, prepare_v02

EXPERIMENTS = ("smoke", "synthetic", "icefall", "V02")
TERMINAL = {"remote_complete", "failed", "interrupted", "completed"}
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
    if dataset and state.get("dataset_version"):
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
    identity = [args.experiment, args.owner, commit, sha256_file(template), args.internet]
    extra = ""
    if args.experiment == "icefall":
        if not args.internet:
            raise ValueError("The synthetic Icefall probe downloads pinned public dependencies")
        helper = template.with_name("icefall_probe.py")
        requirements = helper.with_name("icefall-t4-requirements.txt")
        identity.extend([sha256_file(helper), sha256_file(requirements)])
        extra = helper.read_text(encoding="utf-8")
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
        if args.experiment == "V02":
            dataset_dir = run / "dataset"
            dataset_dir.mkdir(exist_ok=True)
            bundle = prepare_v02(dataset_dir, commit, args.runtime)
            dataset_slug = f"gipbench-v02-{bundle['sha256'][:20]}"
            state.update(bundle=bundle, dataset=f"{args.owner}/{dataset_slug}")
            write_json(dataset_dir / "dataset-metadata.json", {
                "id": state["dataset"], "title": dataset_slug, "licenses": [{"name": "other"}],
            })
            write_json(dataset_dir / "bundle-identity.json", bundle)
        notebook = run / "notebook"
        write_json(notebook / "kernel-metadata.json", notebook_metadata(state))
        config = {k: state[k] for k in ("run", "experiment", "commit", "internet", "bundle") if k in state}
        if args.experiment == "icefall":
            config["requirements"] = requirements.read_text(encoding="utf-8")
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
        if not re.search(r"\b404\b", str(exc)) or state.get("dataset_create_attempted"):
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
        state["transfer_approval_sha256"] = check_approval(args.approval, state["bundle"])
        if not state["runtime_lock"]:
            raise ValueError("Prepare V02 with the approved dependency lock after the GPU smoke")
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
    if state["state"] in {"prepared", "completed"}:
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
        record = registry().parent / "v02/tiny_low_lr"
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
