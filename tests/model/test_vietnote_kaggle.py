import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from gipbench.core.fileio import sha256_file
from gipbench.model.execution.kaggle import __main__ as kaggle
from gipbench.model.execution.kaggle.bundle import check_approval, portable_name, write_zip


def test_portable_bundle_and_deterministic_order(tmp_path):
    audio = tmp_path / "example.flac"
    audio.write_bytes(b"example")
    a, b = tmp_path / "a.zip", tmp_path / "b.zip"
    write_zip(a, {"portable/train.jsonl": b"{}", "portable/audio/example.flac": audio})
    write_zip(b, {"portable/audio/example.flac": audio, "portable/train.jsonl": b"{}"})
    assert sha256_file(a) == sha256_file(b)
    with ZipFile(a) as archive:
        assert archive.read("portable/audio/example.flac") == b"example"
    for unportable in ("/mnt/d/audio.wav", "D:/audio.wav", r"D:\audio.wav", "../audio.wav", "a/file:stream"):
        with pytest.raises(ValueError, match="Nonportable"):
            portable_name(unportable)


def test_cloud_transfer_must_cover_manifest(tmp_path):
    evidence = tmp_path / "terms.txt"
    evidence.write_text("Permission for this research and Kaggle transfer", encoding="utf-8")
    approval = tmp_path / "approval.json"
    record = dict(experiment="V02", train_manifest_sha256="registered", cloud_transfer_allowed=True,
                  training_allowed=True, license="research permission", evidence=evidence.name,
                  evidence_sha256=sha256_file(evidence))
    kaggle.write_json(approval, record)
    assert check_approval(approval, {"train_manifest_sha256": "registered"}) == sha256_file(approval)
    with pytest.raises(ValueError, match="exact training manifest"):
        check_approval(approval, {"train_manifest_sha256": "changed"})
    evidence.write_text("different evidence", encoding="utf-8")
    with pytest.raises(ValueError, match="does not cover"):
        check_approval(approval, {"train_manifest_sha256": "registered"})


def test_missing_credentials_and_secret_redaction(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "registry", lambda: tmp_path)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "private-test-value")

    def official(*args):
        if args == ("--version",):
            return "Kaggle CLI 2.2.5"
        raise kaggle.KaggleError("Authentication required to call the Kaggle API.")

    monkeypatch.setattr(kaggle, "official", official)
    assert kaggle.doctor()["authentication"] == "missing"
    assert kaggle.read_json(tmp_path / "doctor.json")["cli"] == "2.2.5"
    assert kaggle.redact("private-test-value Bearer other-credential https://example.com/run?token=secret") == (
        "[REDACTED] [REDACTED] https://example.com/run?[REDACTED]"
    )


def test_submit_records_identity_and_blocks_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "registry", lambda: tmp_path)
    monkeypatch.setattr(kaggle, "doctor", lambda: {"authentication": "verified"})
    pushes = []

    def official(*args, **kwargs):
        pushes.append(args)
        return "Kernel version 1 successfully pushed. Please check https://www.kaggle.com/code/my-account"

    monkeypatch.setattr(kaggle, "official", official)
    state = kaggle.prepare(argparse.Namespace(owner="my-account", experiment="smoke", internet=False, runtime=None))
    run = tmp_path / state["run"]
    metadata = kaggle.read_json(run / "notebook/kernel-metadata.json")
    assert (metadata["is_private"], metadata["enable_gpu"], metadata["enable_internet"]) == (True, True, False)
    assert metadata["dataset_sources"] == []
    args = argparse.Namespace(timeout_seconds=300, accelerator=None)
    assert kaggle.submit(run, state, args)["version"] == 1
    with pytest.raises(ValueError, match="already been dispatched"):
        kaggle.submit(run, kaggle.read_json(run / "run.json"), args)
    assert len(pushes) == 1


def test_status_reconciles_interrupted_submission(tmp_path, monkeypatch):
    state = dict(run="gipbench-smoke", kernel="user/gipbench-smoke", state="submitting")

    def official(*args):
        assert args == ("kernels", "status", "user/gipbench-smoke/1")
        return 'user/gipbench-smoke/1 has status "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"'

    monkeypatch.setattr(kaggle, "official", official)
    assert kaggle.status(tmp_path, state)["state"] == "interrupted"
    assert kaggle.read_json(tmp_path / "run.json")["version"] == 1


def test_sync_checks_identity_and_hash_before_recording_completion(tmp_path, monkeypatch):
    state = dict(run="gipbench-smoke", kernel="user/gipbench-smoke", version=1,
                 experiment="smoke", commit="commit", state="running")

    def official(*args, **kwargs):
        assert args[2] == "user/gipbench-smoke/1"
        if args[1] == "status":
            return 'user/gipbench-smoke/1 has status "KernelWorkerStatus.COMPLETE"'
        out = Path(args[4]) / "gipbench-output"
        record = out / "environment.json"
        kaggle.write_json(record, {"gpu": "test GPU", "cuda_probe_passed": True,
                                   "gpu_free_bytes": 90, "gpu_total_bytes": 100})
        kaggle.write_json(out / "result-manifest.json", {
            "run": state["run"], "experiment": "smoke", "commit": "commit", "bundle_sha256": None,
            "files": {"environment.json": sha256_file(record)},
        })

    monkeypatch.setattr(kaggle, "official", official)
    assert kaggle.sync(tmp_path, state)["state"] == "completed"
    (tmp_path / "output/environment.json").write_text(json.dumps({"gpu": "changed"}), encoding="utf-8")
    with pytest.raises(ValueError, match="modified result"):
        kaggle.sync(tmp_path, state)
