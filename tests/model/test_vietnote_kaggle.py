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


def test_v02_arms_have_distinct_identities(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "registry", lambda: tmp_path)
    monkeypatch.setattr(kaggle, "prepare_v02", lambda d, c, r: {
        "sha256": "a" * 64, "bytes": 10, "members_manifest_sha256": "b" * 64,
        "train_rows": 163, "train_manifest_sha256": kaggle.TINY_SHA256,
        "model_revision": "rev", "icefall_commit": "commit",
    })
    a = kaggle.prepare(argparse.Namespace(owner="my-account", experiment="V02", internet=False, runtime=None, arm="A"))
    b = kaggle.prepare(argparse.Namespace(owner="my-account", experiment="V02", internet=False, runtime=None, arm="B"))
    assert a["run"] != b["run"]
    assert a["arm"] == "A" and a["lr"] == 0.0045
    assert b["arm"] == "B" and b["lr"] == 0.0015


def test_expl_ft_prepare_verify_and_license_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle, "registry", lambda: tmp_path)
    monkeypatch.setattr(kaggle, "doctor", lambda: {"authentication": "verified"})
    monkeypatch.setattr(kaggle, "expl_ft_inputs", lambda commit: (
        {"train.jsonl": b'{"id":"t1"}\n', "valid.jsonl": b'{"id":"v1"}\n'},
        {"license": kaggle.AUTHORIZED_LICENSE, "members_manifest_sha256": "c" * 64,
         "train_rows": 1, "valid_rows": 1,
         "train_manifest_sha256": "d" * 64, "valid_manifest_sha256": "e" * 64,
         "source_hours": {"datahive_vi_curated": 1.0}},
    ))
    state = kaggle.prepare(argparse.Namespace(owner="my-account", experiment="expl-ft", internet=True, runtime=None))
    assert state["experiment"] == "expl-ft"
    assert state["bundle"]["license"] == kaggle.AUTHORIZED_LICENSE
    run = tmp_path / state["run"]
    assert "DataHive VI" in kaggle.read_json(run / "dataset/dataset-metadata.json")["description"]

    # Mutating bundle license must be rejected before submit
    bad_state = dict(state, bundle=dict(state["bundle"], license="unresolved"))
    with pytest.raises(ValueError, match="transfer policy"):
        kaggle.submit(run, bad_state, argparse.Namespace(timeout_seconds=1800, accelerator=None, approval=None))

    # verify_result checks expl-ft.json fields and checkpoint sha256
    out = tmp_path / "out"
    out.mkdir()
    ckpt = out / "selected.pt"
    ckpt.write_bytes(b"weights")
    env = out / "environment.json"
    kaggle.write_json(env, {"gpu": "Tesla T4", "cuda_probe_passed": True,
                            "gpu_free_bytes": 90, "gpu_total_bytes": 100})
    rep = out / "expl-ft.json"
    kaggle.write_json(rep, {
        "experiment": "expl-ft",
        "train_manifest_sha256": state["bundle"]["train_manifest_sha256"],
        "valid_manifest_sha256": state["bundle"]["valid_manifest_sha256"],
        "selected_checkpoint_sha256": sha256_file(ckpt),
    })
    files = {}
    for name in ("duration-filter.json", "train.log", "train.jsonl", "valid.jsonl", "runtime-packages.json", "requirements.txt"):
        p = out / name
        p.write_text("ok", encoding="utf-8")
        files[name] = sha256_file(p)
    files["environment.json"] = sha256_file(env)
    files["expl-ft.json"] = sha256_file(rep)
    files["selected.pt"] = sha256_file(ckpt)
    kaggle.write_json(out / "result-manifest.json", {
        "run": state["run"], "experiment": "expl-ft", "commit": state["commit"],
        "bundle_sha256": state["bundle"]["sha256"], "files": files,
    })
    assert kaggle.verify_result(out, state)["files"]["selected.pt"] == sha256_file(ckpt)
