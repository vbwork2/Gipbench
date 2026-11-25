"""Prepare deterministic, minimal VietNote research inputs for Kaggle."""

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath
from zipfile import ZipFile, ZipInfo

import yaml

from gipbench.core import release as R
from gipbench.core.download import verify_file
from gipbench.core.manifest import manifest_sha256
from gipbench.core.paths import REPO
from gipbench.model.vietnote import LOCK

PROTOCOL = Path("configs/model/experiments/vietnote_validation.yaml")
ARM = PROTOCOL.with_name("vietnote_tiny_low_lr.yaml")
CATALOG = "catalog/train-v1"
TINY_SHA256 = "f3ff60b1e7739fcc78dbf1eb99e92342113a3ff6958dfca157b26f4d3831b46a"


def portable_name(name):
    posix, windows = PurePosixPath(name), PureWindowsPath(name)
    if (not name or not posix.parts or posix.is_absolute() or windows.drive or "\\" in name
            or ":" in name or ".." in posix.parts):
        raise ValueError(f"Nonportable bundle path: {name}")
    return posix.as_posix()


def check_approval(approval, spec):
    if not approval:
        raise ValueError("V02 requires documented Kaggle transfer and training rights for the reviewed audio")
    approval = approval.resolve()
    record = json.loads(approval.read_text(encoding="utf-8"))
    evidence = approval.parent / record["evidence"]
    if (record.get("experiment") != "V02" or record.get("train_manifest_sha256") != spec["train_manifest_sha256"]
            or record.get("cloud_transfer_allowed") is not True or record.get("training_allowed") is not True
            or not record.get("license") or R.sha256(evidence) != record.get("evidence_sha256")):
        raise ValueError("The approval does not cover this exact training manifest and Kaggle transfer")
    return R.sha256(approval)


def write_zip(destination, files):
    if len({portable_name(name).casefold() for name in files}) != len(files):
        raise ValueError("Bundle contains conflicting file names")
    with ZipFile(destination, "w") as archive:
        for name, source in sorted(files.items()):
            info = ZipInfo(portable_name(name))
            if isinstance(source, bytes):
                archive.writestr(info, source)
            else:
                info.file_size = source.stat().st_size
                with source.open("rb") as reader, archive.open(info, "w") as writer:
                    shutil.copyfileobj(reader, writer, 1024**2)


def tiny_rows():
    sys.path.insert(0, str(REPO / "scripts/diagnostics"))
    try:
        from vietnote_train_validation import reviewed_rows
        cfg = yaml.safe_load((REPO / PROTOCOL).read_text(encoding="utf-8"))["tiny"]
        rows = reviewed_rows(cfg)
    finally:
        sys.path.pop(0)
    if manifest_sha256(rows) != TINY_SHA256 or len(rows) != cfg["rows"]:
        raise ValueError("The reviewed training rows differ from the registered V02 arm")
    return rows


def prepare_v02(directory, commit, runtime=None):
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    base = REPO / "models/vietnote78"
    raw = subprocess.check_output(
        ["git", "archive", "--format=zip", commit, "src", "scripts/diagnostics/vietnote_train_validation.py",
         "scripts/diagnostics/vietnote_torch_check.py", str(PROTOCOL.as_posix()), str(ARM.as_posix()),
         "configs/model/vietnote.json"], cwd=REPO)
    with ZipFile(io.BytesIO(raw)) as archive:
        files = {member.filename: archive.read(member) for member in archive.infolist() if not member.is_dir()}
    for path in (PROTOCOL, ARM, LOCK.relative_to(REPO)):
        if files[path.as_posix()] != (REPO / path).read_bytes():
            # Compare the registered protocol without depending on the checkout's line endings.
            if files[path.as_posix()].replace(b"\r\n", b"\n") != (REPO / path).read_bytes().replace(b"\r\n", b"\n"):
                raise ValueError(f"Commit the updated protocol before packaging: {path.as_posix()}")
    model_files = [item for item in card["model_files"] if item["path"] in {"tokens.txt", "bpe.model", "pytorch/pretrained.pt"}]
    model_files += [{**item, "path": f"upstream/{item['path']}"} for item in card["recipe_files"]]
    model_files.append({**card["icefall_archive"], "path": f"upstream/icefall/{card['icefall_commit']}.tar.gz"})
    for item in model_files:
        source = base / item["path"]
        verify_file(source, item)
        files[f"models/vietnote78/{item['path']}"] = source
    R.forbid_bench_catalog(CATALOG, "V02 Kaggle")
    catalog = R.resolve(CATALOG)
    catalog_file = "data/releases/catalog/train-v1/catalog.tsv"
    files[catalog_file] = catalog.catalog
    files["data/releases/catalog/train-v1/release.json"] = json.dumps({
        "ref": CATALOG, "files": {"catalog.tsv": {"sha256": R.sha256(catalog.catalog)}},
    }).encode()
    portable = []
    for row in tiny_rows():
        source = Path(row["_audio_path"])
        digest = R.sha256(source)
        if digest != row["audio_sha256"] or not source.suffix.isascii() or not source.suffix[1:].isalnum():
            raise ValueError(f"Stale or unsupported V02 audio: {row['id']}")
        filename = f"audio/{digest}{source.suffix.lower()}"
        files[f"portable/{filename}"] = source
        portable.append({**{k: v for k, v in row.items() if not k.startswith("_")}, "audio": filename})
    files["portable/train.jsonl"] = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in portable).encode("utf-8")
    if runtime:
        for name in ("lock.json", "requirements.txt"):
            files[f"runtime/{name}"] = runtime / name
        for wheel in (runtime / "wheels").glob("*.whl"):
            files[f"runtime/wheels/{wheel.name}"] = wheel
    total = sum(len(value) if isinstance(value, bytes) else value.stat().st_size for value in files.values())
    if shutil.disk_usage(directory).free < total + 2 * 2**30:
        raise OSError("Insufficient free disk space for the V02 bundle")
    destination = directory / "bundle.zip"
    pending = destination.with_suffix(".tmp")
    write_zip(pending, files)
    os.replace(pending, destination)
    return {
        "sha256": R.sha256(destination), "bytes": destination.stat().st_size,
        "train_rows": len(portable), "train_manifest_sha256": manifest_sha256(portable),
        "model_revision": card["revision"], "icefall_commit": card["icefall_commit"],
    }
