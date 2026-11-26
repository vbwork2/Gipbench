"""Prepare deterministic, minimal VietNote research inputs for Kaggle."""

import base64
import hashlib
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
from gipbench.core.manifest import manifest_sha256, read_jsonl
from gipbench.core.paths import REPO
from gipbench.model.vietnote import LOCK

PROTOCOL = Path("configs/model/experiments/vietnote_validation.yaml")
ARM = PROTOCOL.with_name("vietnote_tiny_low_lr.yaml")
CATALOG = "catalog/train-v1"
TINY_SHA256 = "f3ff60b1e7739fcc78dbf1eb99e92342113a3ff6958dfca157b26f4d3831b46a"
FLEURS_RELEASE = "pool/vietnote_fleurs_vi@v2"
FLEURS_SHA256 = "d0bd604f368f820456d22837fcab3d950f8d0d43b647c23d5800e5a9d2ba7e49"


def fleurs_pilot():
    import soundfile as sf

    release = R.resolve(FLEURS_RELEASE)
    release_digest = R.sha256(release.manifest)
    if release_digest != FLEURS_SHA256:
        raise ValueError("Unexpected FLEURS release for the Kaggle engineering pilot")
    rows = sorted(read_jsonl(release.manifest), key=lambda row: row["id"])
    sources, texts, result = set(), set(), {}
    for split, count, bounds in (("train", 4, (4.0, 5.0)), ("valid", 32, (4.0, 6.0))):
        selected = []
        for row in rows:
            source = str(row["source"])
            label = row["text_tagged"]
            if not bounds[0] <= row["duration"] <= bounds[1] or source in sources or label in texts:
                continue
            audio = Path(row["_audio_path"])
            if R.sha256(audio) != row["audio_sha256"]:
                raise ValueError(f"FLEURS pilot audio differs from the release: {row['id']}")
            samples, rate = sf.read(audio, dtype="float32")
            if rate != 16000 or samples.ndim != 1:
                raise ValueError("Expected mono 16 kHz FLEURS audio")
            payload = io.BytesIO()
            sf.write(payload, samples, rate, format="FLAC", subtype="PCM_16")
            data = payload.getvalue()
            selected.append(dict(id=row["id"], source=source, text=label, duration=len(samples) / rate,
                                 release_audio_sha256=row["audio_sha256"], audio_sha256=hashlib.sha256(data).hexdigest(),
                                 flac=base64.b64encode(data).decode("ascii")))
            sources.add(source)
            texts.add(label)
            if len(selected) == count:
                break
        if len(selected) != count:
            raise ValueError(f"Not enough distinct FLEURS prompts for {split}")
        result[split] = selected
    return dict(release=release.ref, release_sha256=release_digest, license="CC-BY-4.0", **result)


def fleurs_memory_inputs(commit):
    release = R.resolve(FLEURS_RELEASE)
    if R.sha256(release.manifest) != FLEURS_SHA256:
        raise ValueError("Unexpected FLEURS release for the Kaggle memory probe")
    protocol = PROTOCOL.with_name("vietnote_memory.yaml")
    memory = yaml.safe_load((REPO / protocol).read_text(encoding="utf-8"))
    train = yaml.safe_load((REPO / "configs/model/experiments/vietnote_v03.yaml").read_text(encoding="utf-8"))["train"]
    limit, count, minimum = train["max_duration"], memory["steps_per_case"], memory["long_clip_min_seconds"]
    if limit != 200 or train["fp16"] or memory["precision"] != "fp32":
        raise ValueError("Expected the 200-second FP32 configuration")
    rows = sorted(read_jsonl(release.manifest), key=lambda row: (-float(row["duration"]), row["id"]))
    long = [row for row in rows if minimum <= row["duration"] <= limit]
    anchor = max(long, key=lambda row: (int(limit / row["duration"]) * row["duration"], row["duration"]))
    width = int(limit / anchor["duration"])
    others = [row for row in rows if row["source"] != anchor["source"] and row["duration"] <= anchor["duration"]]
    selected = [anchor, *others[:count * (width - 1)]]
    if len(selected) != 1 + count * (width - 1):
        raise ValueError("Not enough released FLEURS clips for the requested batches")
    groups = [[anchor["id"], *(row["id"] for row in selected[1 + i * (width - 1):1 + (i + 1) * (width - 1)])]
              for i in range(count)]
    raw = subprocess.check_output(["git", "archive", "--format=zip", commit, "src"], cwd=REPO)
    with ZipFile(io.BytesIO(raw)) as archive:
        files = {member.filename: archive.read(member) for member in archive.infolist() if not member.is_dir()}
    files[LOCK.relative_to(REPO).as_posix()] = LOCK
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    for spec in card["recipe_files"]:
        if spec["path"].startswith("src/vnasr/"):
            file = REPO / "models/vietnote78/upstream" / spec["path"]
            verify_file(file, spec)
            files["normalizer/" + spec["path"]] = file
    for name in ("vietnote_train_validation.py", "vietnote_torch_check.py"):
        path = f"scripts/diagnostics/{name}"
        files[path] = REPO / path
    portable = []
    for row in selected:
        audio = Path(row["_audio_path"])
        if R.sha256(audio) != row["audio_sha256"]:
            raise ValueError(f"FLEURS memory audio differs from the release: {row['id']}")
        name = f"audio/{row['audio_sha256']}{audio.suffix.lower()}"
        files[name] = audio
        portable.append({key: row[key] for key in ("id", "source", "text_tagged", "duration", "audio_sha256")}
                        | {"audio": name})
    plan = {"release": release.ref, "release_sha256": FLEURS_SHA256, "license": "CC-BY-4.0",
            "limit": limit, "precision": memory["precision"], "seed": 0, "base_lr": train["lr"],
            "long_clip_min_seconds": minimum, "min_batch_fraction": memory["min_batch_fraction"],
            "min_free_fraction": memory["min_free_fraction"], "groups": groups}
    files["batches.json"] = json.dumps(plan).encode()
    files["train.jsonl"] = "".join(json.dumps(row) + "\n" for row in portable).encode()
    members = add_bundle_members(files)
    return files, {**{key: plan[key] for key in ("release", "release_sha256", "license", "limit")},
                   "members_manifest_sha256": hashlib.sha256(members).hexdigest(),
                   "planned_steps": count}


def add_bundle_members(files):
    members = json.dumps({name: hashlib.sha256(value).hexdigest() if isinstance(value, bytes) else R.sha256(value)
                          for name, value in files.items()}, sort_keys=True).encode("utf-8")
    files["bundle-members.json"] = members
    return members


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
    members = add_bundle_members(files)
    total = sum(len(value) if isinstance(value, bytes) else value.stat().st_size for value in files.values())
    if shutil.disk_usage(directory).free < total + 2 * 2**30:
        raise OSError("Insufficient free disk space for the V02 bundle")
    destination = directory / "bundle.zip"
    pending = destination.with_suffix(".tmp")
    write_zip(pending, files)
    os.replace(pending, destination)
    return {
        "sha256": R.sha256(destination), "bytes": destination.stat().st_size,
        "members_manifest_sha256": hashlib.sha256(members).hexdigest(),
        "train_rows": len(portable), "train_manifest_sha256": manifest_sha256(portable),
        "model_revision": card["revision"], "icefall_commit": card["icefall_commit"],
    }
