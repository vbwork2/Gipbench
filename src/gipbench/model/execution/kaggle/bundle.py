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
CV_RELEASE = "pool/cv_other@v3"
LIBRI_RELEASE = "pool/vietnote_librispeech_en@v2"
DATAHIVE_HELD_OUT_CALLS = ("vi_000002", "vi_000004", "vi_000008")
AUTHORIZED_LICENSE = "CC-BY-4.0 / CC0-1.0"


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


def _encode_flac_bytes(audio_path):
    import soundfile as sf
    from gipbench.infer.adapters.base import load_audio

    samples, rate = load_audio(audio_path, target_sr=16000)
    payload = io.BytesIO()
    sf.write(payload, samples, rate, format="FLAC", subtype="PCM_16")
    data = payload.getvalue()
    return data, len(samples) / float(rate)


def _common_diag_files(commit, include_logged_cli=False):
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
    scripts = ["vietnote_train_validation.py", "vietnote_torch_check.py"]
    if include_logged_cli:
        scripts.append("vietnote_logged_cli.py")
    for name in scripts:
        rel = f"scripts/diagnostics/{name}"
        files[rel] = REPO / rel
    R.forbid_bench_catalog(CATALOG, "Authorized Kaggle diagnostic")
    catalog = R.resolve(CATALOG)
    files["data/releases/catalog/train-v1/catalog.tsv"] = catalog.catalog
    files["data/releases/catalog/train-v1/release.json"] = json.dumps({
        "ref": CATALOG, "files": {"catalog.tsv": {"sha256": R.sha256(catalog.catalog)}},
    }).encode("utf-8")
    return files


def _portable_flac_rows(rows, files):
    portable = []
    for row in rows:
        source = Path(row["_audio_path"]).resolve()
        if row.get("audio_sha256") and R.sha256(source) != row["audio_sha256"]:
            raise ValueError(f"Audio digest mismatch: {row['id']}")
        data, duration = _encode_flac_bytes(source)
        digest = hashlib.sha256(data).hexdigest()
        name = f"audio/{digest}.flac"
        files[name] = data
        item = {k: v for k, v in row.items() if not k.startswith("_")}
        text_tagged = item.get("text_tagged") or item.get("text") or item.get("text_orig") or ""
        speaker = item.get("speaker") or item.get("source") or item["id"]
        item.update(audio=name, audio_sha256=digest, duration=round(duration, 4),
                    text_tagged=text_tagged, speaker=speaker)
        portable.append(item)
    return portable


def tiny_diag_inputs(commit):
    """Build a strictly authorized 163-row (8.1-16.0s -> 489 steps) tiny-set LR diagnostic bundle."""
    fleurs = R.resolve(FLEURS_RELEASE)
    cv = R.resolve(CV_RELEASE)
    libri = R.resolve(LIBRI_RELEASE)
    cv_rows = sorted((r for r in read_jsonl(cv.manifest) if 8.1 <= float(r["duration"]) <= 16.0),
                     key=lambda r: r["id"])
    libri_rows = sorted((r for r in read_jsonl(libri.manifest) if 8.1 <= float(r["duration"]) <= 16.0),
                        key=lambda r: (str(r.get("speaker") or ""), r["id"]))
    fleurs_rows = sorted((r for r in read_jsonl(fleurs.manifest) if 8.1 <= float(r["duration"]) <= 16.0),
                         key=lambda r: (str(r.get("source") or ""), r["id"]))
    # Pick distinct speakers/sources deterministically: all 19 qualifying CV, 30 LibriSpeech, 114 FLEURS = 163 rows.
    selected_cv = cv_rows[:19]
    selected_libri = [libri_rows[i * len(libri_rows) // 30] for i in range(30)]
    seen_src, selected_fleurs = set(), []
    for r in fleurs_rows:
        src = str(r.get("source") or r["id"])
        if src in seen_src:
            continue
        seen_src.add(src)
        selected_fleurs.append(r)
        if len(selected_fleurs) == 163 - len(selected_cv) - len(selected_libri):
            break
    selected = sorted([*selected_cv, *selected_libri, *selected_fleurs], key=lambda r: r["id"])
    if len(selected) != 163:
        raise ValueError(f"Expected 163 authorized tiny-diag rows, got {len(selected)}")
    files = _common_diag_files(commit, include_logged_cli=False)
    portable = _portable_flac_rows(selected, files)
    files["train.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in portable).encode("utf-8")
    members = add_bundle_members(files)
    return files, {
        "license": AUTHORIZED_LICENSE,
        "releases": {
            FLEURS_RELEASE: R.sha256(fleurs.manifest),
            CV_RELEASE: R.sha256(cv.manifest),
            LIBRI_RELEASE: R.sha256(libri.manifest),
        },
        "train_rows": len(portable),
        "train_manifest_sha256": manifest_sha256(portable),
        "members_manifest_sha256": hashlib.sha256(members).hexdigest(),
    }


def _authorized_dev_splits():
    """Build the deterministic source/speaker-disjoint 71-row authorized_dev split and exclusion sets."""
    fleurs = R.resolve(FLEURS_RELEASE)
    cv = R.resolve(CV_RELEASE)
    libri = R.resolve(LIBRI_RELEASE)

    fleurs_all = sorted(read_jsonl(fleurs.manifest), key=lambda r: (str(r.get("source") or ""), r["id"]))
    fleurs_gt20 = [r for r in fleurs_all if float(r["duration"]) > 20.0]
    dev_fleurs_gt20, train_fleurs_gt20 = [], []
    dev_fleurs_sources = set()
    for idx, row in enumerate(fleurs_gt20):
        src = str(row.get("source") or row["id"])
        if idx % 4 == 0 and len(dev_fleurs_gt20) < 14:
            dev_fleurs_gt20.append(row)
            dev_fleurs_sources.add(src)
        else:
            train_fleurs_gt20.append(row)
    train_fleurs_gt20 = [r for r in train_fleurs_gt20 if str(r.get("source") or r["id"]) not in dev_fleurs_sources]
    fleurs_le20 = [r for r in fleurs_all if 2.0 <= float(r["duration"]) <= 20.0
                   and str(r.get("source") or r["id"]) not in dev_fleurs_sources]
    by_src = {}
    for r in fleurs_le20:
        by_src.setdefault(str(r.get("source") or r["id"]), []).append(r)
    src_keys = sorted(by_src)
    dev_fleurs_le20 = [by_src[k][0] for k in src_keys[:26]]
    dev_fleurs_sources.update(src_keys[:26])

    cv_all = sorted(read_jsonl(cv.manifest), key=lambda r: (str(r.get("speaker") or ""), r["id"]))
    cv_by_spk = {}
    for r in cv_all:
        if float(r["duration"]) >= 1.5 and r.get("speaker"):
            cv_by_spk.setdefault(str(r["speaker"]), []).append(r)
    cv_spks = sorted(cv_by_spk)
    dev_cv_speakers = set(cv_spks[:20])
    dev_cv = [cv_by_spk[s][0] for s in cv_spks[:20]]

    libri_all = sorted(read_jsonl(libri.manifest), key=lambda r: (str(r.get("speaker") or ""), r["id"]))
    libri_by_spk = {}
    for r in libri_all:
        if r.get("speaker"):
            libri_by_spk.setdefault(str(r["speaker"]), []).append(r)
    libri_spks = sorted(libri_by_spk)
    dev_libri_speakers = set(libri_spks[:2])
    dev_libri = [r for s in libri_spks[:2] for r in libri_by_spk[s][:10]][:20]

    valid_sel = sorted([*dev_fleurs_gt20, *dev_fleurs_le20, *dev_cv, *dev_libri], key=lambda r: r["id"])
    return {
        "valid_sel": valid_sel,
        "train_fleurs_gt20": train_fleurs_gt20,
        "fleurs_le20_by_src": by_src,
        "fleurs_src_keys": src_keys,
        "dev_fleurs_sources": dev_fleurs_sources,
        "cv_by_spk": cv_by_spk,
        "cv_spks": cv_spks,
        "dev_cv_speakers": dev_cv_speakers,
        "libri_by_spk": libri_by_spk,
        "libri_spks": libri_spks,
        "dev_libri_speakers": dev_libri_speakers,
        "releases": {
            FLEURS_RELEASE: R.sha256(fleurs.manifest),
            CV_RELEASE: R.sha256(cv.manifest),
            LIBRI_RELEASE: R.sha256(libri.manifest),
        },
    }


def diag_ft_inputs(commit):
    """Build a strictly source/speaker-disjoint authorized train/dev bundle including >20s utterances."""
    splits = _authorized_dev_splits()
    train_fleurs_gt20 = splits["train_fleurs_gt20"]
    by_src = splits["fleurs_le20_by_src"]
    src_keys = splits["fleurs_src_keys"]
    train_fleurs_le20 = [by_src[k][0] for k in src_keys[26:346]]
    cv_by_spk, cv_spks = splits["cv_by_spk"], splits["cv_spks"]
    train_cv = [r for s in cv_spks[20:] for r in cv_by_spk[s][:2]][:180]
    libri_by_spk, libri_spks = splits["libri_by_spk"], splits["libri_spks"]
    train_libri = [r for s in libri_spks[2:] for r in libri_by_spk[s][:8]][:80]

    train_sel = sorted([*train_fleurs_gt20, *train_fleurs_le20, *train_cv, *train_libri], key=lambda r: r["id"])
    valid_sel = splits["valid_sel"]
    if ({r["id"] for r in train_sel} & {r["id"] for r in valid_sel}
            or {r["source"] for r in train_sel} & {r["source"] for r in valid_sel}):
        raise ValueError("Source or ID overlap between diag-ft train and dev splits")

    files = _common_diag_files(commit, include_logged_cli=True)
    train_portable = _portable_flac_rows(train_sel, files)
    valid_portable = _portable_flac_rows(valid_sel, files)
    files["train.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_portable).encode("utf-8")
    files["valid.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in valid_portable).encode("utf-8")
    members = add_bundle_members(files)
    return files, {
        "license": AUTHORIZED_LICENSE,
        "releases": splits["releases"],
        "train_rows": len(train_portable),
        "valid_rows": len(valid_portable),
        "train_gt20_rows": sum(1 for r in train_portable if float(r["duration"]) > 20.0),
        "valid_gt20_rows": sum(1 for r in valid_portable if float(r["duration"]) > 20.0),
        "train_manifest_sha256": manifest_sha256(train_portable),
        "valid_manifest_sha256": manifest_sha256(valid_portable),
        "members_manifest_sha256": hashlib.sha256(members).hexdigest(),
    }


def expl_ft_inputs(commit):
    """Build the authorized exploratory VE01 bundle mixing conversational DataHive VI, read VI (>20s incl.), and EN."""
    splits = _authorized_dev_splits()
    valid_sel = splits["valid_sel"]
    valid_ids = {r["id"] for r in valid_sel}
    valid_sources = {str(r.get("source") or "") for r in valid_sel}
    valid_shas = {r.get("audio_sha256") for r in valid_sel if r.get("audio_sha256")}

    # 1) FLEURS VI: all >20s train clips + up to 750 source-disjoint 2-20s clips (~2.9h total)
    by_src = splits["fleurs_le20_by_src"]
    src_keys = splits["fleurs_src_keys"]
    train_fleurs = [*splits["train_fleurs_gt20"], *[by_src[k][0] for k in src_keys[26:776]]]

    # 2) Auto-curated Common Voice VI (speaker-disjoint from authorized_dev)
    cv_curated_path = REPO / "data/work/vietnote/curation/cv_other_curated.jsonl"
    if not cv_curated_path.is_file():
        raise FileNotFoundError(f"Run scripts/dataset/vietnote_auto_curate.py first: missing {cv_curated_path}")
    train_cv = [
        r for r in read_jsonl(cv_curated_path)
        if str(r.get("speaker") or "") not in splits["dev_cv_speakers"]
        and r["id"] not in valid_ids
        and float(r["duration"]) >= 1.5
    ]

    # 3) Curated conversational DataHive VI (5 train calls only; strictly exclude held-out dev calls)
    dh_curated_path = REPO / "data/work/vietnote/curation/datahive_train_curated.jsonl"
    if not dh_curated_path.is_file():
        raise FileNotFoundError(f"Run scripts/dataset/vietnote_auto_curate.py first: missing {dh_curated_path}")
    dh_rows = read_jsonl(dh_curated_path)
    if any(c in str(r.get("source") or "") for r in dh_rows for c in DATAHIVE_HELD_OUT_CALLS):
        raise ValueError("DataHive held-out dev call detected in curated train manifest")
    if any(r.get("license") != "CC-BY-4.0" for r in dh_rows):
        raise ValueError("All DataHive train rows must carry CC-BY-4.0 license")
    train_dh = [r for r in dh_rows if 1.5 <= float(r["duration"]) <= 35.0]

    # 4) Native English LibriSpeech (~1.1h, speaker-disjoint from authorized_dev)
    libri_by_spk, libri_spks = splits["libri_by_spk"], splits["libri_spks"]
    train_libri = []
    libri_sec = 0.0
    for r in [row for s in libri_spks[2:] for row in libri_by_spk[s]]:
        if libri_sec >= 1.15 * 3600.0:
            break
        train_libri.append(r)
        libri_sec += float(r["duration"])

    train_sel = sorted([*train_fleurs, *train_cv, *train_dh, *train_libri], key=lambda r: r["id"])
    if ({r["id"] for r in train_sel} & valid_ids
            or {str(r.get("source") or "") for r in train_sel} & valid_sources
            or {r.get("audio_sha256") for r in train_sel if r.get("audio_sha256")} & valid_shas):
        raise ValueError("Source, ID, or audio_sha256 overlap between VE01 train and authorized_dev")

    files = _common_diag_files(commit, include_logged_cli=True)
    train_portable = _portable_flac_rows(train_sel, files)
    valid_portable = _portable_flac_rows(valid_sel, files)
    files["train.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train_portable).encode("utf-8")
    files["valid.jsonl"] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in valid_portable).encode("utf-8")
    members = add_bundle_members(files)
    source_hours = {
        "fleurs_vi": round(sum(float(r["duration"]) for r in train_fleurs) / 3600.0, 4),
        "cv_other_curated": round(sum(float(r["duration"]) for r in train_cv) / 3600.0, 4),
        "datahive_vi_curated": round(sum(float(r["duration"]) for r in train_dh) / 3600.0, 4),
        "librispeech_en": round(sum(float(r["duration"]) for r in train_libri) / 3600.0, 4),
    }
    return files, {
        "license": AUTHORIZED_LICENSE,
        "releases": splits["releases"],
        "curation_manifests": {
            "cv_other_curated_sha256": R.sha256(cv_curated_path),
            "datahive_train_curated_sha256": R.sha256(dh_curated_path),
        },
        "source_hours": source_hours,
        "train_rows": len(train_portable),
        "train_hours": round(sum(float(r["duration"]) for r in train_portable) / 3600.0, 4),
        "valid_rows": len(valid_portable),
        "train_gt20_rows": sum(1 for r in train_portable if float(r["duration"]) > 20.0),
        "valid_gt20_rows": sum(1 for r in valid_portable if float(r["duration"]) > 20.0),
        "train_manifest_sha256": manifest_sha256(train_portable),
        "valid_manifest_sha256": manifest_sha256(valid_portable),
        "members_manifest_sha256": hashlib.sha256(members).hexdigest(),
    }
