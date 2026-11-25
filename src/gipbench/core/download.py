"""Fetch a pinned public file with a size limit and resumable partial file."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from gipbench.core.fileio import sha256_file


def verify_file(path: Path, spec: dict):
    if path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Incorrect size: {path}")
    digest = sha256_file(path)
    if spec.get("sha256") and digest != spec["sha256"]:
        raise ValueError(f"Incorrect SHA256: {path}")
    if spec.get("git_blob_sha1"):
        git = hashlib.sha1(f"blob {spec['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                git.update(block)
        if git.hexdigest() != spec["git_blob_sha1"]:
            raise ValueError(f"Incorrect Git blob ID: {path}")
    return digest


def download_file(url: str, path: Path, spec: dict, reserve: int):
    if path.exists():
        return verify_file(path, spec)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Closing the descriptor releases the lock even if a download is interrupted.
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if path.exists():
            return verify_file(path, spec)
        return _download_partial(url, path, spec, reserve)


def _download_partial(url: str, path: Path, spec: dict, reserve: int):
    partial = path.with_name(path.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    remaining = spec["bytes"] - offset
    if remaining < 0:
        raise ValueError(f"Partial download exceeds pinned size: {partial}")
    if shutil.disk_usage(path.parent).free < reserve + remaining:
        raise OSError(f"Insufficient disk space for {path}")
    for _ in range(3):
        if not remaining:
            break
        result = subprocess.run([
            "curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "360",
            "--connect-timeout", "20", "--continue-at", "-", "--max-filesize", str(spec["bytes"]),
            "--output", str(partial), url,
        ])
        size = partial.stat().st_size if partial.exists() else 0
        remaining = spec["bytes"] - size
        if not remaining:
            break
        if result.returncode not in (18, 28, 56) or size <= offset:
            result.check_returncode()
            raise OSError(f"No download progress for {path}")
        print(f"Resume {path}: {size:,}/{spec['bytes']:,} bytes", flush=True)
        offset = size
    if remaining:
        raise TimeoutError(f"Download incomplete; retained {partial.stat().st_size:,} bytes in {partial}")
    digest = verify_file(partial, spec)
    try:
        os.link(partial, path)
    except FileExistsError:
        verify_file(path, spec)
    partial.unlink()
    return digest
