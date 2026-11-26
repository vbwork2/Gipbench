"""Read a pinned remote Parquet file through a resumable cache of HTTP ranges."""

import io
import os
import shutil
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


class ByteBudgetExceeded(OSError):
    pass


def _hf_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    cached = Path.home() / ".cache/huggingface/token"
    if cached.is_file():
        return cached.read_text(encoding="utf-8").strip()
    return ""


class ParquetRanges(io.RawIOBase):
    block_size = 4 * 1024**2

    def __init__(self, url, size, cache: Path, budget, reserve):
        self.url, self.size, self.cache = url, size, cache
        self.budget, self.reserve = budget, reserve
        self.offset = self.transferred = 0
        self._last_start, self._last_block = None, b""
        self._token = _hf_token() if url.startswith("https://huggingface.co/") else ""
        cache.mkdir(parents=True, exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.offset

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.offset, io.SEEK_END: self.size}[whence]
        if base + offset < 0:
            raise ValueError("Cannot seek before the beginning of a parquet file")
        self.offset = base + offset
        return self.offset

    def block(self, start):
        length = min(self.block_size, self.size - start)
        path = self.cache / f"{start:014d}.block"
        if path.exists():
            data = path.read_bytes()
            if len(data) != length:
                raise ValueError(f"Incomplete cached range: {path}")
            return data
        if self.transferred + length > self.budget:
            raise ByteBudgetExceeded(f"Range budget exhausted after {self.transferred:,} bytes")
        if shutil.disk_usage(self.cache).free < self.reserve + length:
            raise OSError(f"Insufficient disk space in {self.cache}")
        end = start + length - 1
        headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        request = Request(self.url, headers=headers)
        with urlopen(request, timeout=90) as response:
            if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{self.size}":
                raise OSError("The server did not return the pinned byte range")
            self.transferred += length
            data = response.read(length + 1)
        if len(data) != length:
            raise OSError(f"Short byte range at {start}")
        fd, temp = tempfile.mkstemp(dir=self.cache)
        try:
            with os.fdopen(fd, "wb") as out:
                out.write(data)
            os.replace(temp, path)
        finally:
            Path(temp).unlink(missing_ok=True)
        return data

    def readinto(self, buffer):
        view = memoryview(buffer).cast("B")
        length = min(len(view), max(0, self.size - self.offset))
        done = 0
        while done < length:
            start = self.offset // self.block_size * self.block_size
            if start != self._last_start:
                self._last_block = self.block(start)
                self._last_start = start
            data = self._last_block
            local = self.offset - start
            count = min(length - done, len(data) - local)
            view[done:done + count] = data[local:local + count]
            self.offset += count
            done += count
        return done
