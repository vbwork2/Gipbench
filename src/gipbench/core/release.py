"""Data release: phiên bản dữ liệu bất biến, là hợp đồng giữa pipeline dataset (ghi) và model / bench (đọc).

  data/releases/<kind>/<name>@<version>/   kind: bench | catalog | pool | dev   (bench, catalog: data/releases/<kind>/<version>/)
    manifest.jsonl   (bench, pool, dev)   audio: đường dẫn TƯƠNG ĐỐI theo file manifest
    catalog.tsv      (catalog)
    release.json     metadata: sha256, số dòng / giờ, nguồn (parents), catalog, flags, ghi chú

Tham chiếu bằng chuỗi "<kind>/<name>@<version>" (bench / catalog: "bench/v0-draft"). Đọc qua resolve(); resolve kiểm
sha256 nên release bị sửa tay sẽ báo lỗi. Đặc tả đầy đủ: docs/CONTRACTS.md.
"""
import datetime as dt
import hashlib
import json
import os
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from gipbench.core import paths

KINDS = ("bench", "catalog", "pool", "dev")


class ReleaseError(Exception):
    pass


@dataclass
class Release:
    ref: str
    dir: Path
    meta: Dict

    @property
    def manifest(self) -> Optional[Path]:
        p = self.dir / "manifest.jsonl"
        return p if p.exists() else None

    @property
    def catalog(self) -> Optional[Path]:
        """catalog.tsv của chính release (kind catalog) hoặc của catalog release mà nó ghi trong meta."""
        p = self.dir / "catalog.tsv"
        if p.exists():
            return p
        return resolve(self.meta["catalog"]).catalog if self.meta.get("catalog") else None


def root() -> Path:
    return paths.path("data_root") / "releases"


def parse(ref: str):
    kind, _, rest = ref.partition("/")
    if kind not in KINDS or not rest:
        raise ReleaseError(f"tham chiếu release sai: {ref!r} (dạng <{'|'.join(KINDS)}>/<tên>@<version>)")
    name, _, ver = rest.partition("@")
    if kind in ("bench", "catalog"):
        return kind, name, ver or name
    if not ver:
        raise ReleaseError(f"thiếu @version: {ref!r}")
    return kind, name, ver


def release_dir(ref: str) -> Path:
    kind, name, ver = parse(ref)
    return root() / kind / (ver if kind in ("bench", "catalog") else f"{name}@{ver}")


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def resolve(ref: str, verify: bool = True) -> Release:
    d = release_dir(ref)
    mf = d / "release.json"
    if not mf.exists():
        raise ReleaseError(f"chưa có release {ref} ({d}); dựng bằng: python -m gipbench.dataset.release.register {ref}")
    meta = json.loads(mf.read_text(encoding="utf-8"))
    if verify:
        for fname, info in meta.get("files", {}).items():
            got = sha256(d / fname)
            if got != info["sha256"]:
                raise ReleaseError(f"{ref}: {fname} đã bị sửa sau khi phát hành (sha256 {got[:12]} != {info['sha256'][:12]}). "
                                   "Release là bất biến: dựng version mới thay vì sửa.")
    return Release(ref, d, meta)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _stats(rows: List[Dict]) -> Dict:
    hours = lambda rs: round(sum(float(r.get("duration", 0)) for r in rs) / 3600, 3)  # noqa: E731
    by = {}
    for key in ("split", "dataset", "domain"):
        c = Counter()
        for r in rows:
            v = str(r.get(key, "")).split(":")[0] if key == "dataset" else str(r.get(key, ""))
            c[v] += float(r.get("duration", 0)) / 3600
        by[f"hours_by_{key}"] = {k: round(v, 3) for k, v in c.most_common()}
    by["cs_rows"] = sum("<en>" in (r.get("text_tagged") or "") for r in rows)
    by["hours"] = hours(rows)
    return by


def write_release(ref: str, *, manifest_rows: Optional[List[Dict]] = None, catalog_src: Optional[Path] = None,
                  source: Optional[Path] = None, parents: Optional[List[Path]] = None, catalog_ref: Optional[str] = None,
                  flags: Optional[Dict] = None, notes: str = "", builder: str = "", builder_args: Optional[Dict] = None,
                  overwrite: bool = False) -> Release:
    """Ghi một release mới. manifest_rows: các dòng đã đọc bằng core.manifest.read_jsonl (có _audio_path); audio được ghi
    lại thành đường dẫn tương đối theo thư mục release. Thư mục release đã có thì báo lỗi (trừ overwrite=True)."""
    d = release_dir(ref)
    if (d / "release.json").exists() and not overwrite:
        raise ReleaseError(f"release {ref} đã tồn tại ({d}): release là bất biến, đặt version mới")
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    if manifest_rows is not None:
        base = os.path.abspath(d)
        with open(d / "manifest.jsonl", "w", encoding="utf-8") as f:
            for r in manifest_rows:
                out = {}
                for k, v in r.items():
                    if k.startswith("_"):
                        continue
                    if k == "audio" and r.get("_audio_path"):
                        v = os.path.relpath(os.path.normpath(r["_audio_path"]), base)
                    out[k] = v
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
        files["manifest.jsonl"] = {"sha256": sha256(d / "manifest.jsonl"), "rows": len(manifest_rows),
                                   "hours": round(sum(float(r.get("duration", 0)) for r in manifest_rows) / 3600, 3)}
    if catalog_src is not None:
        (d / "catalog.tsv").write_bytes(Path(catalog_src).read_bytes())
        files["catalog.tsv"] = {"sha256": sha256(d / "catalog.tsv"),
                                "rows": sum(1 for l in open(d / "catalog.tsv", encoding="utf-8") if l.strip()) - 1}
    kind, name, ver = parse(ref)
    rel = lambda p: os.path.relpath(os.path.abspath(p), paths.REPO).replace(os.sep, "/")  # noqa: E731
    meta = {"kind": kind, "name": name, "version": ver, "ref": ref,
            "created": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "builder": builder, "builder_args": builder_args or {}, "files": files,
            "source": rel(source) if source else None,
            "parents": [{"path": rel(p), "sha256": sha256(p)} for p in (parents or []) if Path(p).exists()],
            "catalog": catalog_ref, "flags": flags or {}, "notes": notes,
            "stats": _stats(manifest_rows) if manifest_rows is not None else {}}
    (d / "release.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return Release(ref, d, meta)


def materialize(ref: str, out: Path) -> Path:
    """Chép manifest của release ra `out` với audio là đường dẫn TUYỆT ĐỐI (để ghi sang thư mục khác không hỏng
    đường dẫn). Giữ nguyên thứ tự dòng và thứ tự khoá; chỉ đổi giá trị trường audio."""
    rl = resolve(ref)
    if rl.manifest is None:
        raise ReleaseError(f"{ref} không có manifest")
    base = rl.manifest.resolve().parent
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(rl.manifest, encoding="utf-8") as fi, open(out, "w", encoding="utf-8") as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            if "audio" in r and not os.path.isabs(r["audio"]):
                r["audio"] = os.path.normpath(str(base / r["audio"])).replace(os.sep, "/")
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out
