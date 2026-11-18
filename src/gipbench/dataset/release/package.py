#!/usr/bin/env python3
"""Đóng gói bản share của bench: manifest KHÔNG kèm audio + datasheet + config chấm + SHA256SUMS.

  python -m gipbench.dataset.release.package --benchmarks general_v1 code_switch_v1 cs_ggroup_v1 \
      --name gipbench-bench-v1.0 [--out <data_root>/share]
Benchmark có manifest_sha256 (bản khoá): manifest share phải trùng hash đó (và catalog trùng catalog_sha256), nếu không thì dừng
trước khi ghi gì.

Mỗi manifest share = manifest của benchmark, bỏ trường `audio` (đường dẫn máy) và mọi trường bắt đầu bằng "_", thêm
`audio_derivation` (upstream_bytes | soxr_hq_16k_mono_pcm16) để fetch biết cách kiểm. Từ chối đóng gói nếu có dòng
license test_only / nguồn riêng tư, hoặc nguồn không có trong gipbench.dataset.release.fetch.SOURCES.
Người nhận: python -m gipbench.dataset.release.fetch --manifest manifests/<x>.jsonl --out ... (xem README trong gói).
Gói kèm code/ (src gipbench tối thiểu + configs + pyproject, tính bằng AST từ fetch / leak_audit / evaluate / transcribe):
người nhận cài bằng `pip install -e code/`, không cần repo. Đóng gói dừng nếu code đó import module train / dữ liệu train
(`forbidden` trong configs/dataset/share_code.yaml).

Chính sách nội dung (configs/dataset/share_policy.yaml, IP-08): gói được dựng ở `.<name>.partial/`, rồi kiểm (1) mọi file
khớp allowlist, (2) không file text nào chạm regex cấm (đường dẫn máy, tên module train, bộ riêng tư, token). Đạt thì
đổi tên thành <name>, không đạt thì xoá thư mục tạm và dừng. datasets.yaml trong gói chỉ giữ các bộ có trong benchmark
được đóng gói. Bộ trong `text_not_redistributable` bị bỏ lời thoại (fetch lấy lại từ nguồn). release.json ghi
`distinct_utterances` (hợp theo mã clip / audio_sha256 / (họ bộ, upstream_id)) và `scorer_code_sha256` của code kèm gói.
"""
import argparse
import hashlib
import json
import re
import shutil
import time
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.decontam import clip_key, dataset_family
from gipbench.core.manifest import read_jsonl
from gipbench.core.normprofiles import code_sha256
from gipbench.dataset.release.build_general import manifest_sha256
from gipbench.dataset.release.fetch import DERIVED, SOURCES, sha256_file

PRIVATE_LICENSES = {"test_only"}
SHARE_POLICY_CONFIG = P.REPO / "configs" / "dataset" / "share_policy.yaml"
# bỏ khi chép docs vào gói: (1) khối <!-- share:omit --> ... <!-- /share:omit --> (marker đầu dòng: bỏ luôn xuống dòng),
# (2) cùng cặp marker nằm giữa dòng, (3) cả dòng có <!-- share:omit-line --> (hàng bảng markdown)
OMIT = re.compile(r"^[ \t]*<!--\s*share:omit\s*-->.*?<!--\s*/share:omit\s*-->[ \t]*(?:\n|\Z)"
                  r"|<!--\s*share:omit\s*-->.*?<!--\s*/share:omit\s*-->", re.S | re.M)
OMIT_LINE = re.compile(r"^.*<!--\s*share:omit-line\s*-->.*(?:\n|\Z)", re.M)


def share_policy(path=SHARE_POLICY_CONFIG):
    """Đọc share_policy.yaml. Trả {allow, docs, configs, text_not_redistributable: set, forbidden: [re.Pattern]}."""
    import yaml
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {"allow": list(cfg["allow"]), "docs": list(cfg["docs"]), "configs": list(cfg["configs"]),
            "text_not_redistributable": set(cfg.get("text_not_redistributable") or ()),
            "forbidden": [re.compile(p) for p in cfg["forbidden"]]}


def omit_share_blocks(text: str) -> str:
    """Bản docs cho gói share: bỏ các đoạn đánh dấu share:omit / share:omit-line (xem OMIT)."""
    return OMIT_LINE.sub("", OMIT.sub("", text))


def filtered_datasets_yaml(text: str, keep) -> str:
    """datasets.yaml chỉ giữ mục `datasets:` có trong `keep` (bỏ bộ riêng tư / không đóng gói). Giữ khối comment đầu file."""
    import yaml
    lines = text.splitlines(keepends=True)
    n = next((i for i, l in enumerate(lines) if l.strip() and not l.lstrip().startswith("#")), len(lines))
    cfg = yaml.safe_load(text)
    keep = set(keep)
    missing = sorted(keep - set(cfg["datasets"]))
    if missing:
        raise SystemExit(f"datasets.yaml thiếu bộ có trong benchmark được đóng gói: {missing}")
    cfg["datasets"] = {k: v for k, v in cfg["datasets"].items() if k in keep}
    return "".join(lines[:n]) + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=120)


def glob_re(g: str):
    """Glob đường dẫn -> regex: `*` / `?` không vượt qua `/`, `**/` khớp 0 hoặc nhiều thư mục."""
    out, i = "", 0
    while i < len(g):
        if g.startswith("**/", i):
            out, i = out + "(?:[^/]+/)*", i + 3
        elif g[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif g[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(g[i]), i + 1
    return re.compile(out + r"\Z")


def scan_package(root: Path, policy) -> list:
    """Vi phạm chính sách trong cây gói: file ngoài allowlist, và dòng text chạm regex cấm. Trả list chuỗi mô tả."""
    bad = []
    allow = [glob_re(g) for g in policy["allow"]]
    for p in sorted(x for x in Path(root).rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if not any(rx.match(rel) for rx in allow):
            bad.append(f"{rel}: không có trong allowlist")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad.append(f"{rel}: không phải text UTF-8")
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for rx in policy["forbidden"]:
                if rx.search(line):
                    bad.append(f"{rel}:{i}: chạm /{rx.pattern}/: {line.strip()[:100]}")
    return bad


def distinct_utterances(rows) -> int:
    """Số câu không trùng giữa mọi manifest của gói: hai dòng là một câu nếu chung mã clip gốc (CV / VIVOS, kể cả bản
    resample), chung audio_sha256, hoặc chung (họ bộ dữ liệu, upstream_id). Vd C1 là tập con của G0, CV17 ∩ CV27."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, r in enumerate(rows):
        me = find(("row", i))
        ck = clip_key(r)
        for k in (ck and f"clip:{ck}", r.get("audio_sha256") and f"sha:{r['audio_sha256']}",
                  r.get("upstream_id") and f"up:{dataset_family(r)}:{r['upstream_id']}"):
            if k:
                parent[find(k)] = me
    return len({find(("row", i)) for i in range(len(rows))})


def manifest_path(name: str) -> Path:
    import yaml
    bm = yaml.safe_load((P.REPO / "configs" / "bench" / "benchmarks" / f"{name}.yaml").read_text(encoding="utf-8"))
    return Path(bm["manifest"].replace("${data_root}", str(P.path("data_root")))), bm


def share_rows(rows):
    out = []
    for r in rows:
        if str(r.get("license") or "").lower() in PRIVATE_LICENSES:
            raise SystemExit(f"{r['id']}: license {r.get('license')} không được share")
        if r["dataset"] not in SOURCES:
            raise SystemExit(f"{r['id']}: bộ {r['dataset']} không có nguồn tải trong fetch.SOURCES")
        if not r.get("upstream_id") or not (r.get("audio_sha256") or r.get("audio_derivation") == "pinned_source_extract"):
            raise SystemExit(f"{r['id']}: thiếu upstream_id / audio_sha256")
        s = {k: v for k, v in r.items() if k != "audio" and not k.startswith("_")}
        s.setdefault("audio_derivation", "soxr_hq_16k_mono_pcm16" if r["dataset"] in DERIVED else "upstream_bytes")
        out.append(s)
    return sorted(out, key=lambda x: x["id"])


def strip_text(rows, datasets=()):
    """Bộ không cho phân phối lại lời thoại (text_redistributable = false, vd PhoAudiobook, GigaSpeech2), hoặc có trong
    `datasets` (share_policy.text_not_redistributable: license chưa rà xong): bỏ các trường lời thoại khỏi bản share;
    người nhận chạy fetch (đã đồng ý điều khoản trên HF) để lấy lại đúng giá trị từ nguồn. Không thêm trường nào vào dòng,
    nên manifest sau fetch vẫn trùng manifest_sha256 bản khoá."""
    from gipbench.dataset.ingest.evalsets import TEXT_FIELDS
    datasets = set(datasets)
    out = []
    for r in rows:
        drop = r.get("text_redistributable") is False or r.get("dataset") in datasets
        out.append({k: v for k, v in r.items() if not (drop and k in TEXT_FIELDS)})
    return out


# --- code chấm điểm kèm gói (--with-code) ---
# Gói chạy độc lập: người nhận không cần repo. Chỉ copy các module mà fetch / leak_audit / evaluate / transcribe import
# (tính bằng AST lúc đóng gói, nên không lệch khi code đổi), và DỪNG nếu có module thuộc phần train / dữ liệu train.
# Danh sách module nằm ở configs/dataset/share_code.yaml (tầng dataset không được nhắc tên module bench / model trong code).
SHARE_CODE_CONFIG = P.REPO / "configs" / "dataset" / "share_code.yaml"


def share_code(path=SHARE_CODE_CONFIG):
    """Đọc share_code.yaml (lúc gọi, không lúc import: import module này không cần pyyaml).
    Trả {roots: list, forbidden: tuple, configs: list, skip: set}."""
    import yaml
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {"roots": list(cfg["roots"]), "forbidden": tuple(cfg["forbidden"]),
            "configs": list(cfg["configs"]), "skip": set(cfg.get("skip") or ())}
# core/paths.py của repo có đường dẫn máy, khoá của pipeline train (icefall, exp_root) và file chặn bộ test-only:
# bản kèm gói chỉ có các khoá bench dùng.
CODE_PATHS = '''"""Đường dẫn gốc của gói code (REPO = thư mục chứa src/ và configs/) và thư mục dữ liệu / kết quả.

data_root và results_root đọc từ biến môi trường GIPBENCH_DATA / GIPBENCH_RESULTS, mặc định <REPO>/data và <REPO>/results.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_DEFAULTS = {"data_root": "data", "results_root": "results"}
_ENV = {"data_root": "GIPBENCH_DATA", "results_root": "GIPBENCH_RESULTS"}


def get(key):
    return os.environ.get(_ENV[key]) or _DEFAULTS[key]


def path(key):
    """Đường dẫn tuyệt đối: ~ được mở rộng, đường dẫn tương đối tính từ REPO."""
    p = Path(os.path.expanduser(str(get(key))))
    return p if p.is_absolute() else REPO / p
'''
CODE_PYPROJECT = """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "gipbench"
version = "{version}"
description = "Gipbench: code chấm điểm, tải audio và kiểm trùng dữ liệu cho bench ASR tiếng Việt"
requires-python = ">=3.10"
dependencies = ["numpy", "pyyaml"]

[project.optional-dependencies]
audio = ["soundfile", "soxr"]
asr = ["faster-whisper"]

[tool.setuptools.packages.find]
where = ["src"]
"""


def _module_file(src: Path, mod: str):
    p = src / Path(*mod.split("."))
    if p.with_suffix(".py").exists():
        return p.with_suffix(".py")
    if (p / "__init__.py").exists():
        return p / "__init__.py"
    return None


def code_closure(src: Path, roots=None, skip=None):
    """Tập module gipbench.* mà `roots` import (kể cả gián tiếp, và các package cha). Trả {tên: file}.
    Mặc định `roots` / `skip` lấy từ share_code.yaml."""
    import ast
    if roots is None or skip is None:
        cfg = share_code()
        roots = cfg["roots"] if roots is None else roots
        skip = cfg["skip"] if skip is None else skip
    seen, todo = {}, list(roots)
    while todo:
        m = todo.pop()
        if m in seen or m in skip:
            continue
        f = _module_file(src, m)
        if f is None:
            continue  # tên import không phải module (vd hàm / lớp)
        seen[m] = f
        for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                todo += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                todo += [n.module] + [f"{n.module}.{a.name}" for a in n.names]
        parts = m.split(".")
        todo += [".".join(parts[:i]) for i in range(1, len(parts))]
    return {m: f for m, f in seen.items() if m == "gipbench" or m.startswith("gipbench.")}


def copy_code(dst: Path, version: str, keep_datasets=None):
    """Chép code chấm điểm vào dst/code. `keep_datasets`: datasets.yaml kèm code chỉ giữ các bộ này (None: giữ nguyên)."""
    src = P.REPO / "src"
    cfg = share_code()
    mods = code_closure(src, cfg["roots"], cfg["skip"])
    bad = sorted(m for m in mods if m.startswith(cfg["forbidden"]))
    if bad:
        raise SystemExit(f"code chấm điểm import module không được share: {bad}")
    out = dst / "code"
    for m, f in sorted(mods.items()):
        t = out / "src" / f.relative_to(src)
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes((CODE_PATHS.encode("utf-8") if m == "gipbench.core.paths" else f.read_bytes()).replace(b"\r\n", b"\n"))
    for rel in cfg["configs"]:
        s = P.REPO / rel
        if not s.exists():
            raise SystemExit(f"thiếu {rel} để kèm trong code/")
        t = out / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        data = s.read_bytes().replace(b"\r\n", b"\n")
        if keep_datasets is not None and s.name == "datasets.yaml":
            data = filtered_datasets_yaml(data.decode("utf-8"), keep_datasets).encode("utf-8")
        t.write_bytes(data)
    (out / "pyproject.toml").write_text(CODE_PYPROJECT.format(version=version), encoding="utf-8", newline="\n")
    return sorted(mods)


def packaged_scorer_sha256(code_dir: Path) -> str:
    """scorer_code_sha256 của code kèm gói (cùng cách băm với evaluate: file theo share_code.scorer_files, \\r\\n -> \\n).
    Người nhận chấm bằng code này thì summary.json phải có đúng hash này."""
    import yaml
    files = yaml.safe_load(SHARE_CODE_CONFIG.read_text(encoding="utf-8"))["scorer_files"]
    return code_sha256([Path(code_dir) / "src" / "gipbench" / f for f in files])


def clean_meta(m):
    """run_meta.json cho bản share: bỏ đường dẫn máy (manifest, file model), giữ cấu hình giải mã + sha256."""
    if isinstance(m, dict):
        return {k: clean_meta(v) for k, v in m.items() if k not in ("path", "manifest", "dir", "encoder_dir")}
    if isinstance(m, list):
        return [clean_meta(x) for x in m]
    if isinstance(m, str) and (m.startswith("/") or (len(m) > 2 and m[1] == ":" and m[2] in "\\/")):
        return "<local path>"
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmarks", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", help="mặc định: <data_root>/share")
    ap.add_argument("--baselines", nargs="*", default=[], metavar="DIR",
                    help="thư mục kết quả <results>/<benchmark>/<hệ> (hyps.jsonl + run_meta.json): chép vào baselines/ để "
                         "người nhận so ghép cặp (evaluate --baseline-hyps); bỏ mọi trường đường dẫn máy")
    args = ap.parse_args(argv)
    dst = Path(args.out or P.path("data_root") / "share") / args.name
    if dst.exists():
        raise SystemExit(f"{dst} đã có: bản share không được ghi đè (đổi --name / version)")
    policy = share_policy()
    all_rows = []
    for b in args.benchmarks:  # kiểm trước khi ghi gì
        mp, bm = manifest_path(b)
        rows = share_rows(read_jsonl(str(mp)))
        if bm.get("evidence_level") == "locked" or bm.get("manifest_sha256"):
            # người nhận chấm bằng manifest share (+ đường dẫn audio): phải trùng bản khoá
            want, got = bm.get("manifest_sha256"), manifest_sha256(rows)
            if got != want:
                raise SystemExit(f"{b}: manifest share {got[:12]} khác bản khoá {str(want)[:12]}")
        if bm.get("catalog_from_bench"):
            raise SystemExit(f"{b}: catalog chứa span lấy từ chính bench, không share làm catalog chấm")
        if bm.get("catalog_sha256"):
            cat = Path(bm["catalog"].replace("${data_root}", str(P.path("data_root"))))
            if sha256_file(cat) != bm["catalog_sha256"]:
                raise SystemExit(f"{b}: catalog {cat} khác bản khoá")
        all_rows += rows
    used = {r["dataset"] for r in all_rows}
    no_text = policy["text_not_redistributable"] & used
    cannot = sorted(d for d in no_text if SOURCES[d]["kind"] != "evalset")
    if cannot:  # chỉ fetch_evalset lấy lại được lời thoại từ nguồn
        raise SystemExit(f"text_not_redistributable: fetch không lấy lại được lời thoại của {cannot}")
    tmp = dst.parent / f".{dst.name}.partial"  # dựng ở đây, kiểm chính sách xong mới đổi tên
    if tmp.exists():
        shutil.rmtree(tmp)  # thư mục tạm của lần đóng gói trước bị dừng giữa chừng
    try:
        summary = build(tmp, args, policy, sorted(used), no_text)
        release = {"name": args.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "distinct_utterances": distinct_utterances(all_rows),
                   "scorer_code_sha256": packaged_scorer_sha256(tmp / "code"),
                   "share_policy_sha256": sha256_file(SHARE_POLICY_CONFIG),
                   "text_withheld_datasets": sorted(no_text | {r["dataset"] for r in all_rows
                                                               if r.get("text_redistributable") is False}),
                   "benchmarks": summary}
        (tmp / "release.json").write_text(json.dumps(release, ensure_ascii=False, indent=1), encoding="utf-8")
        bad = scan_package(tmp, policy)
        if bad:
            raise SystemExit("gói vi phạm configs/dataset/share_policy.yaml (đã huỷ, không ghi gì):\n  "
                             + "\n  ".join(bad[:50]) + (f"\n  ... và {len(bad) - 50} chỗ khác" if len(bad) > 50 else ""))
        lines = []
        for p in sorted(x for x in tmp.rglob("*") if x.is_file()):
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(tmp).as_posix()}")
        (tmp / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    tmp.rename(dst)
    print(f"{dst}: {sum(v['rows'] for v in summary.values())} dòng, {release['distinct_utterances']} câu không trùng, "
          f"{len(lines)} file; chính sách share ĐẠT")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def build(dst: Path, args, policy, keep_datasets, no_text):
    """Ghi manifests / baselines / docs / configs / code / README vào dst. Trả summary theo benchmark."""
    (dst / "manifests").mkdir(parents=True)
    summary = {}
    for b in args.benchmarks:
        mp, bm = manifest_path(b)
        rows = share_rows(read_jsonl(str(mp)))
        name = b.replace("_draft", "")
        with open(dst / "manifests" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in strip_text(rows, no_text):
                f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        shutil.copyfile(P.REPO / "configs" / "bench" / "benchmarks" / f"{b}.yaml", dst / "manifests" / f"{name}.benchmark.yaml")
        if bm.get("catalog"):
            cat = Path(bm["catalog"].replace("${data_root}", str(P.path("data_root"))))
            shutil.copyfile(cat, dst / "manifests" / f"{name}.catalog.tsv")
        by = {}
        for r in rows:
            d = by.setdefault(r["dataset"], {"rows": 0, "hours": 0.0, "license": r.get("license")})
            d["rows"] += 1
            if r.get("duration"):
                d["hours"] = round(d["hours"] + r["duration"] / 3600, 4)
                d["_timed"] = True
        for d in by.values():  # bộ khoá chỉ bằng metadata chưa có duration: "chưa đo" (null), không phải 0 giờ
            if not d.pop("_timed", False):
                d["hours"] = None
        summary[name] = {"rows": len(rows), "evidence_level": bm.get("evidence_level"), "track": bm.get("track"),
                         "by_dataset": by}
    for bdir in args.baselines:
        bdir = Path(bdir)
        bench, system = bdir.parent.name.replace("_draft", ""), bdir.name
        if bench not in summary:
            raise SystemExit(f"{bdir}: benchmark '{bench}' không có trong --benchmarks")
        ids = {json.loads(l)["id"] for l in open(dst / "manifests" / f"{bench}.jsonl", encoding="utf-8") if l.strip()}
        hyps = [json.loads(l) for l in open(bdir / "hyps.jsonl", encoding="utf-8") if l.strip()]
        if {h["id"] for h in hyps} != ids or len(hyps) != len(ids):
            raise SystemExit(f"{bdir}: hyps không khớp đúng một dòng cho mỗi id của {bench}")
        out = dst / "baselines" / bench / system
        out.mkdir(parents=True)
        with open(out / "hyps.jsonl", "w", encoding="utf-8", newline="\n") as f:
            for h in hyps:
                f.write(json.dumps({"id": h["id"], "hyp": h.get("hyp", "")}, ensure_ascii=False) + "\n")
        meta = json.loads((bdir / "run_meta.json").read_text(encoding="utf-8"))
        (out / "run_meta.json").write_text(json.dumps(clean_meta(meta), ensure_ascii=False, indent=1), encoding="utf-8")
        if (bdir / "report.md").exists():
            shutil.copyfile(bdir / "report.md", out / "report.md")
        summary[bench].setdefault("baselines", []).append(system)
    for rel in policy["docs"]:
        (dst / "docs").mkdir(exist_ok=True)
        text = (P.REPO / rel).read_text(encoding="utf-8")
        (dst / "docs" / Path(rel).name).write_text(omit_share_blocks(text), encoding="utf-8", newline="\n")
    for rel in policy["configs"]:
        (dst / "configs").mkdir(exist_ok=True)
        data = (P.REPO / rel).read_bytes().replace(b"\r\n", b"\n")
        if Path(rel).name == "datasets.yaml":
            data = filtered_datasets_yaml(data.decode("utf-8"), keep_datasets).encode("utf-8")
        (dst / "configs" / Path(rel).name).write_bytes(data)
    # người nhận không cần repo
    print(f"code/: {len(copy_code(dst, args.name.rsplit('-v', 1)[-1], keep_datasets))} module")
    (dst / "README.md").write_text(README.replace("{name}", args.name).replace(
        "{summary}", "```json\n" + json.dumps(summary, ensure_ascii=False, indent=1) + "\n```"),
                                   encoding="utf-8")
    return summary


README = (Path(__file__).with_name("share_readme.md")).read_text(encoding="utf-8")


if __name__ == "__main__":
    main()
