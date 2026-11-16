#!/usr/bin/env python3
"""Dựng manifest G0 (general_v1_draft) và C0 (code_switch_v1_draft) từ split test CHÍNH THỨC đã có trên máy.

  python -m gipbench.dataset.release.build_general --which general     [--limit 20] [--out DIR]
  python -m gipbench.dataset.release.build_general --which code_switch [--limit 20] [--out DIR]

general     : VIVOS test (prompts.txt), Common Voice vi 27.0 test.tsv, G-Group {tech,edu,finance,pubadmin} test (parquet).
              Lấy MỌI câu của split (không lọc câu có từ Anh, khác converter B1/B5). Text = nhãn gốc của bên phát hành.
code_switch : ViMedCSS split hard (raw/vimedcss_hard/b3_candidates.jsonl, 658 câu), giữ text_tagged (Spec) và
              text_tagged_orig (nhãn gốc). Câu không còn span <en> vẫn giữ (khác v0-draft đã bỏ 134 câu).
Không tải gì, không sửa dữ liệu gốc: audio được THAM CHIẾU bằng đường dẫn tương đối theo manifest (G-Group: file mp3 do
converter đã ghi; thiếu thì tách bytes gốc từ parquet vào <out>/audio/). Mỗi dòng có audio_sha256. Ghi build_meta.json:
số câu so với số câu split công bố, câu bị loại kèm lý do, manifest_sha256 (cùng định nghĩa với gipbench.bench.registry).
"""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.decontam import sha256_file
from gipbench.core.manifest import read_jsonl, write_jsonl

EXPECTED = {"vivos": 760, "commonvoice_vi_27": 1593, "ggroup_tech": 508, "ggroup_edu": 495, "ggroup_finance": 529,
            "ggroup_pubadmin": 515, "vimedcss": 658}
LICENSE = {"vivos": "CC-BY-NC-SA-4.0", "commonvoice_vi_27": "CC0-1.0", "vimedcss": "CC-BY-4.0",
           "ggroup_tech": "CC-BY-NC-4.0", "ggroup_edu": "CC-BY-NC-4.0", "ggroup_finance": "CC-BY-NC-4.0",
           "ggroup_pubadmin": "CC-BY-NC-4.0"}


def cv_region(acc: str) -> str:
    a = (acc or "").lower()
    if "bắc" in a or "north" in a:
        return "north"
    if "trung" in a or "central" in a:
        return "central"
    if "nam" in a or "south" in a:
        return "south"
    return "unk"


def cv_gender(g: str) -> str:
    g = (g or "").lower()
    return "f" if g.startswith("female") else ("m" if g.startswith("male") else "u")


def audio_duration(path: str) -> float:
    import soundfile as sf
    i = sf.info(path)
    return round(i.frames / i.samplerate, 3)


def vivos(data: Path, limit: int):
    root = data / "dl" / "vivos" / "test"
    gender = dict(l.split() for l in open(root / "genders.txt", encoding="utf-8") if l.strip())
    for n, line in enumerate(open(root / "prompts.txt", encoding="utf-8")):
        if limit and n >= limit:
            break
        uid, text = line.rstrip("\n").split(" ", 1)
        spk = uid.split("_")[0]
        yield {"id": f"g0_vivos_{uid}", "_abs": root / "waves" / spk / f"{uid}.wav", "text_tagged": text.strip(),
               "speaker": f"vivos_{spk}", "source": f"vivos:{spk}", "upstream_id": uid, "dataset": "vivos",
               "upstream_split": "test", "domain": "read", "region": "unk", "gender": gender.get(spk, "u"),
               "channel": "wideband", "slice": "read"}


def commonvoice(data: Path, limit: int):
    root = data / "dl" / "cv-corpus-27.0-2026-09-11" / "vi"
    dur = {r["clip"]: int(r["duration[ms]"]) / 1000
           for r in csv.DictReader(open(root / "clip_durations.tsv", encoding="utf-8"), delimiter="\t")}
    rows = csv.DictReader(open(root / "test.tsv", encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)
    for n, r in enumerate(rows):
        if limit and n >= limit:
            break
        stem = r["path"].rsplit(".", 1)[0]
        yield {"id": f"g0_{stem}", "_abs": root / "clips" / r["path"], "text_tagged": r["sentence"].strip(),
               "speaker": f"cv_{r['client_id'][:16]}", "source": f"cv:{r['client_id'][:16]}", "upstream_id": r["path"],
               "dataset": "commonvoice_vi_27", "upstream_split": "test", "domain": "read",
               "region": cv_region(r.get("accents")), "gender": cv_gender(r.get("gender")), "age": r.get("age") or "unk",
               "channel": "wideband", "slice": "read_crowd", "_duration": dur.get(r["path"])}


def ggroup(data: Path, out: Path, limit: int):
    import pyarrow.parquet as pq
    for name in ("tech", "edu", "finance", "pubadmin"):
        pf = pq.ParquetFile(data / "dl" / "ggroup" / f"{name}.parquet")
        n = 0
        for rg in range(pf.num_row_groups):
            meta = pf.read_row_group(rg, columns=["id", "text", "duration"]).to_pylist()
            audio = None
            for i, r in enumerate(meta):
                if limit and n >= limit:
                    break
                n += 1
                uid = f"gg_{name}_{r['id']}"
                p = data / "raw" / "ggroup" / "audio" / name / f"{uid}.mp3"
                if not p.exists():  # tách bytes gốc (không mã hoá lại) từ parquet
                    if audio is None:
                        audio = pf.read_row_group(rg, columns=["audio"]).to_pylist()
                    p = out / "audio" / "ggroup" / name / f"{uid}.mp3"
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_bytes(audio[i]["audio"]["bytes"])
                yield {"id": f"g0_{uid}", "_abs": p, "text_tagged": (r["text"] or "").strip(), "speaker": f"g0_{uid}",
                       "source": f"ggroup:{name}:{r['id']}", "upstream_id": r["id"], "dataset": f"ggroup_{name}",
                       "upstream_split": "test", "domain": name, "region": "unk", "gender": "u",
                       "channel": "wideband", "slice": "spontaneous", "_duration": round(float(r["duration"]), 3)}


def vimedcss_hard(data: Path, limit: int):
    src = data / "raw" / "vimedcss_hard" / "b3_candidates.jsonl"
    for n, r in enumerate(read_jsonl(str(src))):
        if limit and n >= limit:
            break
        row = {k: v for k, v in r.items() if not k.startswith("_") and k not in ("audio", "split", "auto_tag", "review")}
        row.update({"id": f"c0_{r['id']}", "_abs": Path(r["_audio_path"]), "dataset": "vimedcss", "upstream_split": "hard",
                    # segment_id gốc của ViMedCSS (converter thêm tiền tố "vmc_"); fetch dùng để tìm audio trong parquet
                    "upstream_id": r["id"][4:] if r["id"].startswith("vmc_") else r["id"],
                    "slice": "medical_cs", "_duration": r.get("duration")})
        yield row


def manifest_sha256(rows) -> str:
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda x: str(x["id"])):
        h.update(json.dumps({k: v for k, v in r.items() if not k.startswith("_") and k != "audio"}, sort_keys=True,
                            ensure_ascii=False).encode("utf-8") + b"\n")
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["general", "code_switch"], required=True)
    ap.add_argument("--data-root", help="mặc định: data_root của core.paths (env GIPBENCH_DATA)")
    ap.add_argument("--out", help="mặc định: <data_root>/benchmarks/<which>_v1_draft")
    ap.add_argument("--limit", type=int, default=0, help="N câu đầu mỗi bộ (thử nhanh)")
    args = ap.parse_args()
    data = Path(args.data_root or P.path("data_root"))
    name = f"{args.which}_v1_draft"
    out = Path(args.out or data / "benchmarks" / name)
    out.mkdir(parents=True, exist_ok=True)
    gen = (list(vivos(data, args.limit)) + list(commonvoice(data, args.limit)) + list(ggroup(data, out, args.limit))
           if args.which == "general" else list(vimedcss_hard(data, args.limit)))
    split = "G0" if args.which == "general" else "C0"
    rows, dropped = [], []
    for r in gen:
        p = Path(r.pop("_abs"))
        d = r.pop("_duration", None)
        if not p.exists():
            dropped.append({"id": r["id"], "reason": "không có file audio"})
            continue
        if not r.get("text_tagged"):
            dropped.append({"id": r["id"], "reason": "text rỗng"})
            continue
        r.update({"audio": os.path.relpath(p, out).replace(os.sep, "/"), "audio_sha256": sha256_file(p),
                  "duration": d if d else audio_duration(str(p)), "split": split, "track": split,
                  "license": LICENSE.get(r["dataset"]), "annotation_status": "upstream_human",
                  "bench_version": name.replace("_", "-")})
        rows.append(r)
    dup = [i for i, c in Counter(r["id"] for r in rows).items() if c > 1]
    if dup:
        raise SystemExit(f"id trùng: {dup[:5]}")
    write_jsonl(str(out / "manifest.jsonl"), rows)
    by = Counter(r["dataset"] for r in rows)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=P.REPO, capture_output=True,
                                text=True).stdout.strip() or None
    except Exception:
        commit = None
    meta = {"name": name, "builder": "gipbench.dataset.release.build_general", "args": vars(args), "git_commit": commit,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": len(rows),
            "hours": round(sum(r["duration"] for r in rows) / 3600, 4),
            "by_dataset": {k: {"rows": v, "expected": EXPECTED.get(k),
                               "hours": round(sum(r["duration"] for r in rows if r["dataset"] == k) / 3600, 4),
                               "speakers": len({r["speaker"] for r in rows if r["dataset"] == k})}
                           for k, v in sorted(by.items())},
            "dropped": dropped, "manifest_sha256": manifest_sha256(rows)}
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {len(rows)} câu, {meta['hours']:.2f} h, loại {len(dropped)} -> {out}/manifest.jsonl")
    for k, v in meta["by_dataset"].items():
        flag = "" if args.limit or v["expected"] in (None, v["rows"]) else f"  <-- KHÁC số công bố {v['expected']}"
        print(f"  {k:20s} {v['rows']:5d} câu {v['hours']:.2f} h, {v['speakers']} speaker{flag}")
    print(f"manifest_sha256 {meta['manifest_sha256']}")


if __name__ == "__main__":
    main()
