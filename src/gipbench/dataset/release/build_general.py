#!/usr/bin/env python3
"""Dựng manifest G0 (general_v1_draft) và C0 (code_switch_v1_draft) từ split test CHÍNH THỨC đã có trên máy.

  python -m gipbench.dataset.release.build_general --which general     [--limit 20] [--out DIR]
  python -m gipbench.dataset.release.build_general --which code_switch [--limit 20] [--out DIR]

general     : VIVOS test (prompts.txt), Common Voice vi 27.0 test.tsv, G-Group {tech,edu,finance,pubadmin} test (parquet).
              Lấy MỌI câu của split (không lọc câu có từ Anh, khác converter B1/B5). Text = nhãn gốc của bên phát hành.
code_switch : ViMedCSS split hard (raw/vimedcss_hard/b3_candidates.jsonl, 658 câu), giữ text_tagged (Spec) và
              text_tagged_orig (nhãn gốc). Câu không còn span <en> vẫn giữ (khác v0-draft đã bỏ 134 câu).
external    : track E0 = các bộ test công khai bổ sung đã nhập bằng gipbench.dataset.ingest.evalsets
              (<data_root>/dl/evalsets/<bộ>/rows.jsonl): FLEURS vi, Common Voice 17 vi, VietMed, LSVSC, ViMD,
              Speech-MASSIVE vi (dev), ViMedCSS test. Lời thoại = nhãn gốc. ViMedCSS test: text_tagged = lời gốc + tag <en>
              quanh đúng các thuật ngữ trong cs_terms_list của dataset (để chấm CS-WER), text_orig = lời gốc.
cs_ggroup   : track C1 = câu G-Group test có từ Anh, tag <en> theo quyết định thuật ngữ người duyệt
              (configs/dataset/labeling/term_decisions.tsv) + phiếu duyệt từng câu. Đầu vào: manifest do build_bench ghi
              (benchmarks/cs_ggroup_v1_draft/manifest_built.jsonl, lệnh trong docs/dataset_registry.md). Lời thoại = nhãn
              gốc G-Group (text_orig), chỉ thêm tag.
Không tải gì, không sửa dữ liệu gốc: audio được THAM CHIẾU bằng đường dẫn tương đối theo manifest (G-Group: file mp3 do
converter đã ghi; thiếu thì tách bytes gốc từ parquet vào <out>/audio/). Mỗi dòng có audio_sha256. Ghi build_meta.json:
số câu so với số câu split công bố, câu bị loại kèm lý do, manifest_sha256 (cùng định nghĩa với gipbench.bench.registry).
"""
import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from gipbench.core import paths as P
from gipbench.core.fileio import sha256_file
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.dataset.release.fetch import DERIVED

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


def cs_ggroup(data: Path, limit: int):
    import pyarrow.parquet as pq
    built = data / "benchmarks" / "cs_ggroup_v1_draft" / "manifest_built.jsonl"
    src = list(read_jsonl(str(built)))
    orig = {}
    for name in sorted({r["domain"] for r in src}):
        t = pq.read_table(data / "dl" / "ggroup" / f"{name}.parquet", columns=["id", "text"]).to_pylist()
        orig.update({f"gg_{name}_{x['id']}": (x["id"], (x["text"] or "").strip()) for x in t})
    for n, r in enumerate(src):
        if limit and n >= limit:
            break
        if r.get("split") != "B1" or not r.get("trusted_text"):
            raise SystemExit(f"{built}: dòng lạ {r['id']} (cần B1, trusted_text)")
        up, text_orig = orig[r["id"]]
        name = r["domain"]
        yield {"id": f"c1_{r['id']}", "_abs": (built.parent / r["audio"]).resolve(), "text_tagged": r["text_tagged"],
               "text_orig": text_orig, "speaker": f"c1_{r['id']}", "source": f"ggroup:{name}:{up}", "upstream_id": up,
               "dataset": f"ggroup_{name}", "upstream_split": "test", "domain": name, "region": "unk", "gender": "u",
               "channel": "wideband", "slice": "spontaneous_cs",
               # term_decisions = tag theo quyết định thuật ngữ; ok / fix = người duyệt cả câu
               "tag_review": r.get("review"),
               "_duration": r.get("duration")}


MODE: dict = {}  # bộ E0 -> "audio" | "metadata_only"
EXTERNAL_SETS = ["fleurs_vi", "cv17_vi", "vietmed", "lsvsc", "vimd", "speech_massive_vi", "speech_massive_test_vi",
                 "vimedcss_test", "phoaudiobook", "bud500", "gigaspeech2_vi"]


def external(data: Path, limit: int):
    from gipbench.core.tagging import auto_tag
    from gipbench.dataset.ingest.evalsets import SPECS
    for key in EXTERNAL_SETS:
        base = data / "dl" / "evalsets" / key
        # rows.jsonl = đã tách audio (có audio_sha256, duration đo được); rows_meta.jsonl = chỉ metadata (--metadata-only)
        src = base / "rows.jsonl" if (base / "rows.jsonl").exists() else base / "rows_meta.jsonl"
        if not src.exists():
            print(f"  (bỏ qua {key}: chưa nhập, chạy gipbench.dataset.ingest.evalsets --sets {key} [--metadata-only])")
            continue
        MODE[key] = "audio" if src.name == "rows.jsonl" else "metadata_only"
        EXPECTED[key] = SPECS[key].get("expected")
        LICENSE[key] = SPECS[key]["license"]
        for n, r in enumerate(read_jsonl(str(src))):
            if limit and n >= limit:
                break
            row = {k: v for k, v in r.items() if k not in ("audio", "audio_sha256", "duration") and not k.startswith("_")}
            row.update({"_abs": (base / r["audio"]) if r.get("audio") else None, "_duration": r.get("duration"),
                        "slice": key})
            if key == "vimedcss_test":
                terms = [t.strip() for t in str(r.get("cs_terms_list") or "").split(";") if t.strip()]
                # chỉ thêm tag <en> quanh đúng thuật ngữ của dataset (không đổi chữ, không đổi WER); text_orig = bản gốc
                row["text_orig"] = r["text_tagged"]
                row["text_tagged"] = auto_tag(r["text_tagged"], None, terms, vi_words=set(), heuristic=False,
                                              term_dec={})["text_tagged"]
            yield row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--which", choices=["general", "code_switch", "cs_ggroup", "external"], required=True)
    ap.add_argument("--data-root", help="mặc định: data_root của core.paths (env GIPBENCH_DATA)")
    ap.add_argument("--name", help="tên benchmark (mặc định <which>_v1_draft); bản khoá: general_v1 / code_switch_v1 / cs_ggroup_v1")
    ap.add_argument("--bench-version", help="giá trị bench_version mỗi dòng (mặc định = tên, '_' thành '-')")
    ap.add_argument("--out", help="mặc định: <data_root>/benchmarks/<name>")
    ap.add_argument("--limit", type=int, default=0, help="N câu đầu mỗi bộ (thử nhanh)")
    args = ap.parse_args()
    data = Path(args.data_root or P.path("data_root"))
    name = args.name or f"{args.which}_v1_draft"
    out = Path(args.out or data / "benchmarks" / name)
    out.mkdir(parents=True, exist_ok=True)
    if args.which == "general":
        gen = list(vivos(data, args.limit)) + list(commonvoice(data, args.limit)) + list(ggroup(data, out, args.limit))
    elif args.which == "code_switch":
        gen = list(vimedcss_hard(data, args.limit))
    elif args.which == "cs_ggroup":
        gen = list(cs_ggroup(data, args.limit))
    else:
        gen = list(external(data, args.limit))
    split = {"general": "G0", "code_switch": "C0", "cs_ggroup": "C1", "external": "E0"}[args.which]
    rows, dropped = [], []
    for r in gen:
        a = r.pop("_abs")
        d = r.pop("_duration", None)
        if not r.get("text_tagged"):
            dropped.append({"id": r["id"], "reason": "text rỗng"})
            continue
        if a is None:  # dòng metadata-only (E0): không có audio cục bộ; fetch kiểm bằng sha256 file nguồn
            r.update({"split": split, "track": split, "license": LICENSE.get(r["dataset"]),
                      "annotation_status": "upstream_human", "bench_version": args.bench_version or name.replace("_", "-")})
            if d:
                r["duration"] = d
            rows.append(r)
            continue
        p = Path(a)
        if not p.exists():
            dropped.append({"id": r["id"], "reason": "không có file audio"})
            continue
        r.update({"audio": os.path.relpath(p, out).replace(os.sep, "/"), "audio_sha256": sha256_file(p),
                  # cách fetch tái dựng audio; nằm trong manifest để bản share có cùng manifest_sha256
                  "audio_derivation": r.get("audio_derivation") or ("soxr_hq_16k_mono_pcm16" if r["dataset"] in DERIVED
                                                                     else "upstream_bytes"),
                  "duration": d if d else audio_duration(str(p)), "split": split, "track": split,
                  "license": LICENSE.get(r["dataset"]), "annotation_status": "upstream_human",
                  "bench_version": args.bench_version or name.replace("_", "-")})
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
            "hours": round(sum(r.get("duration") or 0 for r in rows) / 3600, 4),
            "by_dataset": {k: {"rows": v, "expected": None if args.which == "cs_ggroup" else EXPECTED.get(k),
                               "hours": round(sum(r.get("duration") or 0 for r in rows if r["dataset"] == k) / 3600, 4), "mode": MODE.get(k, "audio"),
                               "speakers": len({r["speaker"] for r in rows if r["dataset"] == k})}
                           for k, v in sorted(by.items())},
            "dropped": dropped, "manifest_sha256": manifest_sha256(rows)}
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{name}: {len(rows)} câu, {meta['hours']:.2f} h, loại {len(dropped)} -> {out}/manifest.jsonl")
    for k, v in meta["by_dataset"].items():
        flag = "" if args.limit or args.which == "cs_ggroup" or v["expected"] in (None, v["rows"]) else f"  <-- KHÁC số công bố {v['expected']}"
        print(f"  {k:22s} {v['rows']:5d} câu {v['hours']:.2f} h, {v['speakers']} speaker, {v['mode']}{flag}")
    print(f"manifest_sha256 {meta['manifest_sha256']}")


if __name__ == "__main__":
    main()
