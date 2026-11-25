"""Compare official and checkpoint-derived VietNote artifacts on a frozen dev subset."""

import argparse
import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import yaml

from gipbench.bench.evaluate import score_rows, scorer_code_sha256, summarize
from gipbench.core import release as R
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.normprofiles import get_profile
from gipbench.core.textnorm import Catalog
from gipbench.infer.adapters import build_adapter
from gipbench.model.vietnote import LOCK
from vietnote_streaming_probe import probe

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/model/experiments/vietnote_validation.yaml"


def freeze(spec, out):
    rows = {}
    for item in spec["sources"]:
        source = read_jsonl(ROOT / item["path"])
        if manifest_sha256(source) != item["sha256"]:
            raise ValueError(f"Stale parity subset: {item['path']}")
        rows.update((row["id"], row) for row in source)
    release = R.resolve("dev/dev_v2_vi@1")
    candidates = [row for row in read_jsonl(release.manifest) if row["id"] not in rows
                  and float(row["duration"]) >= spec["long_min_seconds"]]
    candidates.sort(key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())
    speakers = set()
    for row in candidates:
        speaker = row.get("speaker_cluster") or row["speaker"]
        if speaker not in speakers:
            rows[row["id"]] = row
            speakers.add(speaker)
        if len(speakers) == spec["long_rows_min"]:
            break
    if len(speakers) < spec["long_rows_min"]:
        raise ValueError("Too few additional long clips with distinct speakers")
    keep = list(rows.values())
    for row in keep:
        actual = R.sha256(row["_audio_path"])
        if row.get("audio_sha256") and actual != row["audio_sha256"]:
            raise ValueError(f"Audio digest mismatch: {row['id']}")
        row["audio_sha256"] = actual
        row["audio"] = os.path.relpath(row["_audio_path"], out.parent).replace(os.sep, "/")
    write_jsonl(out, keep)
    return keep, R.sha256(release.manifest)


def decode(mode, precision, model, rows, out):
    cfg = {"engine": "sherpa_transducer", "system": out.stem,
           "model": {"id": "VietNote-Zipformer-78M", "dir": str(model.resolve())},
           "inference_mode": "open_vocab",
           "decode": {"streaming": mode == "streaming", "quantize": precision, "threads": 4,
                      "decoding_method": "modified_beam_search", "beam": 4, "tail_pad_ms": 660,
                      "feed_ms": 100, "case": "lower", "blank_penalty": 0.0}, "batch": 1}
    adapter = build_adapter(cfg, out.parent)
    adapter.load()
    if mode == "streaming":
        records = probe(adapter, rows)
    else:
        records = [{"id": row["id"], "hyp": adapter.transcribe_batch([row["_audio_path"]])[0]} for row in rows]
    write_jsonl(out, records)
    return records


def compare(before, after, spec, precision, catalog, rows):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or any([row["id"] for row in records] != ids for records in (before, after)):
        raise ValueError("Hypotheses must cover each locked utterance once and in manifest order")

    def score(records):
        hyps = {row["id"]: row["hyp"] for row in records}
        recs = score_rows(rows, hyps, catalog, get_profile("gipbench_strict_v1"), "text_tagged", "speaker", [])
        return summarize(recs, 0, 0, with_ci=False)

    a, b = score(before), score(after)
    mismatches = [x["id"] for x, y in zip(before, after, strict=True) if x["hyp"] != y["hyp"]]
    result = {"exact": len(rows) - len(mismatches), "rows": len(rows), "mismatches": mismatches,
              **{key: [a[key], b[key]] for key in ("wer", "cs_wer", "term_recall")}}
    passed = result["exact"] / len(rows) >= spec[f"{precision}_exact_min"]
    if precision == "int8":
        passed &= (abs(b["wer"] - a["wer"]) <= spec["int8_wer_delta_max_pp"]
                   and abs(b["cs_wer"] - a["cs_wer"]) <= spec["int8_cs_wer_delta_max_pp"])
    if "emissions" in before[0]:
        long = [(x, y) for x, y in zip(before, after, strict=True) if x["duration"] >= spec["long_min_seconds"]]
        shifts = [abs(x["emissions"][0]["audio_sec"] - y["emissions"][0]["audio_sec"])
                  for x, y in long if x["emissions"] and y["emissions"]]
        lost = sum(x["changed_after_halfway"] and not y["changed_after_halfway"] for x, y in long)
        shift = max(shifts, default=None)
        result.update(long_rows=len(long), long_rows_with_emissions=len(shifts), lost_late_emissions=lost,
                      max_first_emission_shift_seconds=shift)
        passed &= (len(long) >= spec["long_rows_min"] and len(shifts) == len(long)
                   and shift <= spec["max_first_emission_shift_seconds"] and lost <= spec["max_lost_late_emissions"])
    return dict(result, passed=bool(passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--locked-parity", type=Path, help="Reuse the exact manifest and order from a prior parity run")
    parser.add_argument("--rescore", action="store_true", help="Recompute summaries from the verified saved hypotheses")
    args = parser.parse_args()
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    spec = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))["parity"]
    exported = json.loads((args.export / "export.json").read_text(encoding="utf-8"))
    base = ROOT / "models/vietnote78"
    original = R.sha256(base / "pytorch/pretrained.pt")
    if exported["model_revision"] != card["revision"] or exported["checkpoint_sha256"] != original:
        raise ValueError("Re-export has a different checkpoint or revision")
    for name, expected in exported["files"].items():
        if R.sha256(args.export / name) != expected:
            raise ValueError(f"Stale export file: {name}")
    if args.rescore:
        old = json.loads((args.out / "result.json").read_text(encoding="utf-8"))
        if old["export_metadata_sha256"] != R.sha256(args.export / "export.json"):
            raise ValueError("Saved hypotheses used a different export")
        for name, digest in old["outputs"].items():
            if R.sha256(args.out / name) != digest:
                raise ValueError(f"Stale saved hypotheses: {name}")
        rows, parent_sha = read_jsonl(args.out / "dev.jsonl"), old["long_parent_sha256"]
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        if args.locked_parity:
            parent = json.loads((args.locked_parity / "result.json").read_text(encoding="utf-8"))
            manifest = args.locked_parity / "dev.jsonl"
            rows = read_jsonl(manifest)
            if (R.sha256(manifest) != parent["outputs"]["dev.jsonl"] or len(rows) != 113
                    or manifest_sha256(rows) != parent["dev_manifest_sha256"]
                    or R.sha256(PROTOCOL) != parent["config_sha256"]):
                raise ValueError("The locked V01 manifest or protocol changed")
            for row in rows:
                if R.sha256(row["_audio_path"]) != row["audio_sha256"]:
                    raise ValueError(f"Audio digest mismatch: {row['id']}")
                row["audio"] = os.path.relpath(row["_audio_path"], args.out).replace(os.sep, "/")
            write_jsonl(args.out / "dev.jsonl", rows)
            parent_sha = parent["long_parent_sha256"]
        else:
            rows, parent_sha = freeze(spec, args.out / "dev.jsonl")
    catalog_path = R.resolve("catalog/train-v1").catalog
    catalog = Catalog.load(str(catalog_path))
    official, results = {}, {}
    for mode in ("offline", "streaming"):
        for precision in ("fp32", "int8"):
            ident = f"{mode}_{precision}"
            original_dir = base / mode / precision
            suffix = ".int8.onnx" if precision == "int8" else ".onnx"
            filenames = [*(f"{part}{suffix}" for part in ("encoder", "decoder", "joiner")), "tokens.txt"]
            file_hashes = {name: R.sha256(original_dir / name) for name in filenames}
            official.update({f"{mode}/{precision}/{name}": digest for name, digest in file_hashes.items()})
            identical = all(file_hashes[name] == exported["files"][f"{mode}/{name}"] for name in filenames)
            a_path, b_path = (args.out / f"{ident}_{side}.jsonl" for side in ("official", "reexport"))
            if args.rescore:
                if any(old["official_files"][f"{mode}/{precision}/{name}"] != digest
                       for name, digest in file_hashes.items()):
                    raise ValueError(f"Saved hypotheses used different official artifacts: {ident}")
                a, b = read_jsonl(a_path), read_jsonl(a_path if identical else b_path)
            else:
                a = decode(mode, precision, original_dir, rows, a_path)
                b = a if identical else decode(mode, precision, args.export / mode, rows, b_path)
            results[ident] = dict(compare(a, b, spec, precision, catalog, rows), identical_files=identical)
            print(f"{ident}: {results[ident]}", flush=True)
    report = {"passed": all(result["passed"] for result in results.values()), "comparisons": results,
              "config_sha256": R.sha256(PROTOCOL), "model_revision": card["revision"], "checkpoint_input_sha256": original,
              "dev_manifest_sha256": manifest_sha256(rows), "long_parent_sha256": parent_sha,
              "export_metadata_sha256": R.sha256(args.export / "export.json"), "official_files": official,
              "outputs": {p.name: R.sha256(p) for p in args.out.glob("*.jsonl")},
              "catalog_sha256": R.sha256(catalog_path), "scorer_code_sha256": scorer_code_sha256()}
    if args.rescore:
        if "runtime" in old:
            report["runtime"] = old["runtime"]
    else:
        report["runtime"] = {"platform": platform.platform(), "python": platform.python_version(),
                             "sherpa-onnx": version("sherpa-onnx")}
    (args.out / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
