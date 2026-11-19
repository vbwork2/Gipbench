"""Command-line interface for a standalone Gipbench Evaluation Kit."""

import argparse
import json
from pathlib import Path

from gipbench.bench import evaluate, registry
from gipbench.core.manifest import read_jsonl
from gipbench.dataset.release import fetch, leak_audit


def verify(config, manifest, require_audio_sha):
    benchmark = registry.load_benchmark(str(config))
    rows = read_jsonl(str(manifest))
    error = registry.check_frozen(benchmark, rows)
    missing, mismatched, unhashed = [], [], []
    for row in rows:
        path = row.get("_audio_path")
        expected = row.get("audio_sha256")
        actual = registry.file_sha256(path) if path and Path(path).is_file() and expected else None
        if not path or not Path(path).is_file():
            missing.append(row["id"])
        elif not expected:
            unhashed.append(row["id"])
        elif actual != expected:
            mismatched.append(row["id"])
    print(json.dumps({"rows": len(rows), "manifest_error": error, "missing_audio": missing,
                      "mismatched_audio": mismatched, "audio_without_hash": unhashed}, indent=2))
    return int(bool(not rows or error or missing or mismatched or (require_audio_sha and unhashed)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kit", type=Path, default=Path.cwd(), help="Directory containing the kit manifests")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("fetch")
    download.add_argument("--out", required=True)
    download.add_argument("--cv-root")
    download.add_argument("--datasets", nargs="+")
    download.add_argument("--local-source", nargs="+")
    verify_cmd = commands.add_parser("verify")
    verify_cmd.add_argument("--require-audio-sha", action="store_true")
    audit = commands.add_parser("decontam")
    audit.add_argument("--train", nargs="+", required=True)
    audit.add_argument("--dev", nargs="+")
    audit.add_argument("--text-files", nargs="+")
    audit.add_argument("--require-audio-sha", action=argparse.BooleanOptionalAction, default=True)
    audit.add_argument("--out")
    for name in ("score", "compare"):
        score = commands.add_parser(name)
        score.add_argument("--hyps", required=True)
        score.add_argument("--run-meta")
        score.add_argument("--mode", choices=("final", "diagnostic"), default="final")
        score.add_argument("--out")
        score.add_argument("--baseline", required=name == "compare")
    for name in ("fetch", "verify", "decontam", "score", "compare"):
        commands.choices[name].add_argument("--benchmark", required=True)
        if name != "fetch":
            commands.choices[name].add_argument("--data", type=Path, required=name == "verify",
                                                help="Fetched directory or manifest.jsonl")
    card = commands.add_parser("card")
    card.add_argument("--summary", type=Path, required=True)
    card.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.command == "card":
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        evaluate.write_report(args.out or args.summary.with_name("report.md"), summary)
        return 0

    name = args.benchmark
    if not name or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for char in name):
        parser.error("Use a benchmark name from the kit manifests directory")
    root = args.kit / "manifests"
    config = root / f"{name}.benchmark.yaml"
    bundled = root / f"{name}.jsonl"
    if not config.is_file() or not bundled.is_file():
        parser.error(f"Benchmark {name} is absent from {root}")
    if args.command == "fetch":
        command = ["--manifest", str(bundled), "--out", args.out]
        if args.cv_root:
            command += ["--cv-root", args.cv_root]
        if args.datasets:
            command += ["--datasets", *args.datasets]
        if args.local_source:
            command += ["--local-source", *args.local_source]
        return fetch.main(command)

    manifest = args.data or bundled
    if manifest.is_dir():
        manifest /= "manifest.jsonl"
    if args.command == "verify":
        return verify(config, manifest, args.require_audio_sha)
    if args.command == "decontam":
        command = ["--test", str(manifest), "--train", *args.train,
                   "--require-audio-sha" if args.require_audio_sha else "--no-require-audio-sha"]
        if args.dev:
            command += ["--dev", *args.dev]
        if args.text_files:
            command += ["--text-files", *args.text_files]
        if args.out:
            command += ["--out", args.out]
        return leak_audit.main(command)

    command = ["--benchmark", str(config), "--manifest", str(manifest), "--hyps", args.hyps, "--mode", args.mode]
    catalog = root / f"{name}.catalog.tsv"
    if catalog.is_file():
        command += ["--catalog", str(catalog)]
    for flag, value in (("--baseline-hyps", args.baseline), ("--run-meta", args.run_meta), ("--out", args.out)):
        if value:
            command += [flag, value]
    evaluate.main(command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
