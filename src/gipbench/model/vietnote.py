"""Acquire the pinned VietNote ASR model and upstream recipe."""

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath

from gipbench.core.download import download_file, verify_file
from gipbench.core.paths import REPO

LOCK = REPO / "configs/model/vietnote.json"


def load_checkpoint(checkpoint):
    import torch

    # Treat training metadata paths as values even when they come from another OS.
    allowed = [(PurePosixPath, "pathlib.PosixPath"), (PureWindowsPath, "pathlib.WindowsPath"),
               torch.torch_version.TorchVersion]
    with torch.serialization.safe_globals(allowed):
        return torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)


def verify_tokenizer(root, vocab_size):
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as model_pb

    sp = spm.SentencePieceProcessor(model_file=str(root / "bpe.model"))
    vocab = {
        int(index): piece for piece, index in
        (line.rsplit(maxsplit=1) for line in (root / "tokens.txt").read_text(encoding="utf-8").splitlines())
    }
    if len(vocab) != vocab_size or sp.get_piece_size() != vocab_size:
        raise ValueError("VietNote vocabulary size differs from the model lock")
    if any(vocab.get(i) != sp.id_to_piece(i) for i in range(vocab_size)):
        raise ValueError("VietNote tokens.txt and SentencePiece disagree")
    spec = model_pb.ModelProto.FromString((root / "bpe.model").read_bytes()).normalizer_spec
    return {"blank_id": sp.piece_to_id("<blk>"), "unk_id": sp.unk_id(),
            "sentencepiece_normalizer": {"name": spec.name, "add_dummy_prefix": spec.add_dummy_prefix,
                                         "remove_extra_whitespaces": spec.remove_extra_whitespaces,
                                         "escape_whitespaces": spec.escape_whitespaces}}


def acquire(cfg, root, reserve, download, jobs):
    if download:
        root.mkdir(parents=True, exist_ok=True)
    sources = (
        (root, cfg["model_files"], f"https://huggingface.co/{cfg['model']}/resolve/{cfg['revision']}"),
        (root / "upstream", cfg["recipe_files"],
         f"https://raw.githubusercontent.com/{cfg['upstream']}/{cfg['upstream_commit']}"),
    )
    files = [(base, spec, url) for base, specs, url in sources for spec in specs if "same_as" not in spec]
    if download:
        remaining = sum(spec["bytes"] for base, spec, _ in files if not (base / spec["path"]).exists())
        if shutil.disk_usage(root).free < reserve + remaining:
            raise OSError("Insufficient space for selected VietNote artifacts")

    def fetch(item):
        base, spec, url = item
        path = base / spec["path"]
        print(f"{'Acquire' if download else 'Verify'} {path.relative_to(root)}", flush=True)
        digest = (download_file(f"{url}/{spec['path']}", path, spec, reserve)
                  if download else verify_file(path, spec))
        hashes = {path.relative_to(root).as_posix(): digest}
        for alias in aliases.get(path, []):
            dest = base / alias["path"]
            if download:
                dest.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(path, dest)
                except FileExistsError:
                    pass
            hashes[dest.relative_to(root).as_posix()] = verify_file(dest, {**alias, "sha256": digest})
        return hashes

    aliases = {}
    for base, specs, _ in sources:
        for spec in specs:
            if "same_as" in spec:
                aliases.setdefault(base / spec["same_as"], []).append(spec)
    files.sort(key=lambda item: (item[0] != root / "upstream" and "/int8/" not in item[1]["path"],
                                 item[1]["bytes"]))
    hashes = {}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        for result in executor.map(fetch, files):
            hashes.update(result)
    recipe = root / "upstream/docs/FINETUNE.md"
    if cfg["icefall_commit"] not in recipe.read_text(encoding="utf-8"):
        raise ValueError("The Icefall pin differs from the upstream fine-tuning guide")
    card = {k: v for k, v in cfg.items() if not k.endswith("_files")}
    card.update(verify_tokenizer(root, cfg["vocab_size"]))
    card.update(schema="gipbench.model_card/1", sha256=hashes)
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPO / "models/vietnote78")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--variant", choices=("streaming/int8", "streaming/fp32", "offline/int8", "offline/fp32"))
    args = parser.parse_args()
    cfg = json.loads(LOCK.read_text(encoding="utf-8"))
    if args.variant:
        refs = {"tokens.txt", "NOTICE", "README.md", "bpe.model"}
        for spec in cfg["model_files"]:
            if spec["path"].startswith(args.variant + "/"):
                refs.update((spec["path"], spec.get("same_as", spec["path"])))
        cfg["model_files"] = [spec for spec in cfg["model_files"] if spec["path"] in refs]
        cfg["variant"] = args.variant
    if args.dry_run:
        size = sum(file["bytes"] for file in cfg["model_files"] + cfg["recipe_files"] if "same_as" not in file)
        print(f"{cfg['model']}@{cfg['revision']}: {size:,} bytes before reuse; destination {args.out}")
        return
    card = acquire(cfg, args.out, int(args.reserve_gib * 2**30), download=not args.verify, jobs=args.jobs)
    suffix = "." + args.variant.replace("/", "-") if args.variant else ""
    path = args.out / f"model_card{suffix}.json"
    data = json.dumps(card, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        immutable_fields = ("model", "revision", "upstream_commit", "icefall_commit", "sha256")
        if any(saved.get(key) != card[key] for key in immutable_fields):
            raise ValueError(f"Existing model card differs: {path}")
    elif not args.verify:
        with path.open("x", encoding="utf-8") as out:
            out.write(data)
    print(f"Verified {len(card['sha256'])} paths; blank={card['blank_id']}; unk={card['unk_id']}")


if __name__ == "__main__":
    main()
