"""Run the pinned VietNote recipe with GipBench's prepared CutSets."""

import argparse
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

from gipbench.core.download import verify_file
from gipbench.model.families import family
from gipbench.model.recipe import experiment
from gipbench.model.vietnote import LOCK, verify_tokenizer


def model_args(card, export=False):
    arch = card["architecture"]
    values = {
        "num-encoder-layers": arch["num_encoder_layers"], "encoder-dim": arch["encoder_dim"],
        "feedforward-dim": arch["feedforward_dim"], "decoder-dim": arch["decoder_dim"],
        "joiner-dim": arch["joiner_dim"], "causal": 1,
        "chunk-size": arch["streaming_export_chunk_size"] if export else arch["train_chunk_sizes"],
        "left-context-frames": (arch["streaming_export_left_context_frames"] if export
                                else arch["train_left_context_frames"]),
    }
    return [item for key, value in values.items()
            for item in (f"--{key}", ",".join(map(str, value)) if isinstance(value, list) else str(value))]


def icefall_source(card, source):
    if source.is_file():
        verify_file(source, card["icefall_archive"])
        extracted = source.parent / f"icefall-{card['icefall_commit']}"
        if not extracted.exists():
            with tarfile.open(source) as archive:
                archive.extractall(source.parent, filter="data")
        return extracted
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != card["icefall_commit"] or not (source / ".git").exists():
        raise ValueError(f"Icefall must be at {card['icefall_commit']}, found {commit}")
    return source


def prepare_recipe(card, base, icefall, out):
    if out.exists():
        raise FileExistsError(f"Recipe already exists: {out}")
    spec = next(f for f in card["recipe_files"] if f["path"] == "recipe/make_recipe.py")
    script = base / "upstream" / spec["path"]
    verify_file(script, spec)
    subprocess.run([sys.executable, str(script), "--icefall", str(icefall), "--out", str(out)], check=True)


def train_cmd(config, card, base, recipe, cuts, output, seed, start_epoch, start_batch):
    t = config["train"]
    return [
        sys.executable, str(recipe / "finetune.py"), "--world-size", "1", "--seed", str(seed),
        "--num-epochs", str(t["num_epochs"]), "--start-epoch", str(start_epoch),
        "--start-batch", str(start_batch), "--use-fp16", str(int(t["fp16"])),
        "--exp-dir", str(output), "--manifest-dir", str(cuts), "--bpe-model", str(base / "bpe.model"),
        "--do-finetune", str(int(start_epoch == 1 and start_batch == 0)),
        "--finetune-ckpt", str(base / "pytorch/pretrained.pt"), "--use-mux", "0",
        *model_args(card), "--context-size", str(card["context_size"]), "--use-transducer", "1", "--use-ctc", "0",
        "--base-lr", str(t["lr"]), "--lr-batches", "100000", "--lr-epochs", "100",
        "--max-duration", str(t["max_duration"]), "--on-the-fly-feats", "1", "--enable-musan", "0",
        "--bucketing-sampler", str(int(t.get("bucketing_sampler", True))),
        "--num-buckets", str(t["buckets"]), "--num-workers", str(t["workers"]),
        "--save-every-n", str(t["save_every"]), "--keep-last-k", str(t["keep_last_k"]),
    ]


def strict_check(card, recipe, icefall, base):
    # Keep the Icefall recipe imports inside a subprocess with a verified source path.
    import shlex

    spec = next(f for f in card["model_files"] if f["path"] == "pytorch/pretrained.pt")
    checkpoint = base / spec["path"]
    verify_file(checkpoint, spec)
    token_ids = verify_tokenizer(base, card["vocab_size"])
    argv = shlex.join(model_args(card) + ["--context-size", str(card["context_size"]), "--use-transducer", "1", "--use-ctc", "0"])
    snippet = """
import json
import shlex
import sys
import torch
from finetune import get_model, get_params, get_parser

parser = get_parser()
params = get_params()
params.update(vars(parser.parse_args(shlex.split(sys.argv[2]))))
params.update(vocab_size=int(sys.argv[3]), blank_id=int(sys.argv[4]))
model = get_model(params)
state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
model.load_state_dict(state["model"], strict=True)
print(json.dumps({"strict_load": True, "parameters": sum(p.numel() for p in model.parameters())}))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(recipe), str(icefall)]))
    subprocess.run([sys.executable, "-c", snippet, str(checkpoint), argv,
                    str(card["vocab_size"]), str(token_ids["blank_id"])], env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--icefall", required=True, type=Path)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--cuts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--stage", choices=["recipe", "strict", "train"], default="train")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = experiment.load(args.experiment)
    if family(cfg)["training_adapter"] != "gipbench.model.train.vietnote_icefall":
        parser.error("A VietNote experiment configuration is required")
    card = json.loads(LOCK.read_text(encoding="utf-8"))
    base = experiment.abspath(cfg["base_model"])
    icefall, recipe = args.icefall.resolve(), args.recipe.resolve()
    if args.stage == "recipe":
        if args.dry_run:
            print(f"Build official recipe at {recipe} using Icefall {card['icefall_commit']} from {icefall}")
            return
        prepare_recipe(card, base, icefall_source(card, icefall), recipe)
        return
    if args.stage == "strict":
        if args.dry_run:
            print(f"Strict load of {base / 'pytorch/pretrained.pt'} with {recipe}")
            return
        strict_check(card, recipe, icefall_source(card, icefall), base)
        return
    if args.cuts is None or args.output is None:
        parser.error("Training requires --cuts and --output")
    cuts, output = args.cuts.resolve(), args.output.resolve()
    command = train_cmd(cfg, card, base, recipe, cuts, output, args.seed, args.start_epoch, args.start_batch)
    print(json.dumps(command, ensure_ascii=False), flush=True)
    if args.dry_run:
        return
    import torch

    from gipbench.model.readiness import require_training

    if not torch.cuda.is_available():
        parser.error("CUDA is required for the registered training configuration")
    if args.seed not in cfg["seeds"]:
        parser.error("Select a registered training seed")
    try:
        require_training(cfg, args.seed, cuts, torch.cuda.get_device_name())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    icefall = icefall_source(card, icefall)
    strict_check(card, recipe, icefall, base)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(icefall), str(recipe)]))
    subprocess.run(command, cwd=recipe, env=env, check=True)


if __name__ == "__main__":
    main()
