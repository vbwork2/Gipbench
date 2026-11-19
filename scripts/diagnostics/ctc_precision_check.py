"""Compare the Gipformer auxiliary CTC loss for 200 training steps without saving weights.

Run with the fine-tune Python and add the icefall root and zipformer recipe folder to PYTHONPATH.
"""

import argparse
import faulthandler
import itertools
import json
import math
from pathlib import Path

import sentencepiece as spm
import torch
from lhotse import CutSet, load_manifest_lazy

import finetune_cs as recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-duration", type=int, default=40)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.max_duration < 20:
        parser.error("Use at least one step and a batch budget of at least 20 seconds")
    if not torch.cuda.is_available() or (args.precision == "bf16" and not torch.cuda.is_bf16_supported()):
        parser.error(f"CUDA with {args.precision} support is required")

    base = Path("models/gip15")
    cuts_dir = Path("data/ft_trial/data/lhotse")
    recipe_parser = recipe.get_parser()
    recipe.LibriSpeechAsrDataModule.add_arguments(recipe_parser)
    cfg = recipe_parser.parse_args([
        "--use-transducer", "1", "--use-ctc", "1", "--base-lr", "0.001",
        "--max-duration", str(args.max_duration), "--num-buckets", "10", "--num-workers", "0",
        "--on-the-fly-feats", "True", "--enable-musan", "False", "--enable-spec-aug", "True",
    ])
    params = recipe.get_params()
    params.update(vars(cfg))
    recipe.fix_random_seed(params.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sp = spm.SentencePieceProcessor(model_file=str(base / "bpe.model"))
    params.blank_id, params.vocab_size = sp.piece_to_id("<blk>"), sp.get_piece_size()
    model = recipe.get_model(params)
    with (base / "model.pt").open("rb") as f:
        ckpt = torch.load(f, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    model.load_state_dict({name.removeprefix("module."): tensor for name, tensor in state.items()})
    del ckpt, state
    model = model.cuda().train()
    head = model.ctc_output[1].weight
    initial_head = head.detach().clone()
    groups = recipe.get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True)
    optimizer = recipe.ScaledAdam(groups, lr=params.base_lr, clipping_scale=2.0)
    scheduler = recipe.Eden(optimizer, params.lr_batches, params.lr_epochs)

    def eligible(cut):
        frames = cut.num_frames or round(cut.duration * 100)
        return 1 <= cut.duration <= 20 and ((frames - 7) // 2 + 1) // 2 >= len(
            sp.encode(cut.supervisions[0].text))

    sources = [cuts_dir / f"{name}_cuts_raw.jsonl.gz" for name in ("cs_train", "vi_replay")]
    cuts = CutSet.mux(*(load_manifest_lazy(path) for path in sources), weights=[0.5, 0.5],
                      stop_early=True, seed=params.seed)
    loader = recipe.LibriSpeechAsrDataModule(cfg).train_dataloaders(cuts.filter(eligible))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as f:
        def record(row):
            output = json.dumps(row, allow_nan=False)
            print(output, flush=True)
            f.write(output + "\n")
            f.flush()

        record({"precision": args.precision, "seed": params.seed, "steps": args.steps,
                "max_duration": args.max_duration, "ctc_loss_scale": params.ctc_loss_scale,
                "checkpoint": str(base / "model.pt"), "cuts": [str(path) for path in sources],
                "torch": torch.__version__, "device": torch.cuda.get_device_name()})
        step = 0
        for step, batch in enumerate(itertools.islice(loader, args.steps), 1):
            faulthandler.dump_traceback_later(180)
            params.batch_idx_train = step
            if (step - 1) % 10 == 0:
                recipe.set_batch_count(model, recipe.get_adjusted_batch_count(params))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.precision == "bf16"):
                loss, info = recipe.compute_loss(params, model, sp, batch, is_training=True)
            if not all(math.isfinite(info[key]) for key in ("loss", "ctc_loss", "simple_loss", "pruned_loss")):
                record({"step": step, "status": "nonfinite_loss"})
                raise SystemExit(2)
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            norm = float(torch.nn.utils.get_total_norm(grads))
            if not math.isfinite(norm):
                record({"step": step, "status": "nonfinite_gradient"})
                raise SystemExit(2)
            scheduler.step_batch(step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            record({"step": step, "cuts": [cut.id for cut in batch["supervisions"]["cut"]],
                    "loss_per_frame": info["loss"] / info["frames"],
                    "ctc_per_frame": info["ctc_loss"] / info["frames"], "grad_norm": norm})
            faulthandler.cancel_dump_traceback_later()
        change = float((head.detach() - initial_head).norm())
        peak = round(torch.cuda.max_memory_allocated() / 2**20)
        success = step == args.steps and math.isfinite(change) and change > 0
        record({"status": "complete" if success else "incomplete", "steps": step,
                "head_delta_l2": change if math.isfinite(change) else None, "peak_cuda_mib": peak})
        if not success:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
