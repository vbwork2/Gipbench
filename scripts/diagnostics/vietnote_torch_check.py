"""Check the pinned VietNote PyTorch model with the official Icefall recipe on PYTHONPATH."""

import argparse
import json
from pathlib import Path

from gipbench.core.download import verify_file
from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
from gipbench.core.textnorm import plain_text
from gipbench.infer.adapters.base import load_audio
from gipbench.model.train.vietnote_icefall import model_args
from gipbench.model.vietnote import LOCK
from gipformer_mech.fbank import fbank
from gipformer_mech.rnnt_loader import greedy


def load_model(base, offline):
    import sentencepiece as spm
    import torch
    from finetune import get_model, get_params, get_parser

    card = json.loads(LOCK.read_text(encoding="utf-8"))
    spec = next(f for f in card["model_files"] if f["path"] == "pytorch/pretrained.pt")
    checkpoint = base / spec["path"]
    verify_file(checkpoint, spec)
    sp = spm.SentencePieceProcessor(model_file=str(base / "bpe.model"))
    flags = model_args(card) + ["--context-size", str(card["context_size"]), "--use-transducer", "1",
                                "--use-ctc", "0", "--base-lr", "0.002"]
    if offline:
        flags += ["--chunk-size=-1", "--left-context-frames=-1"]
    params = get_params()
    params.update(vars(get_parser().parse_args(flags)))
    params.update(vocab_size=sp.get_piece_size(), blank_id=sp.piece_to_id("<blk>"),
                  batch_idx_train=1, max_duration=8)
    model = get_model(params)
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"], strict=True)
    return model, sp, params


def inputs(row):
    import torch

    wave, _ = load_audio(row["_audio_path"], target_sr=16000)
    frames = fbank(wave)
    return torch.from_numpy(frames).unsqueeze(0), torch.tensor([len(frames)])


def offline_decode(model, sp, rows, out):
    import torch

    model.eval()
    hyps = []
    with torch.inference_mode():
        for row in rows:
            enc, lengths = model.forward_encoder(*inputs(row))
            hyps.append({"id": row["id"], "hyp": greedy(model, enc, lengths, sp)})
    write_jsonl(str(out), hyps)


def smoke(model, sp, params, row, checkpoint):
    import torch
    from finetune import (Eden, ScaledAdam, compute_loss, get_adjusted_batch_count,
                          get_parameter_groups_with_lrs, set_batch_count)
    from vnasr.text import normalize_text

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    if not 1 <= float(row["duration"]) <= 4:
        raise ValueError("The smoke test requires one training clip of 1–4 seconds")
    text = normalize_text(plain_text(row["text_tagged"]))
    feature, lengths = inputs(row)
    batch = {"inputs": feature, "supervisions": {"num_frames": lengths, "text": [text]}}
    model = model.to("cuda").train()
    set_batch_count(model, get_adjusted_batch_count(params))
    optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
                           lr=params.base_lr, clipping_scale=2.0)
    scheduler = Eden(optimizer, lr_batches=100000, lr_epochs=100)
    probe = model.joiner.output_linear.weight
    before = probe.detach().cpu().clone()
    loss, stats = compute_loss(params, model, sp, batch, is_training=True)
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Non-finite pruned RNN-T loss")
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
        raise RuntimeError("Missing or non-finite gradients")
    scheduler.step_batch(1)
    optimizer.step()
    delta = float((probe.detach().cpu() - before).abs().max())
    if delta == 0:
        raise RuntimeError("The optimizer did not update the Joiner probe")
    model.zero_grad(set_to_none=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, checkpoint)
    peak = torch.cuda.max_memory_allocated()
    updated = probe.detach().cpu().clone()
    del optimizer, scheduler, grads, loss
    reloaded = torch.load(checkpoint, weights_only=True, map_location="cpu")
    model.load_state_dict(reloaded["model"], strict=True)
    if not torch.equal(updated, model.joiner.output_linear.weight.detach().cpu()):
        raise RuntimeError("Checkpoint reload differed")
    return {"id": row["id"], "loss": stats["loss"], "simple_loss": stats["simple_loss"],
            "pruned_loss": stats["pruned_loss"], "optimizer_state_groups": len(reloaded["optimizer"]["state"]),
            "max_joiner_weight_update": delta, "peak_cuda_allocated_bytes": peak, "reloaded": True}


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("offline", "smoke"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/vietnote78"))
    args = parser.parse_args()
    torch.set_num_threads(4 if args.mode == "offline" else 2)
    torch.manual_seed(0)
    rows = read_jsonl(args.manifest)
    if not rows or (args.mode == "smoke" and len(rows) != 1):
        parser.error("Provide a nonempty manifest, with exactly one training utterance for smoke")
    model, sp, params = load_model(args.model, offline=args.mode == "offline")
    if args.mode == "offline":
        offline_decode(model, sp, rows, args.out)
    else:
        result = smoke(model, sp, params, rows[0], args.out.with_suffix(".pt"))
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.mode}: {len(rows)} rows, manifest {manifest_sha256(rows)}, output {args.out}")


if __name__ == "__main__":
    main()
