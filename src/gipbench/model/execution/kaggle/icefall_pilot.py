"""Run a FLEURS engineering pilot with the pinned official Icefall recipe."""

import base64
import re


def materialize_pilot(pilot, root, out):
    from lhotse import CutSet, Recording, SupervisionSegment

    for split in ("train", "valid"):
        split_rows, cuts = [], []
        for source in pilot[split]:
            row = {key: value for key, value in source.items() if key != "flac"}
            audio = root / (row["id"] + ".flac")
            audio.write_bytes(base64.b64decode(source["flac"], validate=True))
            if digest(audio) != row["audio_sha256"]:
                raise ValueError("Transferred FLEURS audio differs from the prepared pilot")
            cut = Recording.from_file(audio, recording_id=row["id"]).to_cut()
            if cut.duration != row["duration"]:
                raise ValueError("Transferred FLEURS duration differs")
            cut.supervisions = [SupervisionSegment(cut.id, cut.id, 0, cut.duration, text=row["text"])]
            cuts.append(cut)
            split_rows.append(row)
        CutSet.from_cuts(cuts).to_file(root / f"{split}.jsonl.gz")
        (out / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in split_rows), encoding="utf-8")


def run_pilot(pilot, root, out, recipe, icefall, args, env, public_model):
    import onnx
    import torch

    def train(log_name, settings):
        return run_pilot_command([*args, *settings], root, env, out / log_name)

    first = train("train.log", [])
    resumed = train("resume.log", ["--do-finetune", "0", "--start-epoch", "2", "--num-epochs", "2"])
    reference_exp = root / "uninterrupted"
    train("uninterrupted.log", ["--exp-dir", str(reference_exp), "--num-epochs", "2"])
    epoch_1 = root / "exp/epoch-1.pt"
    epoch_2 = root / "exp/epoch-2.pt"
    before, after, continuous = [
        torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
        for checkpoint in (epoch_1, epoch_2, reference_exp / "epoch-2.pt")
    ]
    if not all(f"Loading {key} state dict" in resumed for key in ("optimizer", "scheduler", "grad scaler")):
        raise ValueError("Expected official resume state reload logs")
    matches = {}
    for key in ("model", "optimizer", "scheduler", "grad_scaler"):
        try:
            check_state(after[key], continuous[key])
        except AssertionError:
            matches[key] = False
        else:
            matches[key] = True
    losses = [float(value) for log in (first, resumed)
              for value in re.findall(r"validation: loss=([0-9.eE+-]+)", log)]
    if len(losses) != 2:
        raise ValueError("Expected a validation loss in both epochs")
    report = {
        "release": pilot["release"], "release_sha256": pilot["release_sha256"], "license": pilot["license"],
        "pilot_payload_sha256": pilot["pilot_payload_sha256"],
        "train_cuts": len(pilot["train"]), "valid_cuts": len(pilot["valid"]), "seed": 0, "base_lr": 0.0015,
        "steps_before_resume": before["batch_idx_train"], "steps_resumed": after["batch_idx_train"],
        "steps_uninterrupted": continuous["batch_idx_train"], "epoch_boundary_equal": matches,
        "sampler_saved": before.get("sampler") is not None,
        "rng_saved": any("rng" in key.lower() for key in before), "validation_losses": losses,
        "checkpoint_sha256": digest(epoch_2),
    }
    write_json(out / "pilot.json", report)
    print(json.dumps(report), flush=True)

    fetch_checked(f"{public_model}/tokens.txt", out / "tokens.txt",
                  "ddcb57e85bea2adaf7ffddaf3398ac3046eff16a44ca5b5f3d291f1558638e80")
    export = root / "export"
    export.mkdir()
    torch.save({"model": after["model"]}, export / "epoch-1.pt")
    del before, after, continuous
    launcher = (
        "import functools, runpy, sys, torch; "
        "torch.onnx.export = functools.partial(torch.onnx.export, dynamo=False); "
        "sys.path.insert(0, '.'); sys.argv = sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')"
    )
    arch = args[args.index("--num-encoder-layers"):args.index("--chunk-size")]
    command = [sys.executable, "-c", launcher, "export-onnx.py", "--exp-dir", str(export),
               "--tokens", str(out / "tokens.txt"), "--epoch", "1", "--avg", "1", "--use-averaged-model", "0",
               "--enable-int8-quantization", "0", "--context-size", "2", *arch,
               "--chunk-size=-1", "--left-context-frames=-1"]
    export_env = dict(env, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2")
    run_pilot_command(command, icefall / "egs/librispeech/ASR/zipformer", export_env, out / "export.log")
    for part in ("encoder", "decoder", "joiner"):
        model = export / f"{part}-epoch-1-avg-1.onnx"
        onnx.checker.check_model(str(model))
        shutil.move(model, out / f"{part}.onnx")
    shutil.move(epoch_2, out / "epoch-2.pt")


def run_pilot_command(command, cwd, env, log):
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
    text = log.read_text(encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(text[-5000:])
    return text
