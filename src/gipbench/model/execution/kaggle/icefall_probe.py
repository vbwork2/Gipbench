"""Run the published VietNote fine-tuning CLI on generated tones in the Kaggle notebook."""

import math
import tarfile
from urllib.request import urlopen


def fetch_checked(url, path, expected):
    with urlopen(url, timeout=60) as response, path.open("wb") as output:
        shutil.copyfileobj(response, output)
    if digest(path) != expected:
        raise ValueError(f"Unexpected public artifact: {path.name}")


def make_probe_cuts(root):
    import numpy as np
    import soundfile as sf
    from lhotse import CutSet, Recording, SupervisionSegment

    for split, count in (("train", 1), ("valid", 32)):
        cuts = []
        for index in range(count):
            name = f"tone-{split}-{index}"
            samples = 32000 + 160 * index
            t = np.arange(samples) / 16000
            frequency = 200 if split == "train" else 380 + index
            audio = root / f"{name}.wav"
            sf.write(audio, 0.25 * np.sin(2 * np.pi * (frequency * t + 40 * t**2)), 16000)
            cut = Recording.from_file(audio, recording_id=name).to_cut()
            cut.supervisions = [SupervisionSegment(name, name, 0, cut.duration, text="xin chào")]
            cuts.append(cut)
        CutSet.from_cuts(cuts).to_file(root / f"{split}.jsonl.gz")


def check_state(actual, expected):
    import torch

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            check_state(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        for a, b in zip(actual, expected, strict=True):
            check_state(a, b)
    else:
        assert actual == expected


def verify_probe_checkpoint(checkpoint, initial):
    import torch
    from finetune import get_model
    from icefall.checkpoint import load_checkpoint
    from icefall.utils import AttributeDict, get_parameter_groups_with_lrs
    from optim import Eden, ScaledAdam

    saved = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    params = AttributeDict(saved)
    if params.batch_idx_train != 1 or not math.isfinite(params.train_loss) or not saved["optimizer"]["state"]:
        raise ValueError("Missing one-step optimizer or loss evidence")
    model = get_model(params)
    optimizer = ScaledAdam(get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
                           lr=params.base_lr, clipping_scale=2.0)
    scheduler = Eden(optimizer, params.lr_batches, params.lr_epochs)
    load_checkpoint(checkpoint, model, optimizer=optimizer, scheduler=scheduler, strict=True)
    torch.testing.assert_close(model.state_dict(), saved["model"], rtol=0, atol=0, equal_nan=False)
    check_state(optimizer.state_dict(), saved["optimizer"])
    check_state(scheduler.state_dict(), saved["scheduler"])
    pretrained = torch.load(initial, map_location="cpu", weights_only=True)["model"]
    changed = next(name for name, value in model.named_parameters()
                   if not torch.equal(value.detach(), pretrained[name]))
    return dict(steps=params.batch_idx_train, train_loss=params.train_loss, strict_reload=True,
                optimizer_reload=True, scheduler_reload=True, updated_parameter=changed,
                parameters=sum(p.numel() for p in model.parameters()), checkpoint_sha256=digest(checkpoint))


def run_icefall_probe(config, environment, out):
    if (environment["gpu"], environment["python"], environment["torch"], environment["cuda"]) != (
            "Tesla T4", "3.13.16", "2.11.0+cu130", "13.0"):
        raise RuntimeError("Kaggle base runtime differs from the verified T4")
    requirements = out / "requirements.txt"
    requirements.write_text(config["requirements"], encoding="utf-8")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:",
                    "--require-hashes", "-r", str(requirements)], check=True)
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                if dist.metadata["Name"]}
    write_json(out / "runtime-packages.json", dict(sorted(packages.items())))
    commit = "9c966762e058bd7ba79d9da71e42fc06b64b703e"
    recipe_rev = "fd4dcde6f41ca888823a0c50fcce8cf8d274487c"
    weights_rev = "69eec0a86fef25639996813d3e10fbe656659650"
    public_model = f"https://huggingface.co/Kaisye/VietNote-Zipformer-78M-Streaming/resolve/{weights_rev}"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        fetch_checked(f"https://codeload.github.com/k2-fsa/icefall/tar.gz/{commit}", root / "icefall.tar.gz",
                      "70e23f83cfe0b3843b3d296b96ed6c7fc3de743c3c1f65a9102d67718db3bef7")
        with tarfile.open(root / "icefall.tar.gz") as archive:
            archive.extractall(root, filter="data")
        icefall = root / f"icefall-{commit}"
        fetch_checked(f"https://raw.githubusercontent.com/kaisye/vietnote-asr/{recipe_rev}/recipe/make_recipe.py",
                      root / "make_recipe.py", "9cc567ebaed1a2bd76d6c130f6c210f199e203f1f7defa4c9ce7050fe31af3f6")
        recipe = root / "recipe"
        subprocess.run([sys.executable, str(root / "make_recipe.py"), "--icefall", str(icefall),
                        "--out", str(recipe)], check=True)
        fetch_checked(f"{public_model}/bpe.model", root / "bpe.model",
                      "0534a242b49e7d82e751f454645dc8211870955f9da0c57fe480f0eb345ab8af")
        initial = root / "pretrained.pt"
        fetch_checked(f"{public_model}/pytorch/pretrained.pt", initial,
                      "90f840fd8d4939bd394be6968fe2965100192411e6ffb97959da332d0d7cd9d3")
        make_probe_cuts(root)
        args = [sys.executable, str(recipe / "finetune.py"), "--world-size", "1", "--seed", "0",
                "--exp-dir", str(root / "exp"), "--manifest-dir", str(root), "--bpe-model", str(root / "bpe.model"),
                "--num-epochs", "1", "--do-finetune", "1", "--finetune-ckpt", str(initial),
                "--num-encoder-layers", "2,2,3,4,3,2", "--encoder-dim", "192,256,384,512,384,256",
                "--feedforward-dim", "512,768,1024,1536,1024,768", "--decoder-dim", "512", "--joiner-dim", "512",
                "--causal", "1", "--chunk-size", "16,32,64,-1", "--left-context-frames", "64,128,256,-1",
                "--use-fp16", "0", "--use-mux", "0", "--bucketing-sampler", "0", "--num-workers", "0",
                "--max-duration", "8", "--on-the-fly-feats", "1", "--enable-musan", "0", "--tensorboard", "0",
                "--base-lr", "0.0015", "--lr-batches", "100000", "--lr-epochs", "100"]
        env = dict(os.environ, PYTHONPATH=str(icefall), OMP_NUM_THREADS="1", LHOTSE_AUDIO_BACKEND="LibsndfileBackend")
        log = out / "train.log"
        with log.open("w") as handle:
            proc = subprocess.run(args, cwd=root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(log.read_text()[-5000:])
        sys.path[:0] = [str(recipe), str(icefall)]
        ckpt = out / "epoch-1.pt"
        shutil.move(root / "exp/epoch-1.pt", ckpt)
        result = verify_probe_checkpoint(ckpt, initial)
        write_json(out / "icefall.json", dict(result, icefall_commit=commit, recipe_commit=recipe_rev,
                   pretrained_revision=weights_rev, train_cuts=1, valid_cuts=32, source="synthetic_chirp"))
