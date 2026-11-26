"""Run a bounded VietNote job in one private Kaggle notebook."""

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def digest(file):
    value = hashlib.sha256()
    with file.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(file, value):
    pending = file.with_suffix(".tmp")
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    pending.replace(file)


def gpu_environment():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU was available in the Kaggle session")
    free, total = torch.cuda.mem_get_info(0)
    probe = torch.ones((128, 128), device="cuda:0")
    if (probe @ probe)[0, 0].item() != 128:
        raise RuntimeError("CUDA calculation failed")
    packages = {}
    for name in ("torch", "k2", "lhotse", "sentencepiece", "onnx", "onnxruntime", "sherpa-onnx", "sherpa-onnx-core"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    command = ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv,noheader"]
    driver = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip()
    return {
        "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu_count": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0), "gpu_total_bytes": total, "gpu_free_bytes": free,
        "driver": driver, "cuda_probe_passed": True,
        "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
        "free_disk_bytes": shutil.disk_usage("/kaggle/working").free,
    }


def extract_inputs(config, root):
    expected = config["bundle"]
    mounted = Path("/kaggle/input")
    matches = [file for file in mounted.rglob("bundle.zip") if file.stat().st_size == expected["bytes"]]
    verified = [file for file in matches if digest(file) == expected["sha256"]]
    if not verified and expected.get("members_manifest_sha256"):
        manifests = [file for file in mounted.rglob("bundle/bundle-members.json")
                     if digest(file) == expected["members_manifest_sha256"]]
        if len(manifests) == 1:
            manifest = manifests[0]
            members = json.loads(manifest.read_text(encoding="utf-8"))
            root.mkdir()
            for name, checksum in members.items():
                source = manifest.parent / name
                target = root / name
                if (not target.resolve().is_relative_to(root) or "\\" in name
                        or not source.resolve().is_relative_to(manifest.parent.resolve())
                        or not source.is_file() or digest(source) != checksum):
                    raise ValueError(f"Missing or modified mounted bundle member: {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            return
    if len(verified) != 1:
        raise ValueError("The pinned dataset archive was unavailable or ambiguous")
    root.mkdir()
    with ZipFile(verified[0]) as archive:
        for member in archive.infolist():
            target = root / member.filename
            if not target.resolve().is_relative_to(root) or "\\" in member.filename:
                raise ValueError("Unsafe path in the bundle")
        archive.extractall(root)


def bootstrap(root, environment, allow_internet):
    lock = root / "runtime/lock.json"
    if not lock.is_file():
        raise RuntimeError("V02 requires a dependency lock matching the Kaggle GPU smoke")
    expected = json.loads(lock.read_text(encoding="utf-8"))
    actual = {key: environment[key] for key in ("torch", "cuda", "python")}
    if actual != {key: expected[key] for key in actual}:
        raise RuntimeError(f"Existing Kaggle runtime differs from the approved lock: {actual}")
    wheelhouse = root / "runtime/wheels"
    if wheelhouse.is_dir():
        source = ["--no-index", "--find-links", str(wheelhouse)]
    elif allow_internet:
        source = []
    else:
        raise RuntimeError("Offline execution requires the approved wheelhouse")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--require-hashes", "--only-binary", ":all:",
                    *source, "-r", str(root / "runtime/requirements.txt")], check=True)
    if "packages" in expected:
        installed = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()
                     if dist.metadata["Name"]}
        changed = sorted(name for name, version in expected["packages"].items() if installed.get(name) != version)
        if changed:
            raise RuntimeError(f"Kaggle distributions differ from the verified Icefall runtime: {changed}")
    import k2
    import lhotse
    import sentencepiece
    import torch

    if str(torch.__version__) != environment["torch"]:
        raise RuntimeError("PyTorch changed during bootstrap")
    return {"k2": k2.__version__, "lhotse": lhotse.__version__, "sentencepiece": sentencepiece.__version__}


def train_v02(config, root, out, environment):
    import torch
    import yaml

    from gipbench.core.manifest import manifest_sha256, read_jsonl, write_jsonl
    from gipbench.model.train.vietnote_icefall import icefall_source, prepare_recipe, strict_check

    protocol = root / "configs/model/experiments/vietnote_validation.yaml"
    arm = protocol.with_name("vietnote_tiny_low_lr.yaml")
    config_hash = digest(protocol)
    arm_hash = digest(arm)
    settings = yaml.safe_load(protocol.read_text(encoding="utf-8"))["tiny"]
    lr = yaml.safe_load(arm.read_text(encoding="utf-8"))["lr"]
    if not 0 < lr < settings["lr"]:
        raise ValueError("Invalid registered lower learning-rate arm")
    settings["lr"] = lr
    base = root / "models/vietnote78"
    card = json.loads((root / "configs/model/vietnote.json").read_text(encoding="utf-8"))
    icefall = icefall_source(card, base / "upstream/icefall" / f"{card['icefall_commit']}.tar.gz")
    recipe = root / "recipe"
    prepare_recipe(card, base, icefall, recipe)
    strict_check(card, recipe, icefall, base)
    sys.path[:0] = [str(recipe), str(icefall), str(base / "upstream/src"), str(root / "scripts/diagnostics")]
    from vietnote_train_validation import tiny

    train = read_jsonl(root / "portable/train.jsonl")
    digest_train = manifest_sha256(train)
    if len(train) != settings["rows"] or digest_train != config["bundle"]["train_manifest_sha256"]:
        raise ValueError("V02 audio manifest identity mismatch")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_jsonl(out / "train.jsonl", train)
    result = tiny(base, train, settings, out)
    result.update(config_sha256=config_hash, lr=lr, lr_protocol_sha256=arm_hash,
                  checkpoint_input_sha256=digest(base / "pytorch/pretrained.pt"), model_revision=card["revision"],
                  train_rows=len(train), train_hours=sum(row["duration"] for row in train) / 3600,
                  train_manifest_sha256=digest_train, gpu=environment["gpu"], torch_version=environment["torch"])
    write_json(out / "result.json", result)


def synthetic_probe(environment, out):
    import torch

    if environment["gpu"] != "Tesla T4" or torch.__version__ != "2.11.0+cu130" or sys.version_info[:2] != (3, 13):
        raise RuntimeError("Unexpected GPU or Python/PyTorch runtime")
    build = "1.24.4.dev20260625"
    wheel = (f"https://huggingface.co/csukuangfj2/k2/resolve/main/ubuntu-cuda/{build}/"
             f"k2-{build}%2Bcuda13.0.torch2.11.0-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl")
    pip = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check"]
    with tempfile.TemporaryDirectory() as tmp:
        constraint = Path(tmp) / "constraints.txt"
        constraint.write_text(f"torch=={torch.__version__}\n", encoding="utf-8")
        subprocess.run([*pip, "-c", str(constraint), "lhotse==1.33.0"], check=True, timeout=160)
    subprocess.run([*pip, "--no-deps", wheel], check=True, timeout=200)
    import k2
    from lhotse import Fbank, FbankConfig

    torch.manual_seed(163)
    t = torch.arange(16000, dtype=torch.float32) / 16000
    signal = (0.3 * torch.sin(2 * torch.pi * (200 * t + 400 * t.square()))).unsqueeze(0)
    fbank = Fbank(FbankConfig(num_mel_bins=80, dither=0.0))
    features = torch.as_tensor(fbank.extract(signal, sampling_rate=16000))
    if features.shape != (100, 80) or not features.isfinite().all() or features.std() <= 0:
        raise RuntimeError("Invalid filterbank features")
    frames = torch.nn.functional.layer_norm(features.to("cuda"), (80,))
    projection = torch.nn.Linear(80, 4, device="cuda")
    prediction = torch.randn(1, 1, 4, 4, device="cuda") * 0.1
    logits = (projection(frames)[None, :, None, :] + prediction).contiguous()
    targets = torch.tensor([[1, 2, 3]], device="cuda")
    loss = k2.rnnt_loss(logits, targets, termination_symbol=0)
    loss.backward()
    grad = projection.weight.grad
    if not loss.is_cuda or not loss.isfinite() or not grad.is_cuda or not grad.isfinite().all() or grad.norm() <= 0:
        raise RuntimeError("Invalid RNN-T loss or projection gradient")
    write_json(out / "probe.json", {
        "source": "synthetic_chirp", "sample_rate": 16000, "samples": signal.numel(),
        "features": list(features.shape), "loss": loss.item(), "gradient_norm": grad.norm().item(),
        "device": str(loss.device), "k2": importlib.metadata.version("k2"), "lhotse": importlib.metadata.version("lhotse"),
    })


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUTF8"] = "1"
    working = Path("/kaggle/working")
    out = working / "gipbench-output"
    out.mkdir()
    environment = gpu_environment()
    environment.update(experiment=config["experiment"], git_commit=config["commit"],
                       bundle_sha256=(config.get("bundle") or {}).get("sha256"),
                       icefall_commit=(config.get("bundle") or {}).get("icefall_commit"))
    write_json(out / "environment.json", environment)
    if config["experiment"] == "synthetic":
        synthetic_probe(environment, out)
    if config["experiment"] in {"icefall", "icefall-pilot", "icefall-memory"}:
        run_icefall_probe(config, environment, out)
    if config["experiment"] == "V02":
        with tempfile.TemporaryDirectory(prefix="vietnote-") as temporary:
            root = Path(temporary) / "inputs"
            extract_inputs(config, root)
            environment["verified_dependencies"] = bootstrap(root, environment, config["internet"])
            write_json(out / "environment.json", environment)
            sys.path.insert(0, str(root / "src"))
            os.environ["PYTHONPATH"] = str(root / "src")
            train_v02(config, root, out, environment)
    write_json(out / "result-manifest.json", {
        "run": config["run"], "experiment": config["experiment"], "commit": config["commit"],
        "bundle_sha256": environment["bundle_sha256"],
        "files": {file.relative_to(out).as_posix(): digest(file) for file in sorted(out.rglob("*")) if file.is_file()},
    })
