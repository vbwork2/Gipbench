"""Apply reversible sparse deltas to an in-memory model and serialize accepted edits."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from gipformer_mech.model_audit import checksum


def model_checksum(model: torch.nn.Module) -> str:
    digest = sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().to("cpu").contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def prepare(model: torch.nn.Module, patch: dict, hard_cap: float = 0.005) -> tuple[dict, float]:
    parameters = dict(model.named_parameters())
    if not patch or not 0 < hard_cap <= 0.005:
        raise ValueError("A nonempty delta and an edit cap of at most 0.5% are required.")
    staged = {}
    edited = 0
    for name, values in patch.items():
        tensor = parameters[name]
        index = values["indices"]
        delta = values["delta"]
        if index.ndim != 1 or index.dtype != torch.long or delta.shape != index.shape or not delta.is_floating_point():
            raise ValueError(f"Invalid sparse index or delta format for {name}.")
        if (
            not tensor.is_contiguous() or not tensor.is_floating_point() or not len(index)
            or index.min() < 0 or index.max() >= tensor.numel() or len(index.unique()) != len(index)
            or not torch.isfinite(delta).all()
        ):
            raise ValueError(f"Invalid sparse edit for {name}.")
        index = index.to(tensor.device)
        before = tensor.view(-1).index_select(0, index)
        after = before + delta.to(tensor)
        if not torch.isfinite(after).all():
            raise ValueError(f"Sparse edit would create invalid values in {name}.")
        changed = after != before
        edited += int(changed.sum())
        staged[name] = (tensor, index[changed], before[changed], after[changed])
    ratio = edited / sum(tensor.numel() for tensor in parameters.values())
    if not 0 < ratio <= hard_cap:
        raise ValueError(f"Changed parameter fraction {ratio:.6%} is empty or exceeds the {hard_cap:.2%} limit.")
    return staged, ratio


@torch.no_grad()
def apply(model: torch.nn.Module, patch: dict, hard_cap: float = 0.005) -> tuple[dict, float]:
    staged, ratio = prepare(model, patch, hard_cap)
    snapshots = {}
    try:
        for name, (tensor, index, before, after) in staged.items():
            snapshots[name] = (index, before)
            tensor.view(-1).index_copy_(0, index, after)
    except BaseException:
        rollback(model, snapshots)
        raise
    return snapshots, ratio


@torch.no_grad()
def rollback(model: torch.nn.Module, snapshots: dict, expected_checksum: str | None = None) -> None:
    parameters = dict(model.named_parameters())
    for name, (index, before) in snapshots.items():
        tensor = parameters[name]
        tensor.view(-1).index_copy_(0, index.to(tensor.device), before.to(tensor))
    if expected_checksum and model_checksum(model) != expected_checksum:
        raise ValueError("Rollback did not restore the expected model checksum.")


def save(path: Path, patch: dict, metadata: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Patch already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent, prefix=".staging-") as tmp:
        staging = Path(tmp)
        torch.save({
            name: {key: value.detach().cpu() for key, value in item.items()} for name, item in patch.items()
        }, staging / "delta.pt")
        manifest = {**metadata, "delta_sha256": checksum(staging / "delta.pt")}
        (staging / "meta.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        staging.rename(path)


def load(path: Path, checkpoint_sha256: str) -> tuple[dict, dict]:
    metadata = json.loads((path / "meta.json").read_text(encoding="utf-8"))
    if metadata["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("This sparse delta targets a different base checkpoint.")
    if checksum(path / "delta.pt") != metadata["delta_sha256"]:
        raise ValueError("Sparse delta checksum does not match its metadata.")
    return torch.load(path / "delta.pt", map_location="cpu", weights_only=True), metadata
