import importlib.util
import json
from pathlib import Path, PosixPath, PurePosixPath

import pytest

from gipbench.core.fileio import sha256_file
from gipbench.core.paths import REPO
from gipbench.model.vietnote import load_checkpoint


class TrainingPath:
    def __reduce__(self):
        return PosixPath, ("/kaggle/working",)


def test_loads_posix_training_metadata_on_any_host(tmp_path):
    torch = pytest.importorskip("torch")
    file = tmp_path / "posix.pt"
    torch.save({"model": {"weights": torch.ones(2)}, "exp_dir": TrainingPath()}, file)
    assert load_checkpoint(file)["exp_dir"] == PurePosixPath("/kaggle/working")


def test_averaging_requires_compatible_verified_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    spec = importlib.util.spec_from_file_location(
        "vietnote_reexport", REPO / "scripts/diagnostics/vietnote_reexport.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    card = {"revision": "registered", "model_files": [{"path": "pytorch/pretrained.pt", "sha256": "base"}]}
    members = []
    for i, value in enumerate((1.0, 3.0)):
        file = tmp_path / f"epoch-{i}.pt"
        torch.save({"model": {"weights": torch.tensor([value]), "index": torch.tensor([1])}}, file)
        members.append({"path": file.name, "sha256": sha256_file(file)})
    source = tmp_path / "average.json"
    config = {"model_revision": "registered", "base_checkpoint_sha256": "base", "checkpoints": members}
    source.write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "model.pt"
    result = exporter.average_checkpoints(source, card, out)
    assert result["checkpoints"] == [item["sha256"] for item in members]
    assert torch.load(out, weights_only=True)["model"]["weights"].tolist() == [2.0]
    changed = Path(tmp_path / members[1]["path"])
    torch.save({"model": {"weights": torch.tensor([3.0, 5.0]), "index": torch.tensor([1])}}, changed)
    with pytest.raises(ValueError, match="digest mismatch"):
        exporter.average_checkpoints(source, card, tmp_path / "rejected.pt")
    members[1]["sha256"] = sha256_file(changed)
    source.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="shape or dtype"):
        exporter.average_checkpoints(source, card, tmp_path / "rejected.pt")
    assert not (tmp_path / "rejected.pt").exists()
