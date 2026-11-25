"""Resolve the model-specific training and feature contracts."""

import yaml

from gipbench.core.paths import REPO

REGISTRY = REPO / "configs/model/families.yaml"


def family(config):
    name = config.get("family", "gipformer")
    entries = yaml.safe_load(REGISTRY.read_text(encoding="utf-8"))
    if name not in entries:
        raise ValueError(f"Unknown model family: {name}")
    return entries[name]
