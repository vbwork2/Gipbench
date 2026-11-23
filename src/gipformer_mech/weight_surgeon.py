"""Compute non-gradient FFN and Joiner weight edits for sparse patch evaluation."""

import torch


def low_variance_basis(activations: torch.Tensor, rank_ratio: float) -> torch.Tensor:
    if activations.ndim != 2 or not len(activations) or not 0 < rank_ratio < 1:
        raise ValueError("Expected a nonempty activation matrix and a strict rank fraction.")
    data = activations.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(data).all():
        raise ValueError("Activations must be finite.")
    covariance = (data.T @ data) / len(data)
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, :max(1, round(rank_ratio * data.shape[1]))]


@torch.no_grad()
def gate_tighten(name: str, weight: torch.Tensor, neuron: int, clean: torch.Tensor,
                 bug: torch.Tensor, rank_ratio: float, strength: float) -> dict:
    if bug.ndim != 2 or not len(bug) or not 0 <= neuron < len(weight) or not 0 < strength <= 1:
        raise ValueError("Invalid gate edit activation matrix, neuron or strength.")
    basis = low_variance_basis(clean, rank_ratio)
    bug_mean = bug.detach().to(basis).mean(dim=0)
    direction = basis @ (basis.T @ bug_mean)
    norm = direction.norm()
    if not torch.isfinite(norm) or norm <= torch.finfo(direction.dtype).eps:
        raise ValueError("The error mean has no usable low-variance direction.")
    direction = direction / norm
    row = weight[neuron].detach().to(basis)
    delta = -strength * (row @ direction) * direction
    indices = torch.arange(neuron * row.numel(), (neuron + 1) * row.numel(), dtype=torch.long)
    return {name: {"indices": indices, "delta": delta.to(weight).cpu()}}


@torch.no_grad()
def out_proj(name: str, weight: torch.Tensor, neuron: int, clean_sensitivities: torch.Tensor,
             edit_direction: torch.Tensor, rank_ratio: float, strength: float) -> dict:
    if not 0 <= neuron < weight.shape[1] or not 0 < strength <= 1:
        raise ValueError("Invalid out-projection neuron or strength.")
    basis = low_variance_basis(clean_sensitivities, rank_ratio)
    proposed = edit_direction.detach().to(basis)
    delta = strength * (basis @ (basis.T @ proposed))
    indices = torch.arange(weight.shape[0], dtype=torch.long) * weight.shape[1] + neuron
    return {name: {"indices": indices, "delta": delta.to(weight).cpu()}}


def joiner_local(name: str, indices: list[int], delta: list[float]) -> dict:
    if name not in {"joiner.output_linear.weight", "joiner.decoder_proj.weight"}:
        raise ValueError("Local Joiner edits must target the output or decoder projection weight.")
    return {name: {"indices": torch.tensor(indices, dtype=torch.long), "delta": torch.tensor(delta, dtype=torch.float32)}}
