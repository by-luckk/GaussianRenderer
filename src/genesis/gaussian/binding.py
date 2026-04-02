from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial import cKDTree


@dataclass
class GaussianParticleBinding:
    particle_indices: torch.Tensor
    weights: torch.Tensor
    particle_rest: torch.Tensor
    gaussian_rest_xyz: torch.Tensor


def build_gaussian_particle_binding(
    gaussian_xyz: np.ndarray,
    particle_xyz: np.ndarray,
    k: int,
    device: torch.device | str,
) -> GaussianParticleBinding:
    k = min(int(k), len(particle_xyz))
    tree = cKDTree(particle_xyz)
    distances, indices = tree.query(gaussian_xyz, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    weights = 1.0 / np.maximum(distances, 1e-8)
    zero_mask = distances <= 1e-8
    if zero_mask.any():
        weights[zero_mask.any(axis=1)] = zero_mask[zero_mask.any(axis=1)]
    weights = weights / weights.sum(axis=1, keepdims=True)

    return GaussianParticleBinding(
        particle_indices=torch.as_tensor(indices, device=device, dtype=torch.long),
        weights=torch.as_tensor(weights, device=device, dtype=torch.float32),
        particle_rest=torch.as_tensor(particle_xyz, device=device, dtype=torch.float32),
        gaussian_rest_xyz=torch.as_tensor(gaussian_xyz, device=device, dtype=torch.float32),
    )
