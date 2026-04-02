import numpy as np
import torch
from torch import Tensor

from .binding import GaussianParticleBinding


def deform_gaussian_xyz(
    particle_pos: np.ndarray | Tensor,
    binding: GaussianParticleBinding,
) -> Tensor:
    if not isinstance(particle_pos, Tensor):
        particle_pos = torch.as_tensor(particle_pos, device=binding.particle_rest.device, dtype=torch.float32)
    else:
        particle_pos = particle_pos.to(device=binding.particle_rest.device, dtype=torch.float32)

    if particle_pos.ndim == 3:
        particle_pos = particle_pos[0]

    particle_disp = particle_pos[binding.particle_indices] - binding.particle_rest[binding.particle_indices]
    return binding.gaussian_rest_xyz + (particle_disp * binding.weights[..., None]).sum(dim=1)
