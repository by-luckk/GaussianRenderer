import numpy as np
import torch
from torch import Tensor

from gaussian_renderer.core.batch_rasterization import quaternion_multiply

from .binding import GaussianParticleBinding


def _as_particle_pos(particle_pos: np.ndarray | Tensor, binding: GaussianParticleBinding) -> Tensor:
    if not isinstance(particle_pos, Tensor):
        particle_pos = torch.as_tensor(particle_pos, device=binding.particle_rest.device, dtype=torch.float32)
    else:
        particle_pos = particle_pos.to(device=binding.particle_rest.device, dtype=torch.float32)

    if particle_pos.ndim == 3:
        particle_pos = particle_pos[0]
    return particle_pos


def deform_gaussian_xyz(
    particle_pos: np.ndarray | Tensor,
    binding: GaussianParticleBinding,
) -> Tensor:
    particle_pos = _as_particle_pos(particle_pos, binding)

    particle_disp = particle_pos[binding.particle_indices] - binding.particle_rest[binding.particle_indices]
    return binding.gaussian_rest_xyz + (particle_disp * binding.weights[..., None]).sum(dim=1)


def _rotation_matrix_to_quaternion(rotmat: Tensor) -> Tensor:
    m00 = rotmat[..., 0, 0]
    m01 = rotmat[..., 0, 1]
    m02 = rotmat[..., 0, 2]
    m10 = rotmat[..., 1, 0]
    m11 = rotmat[..., 1, 1]
    m12 = rotmat[..., 1, 2]
    m20 = rotmat[..., 2, 0]
    m21 = rotmat[..., 2, 1]
    m22 = rotmat[..., 2, 2]

    q_abs = torch.sqrt(
        torch.clamp(
            torch.stack(
                [
                    1.0 + m00 + m11 + m22,
                    1.0 + m00 - m11 - m22,
                    1.0 - m00 + m11 - m22,
                    1.0 - m00 - m11 + m22,
                ],
                dim=-1,
            ),
            min=0.0,
        )
    )

    candidates = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    denom = (2.0 * q_abs[..., None]).clamp_min(1e-8)
    quats = candidates / denom
    best = q_abs.argmax(dim=-1, keepdim=True)
    quats = quats.gather(-2, best[..., None].expand(*best.shape[:-1], 1, 4)).squeeze(-2)
    return quats / quats.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def deform_gaussian_rot(
    particle_pos: np.ndarray | Tensor,
    binding: GaussianParticleBinding,
) -> Tensor:
    if binding.gaussian_rest_rot is None:
        raise ValueError("Gaussian rest rotations are required for rotation deformation.")

    particle_pos = _as_particle_pos(particle_pos, binding)
    rest = binding.particle_rest[binding.particle_indices]
    curr = particle_pos[binding.particle_indices]
    weights = binding.weights

    rest_center = (rest * weights[..., None]).sum(dim=1, keepdim=True)
    curr_center = (curr * weights[..., None]).sum(dim=1, keepdim=True)
    rest_local = rest - rest_center
    curr_local = curr - curr_center

    cov = torch.einsum("nk,nki,nkj->nij", weights, rest_local, curr_local)
    u, _, vh = torch.linalg.svd(cov)
    v = vh.transpose(-2, -1)
    flip = torch.det(v @ u.transpose(-2, -1)) < 0.0
    if flip.any():
        v[flip, :, -1] *= -1.0
    rotmat = v @ u.transpose(-2, -1)
    delta_rot = _rotation_matrix_to_quaternion(rotmat)
    return quaternion_multiply(delta_rot, binding.gaussian_rest_rot)
