from pathlib import Path

import numpy as np

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply

from .config import SamplingConfig
from .gaussian_asset import GaussianAssetTransform, compute_asset_transform, transform_points


def voxel_downsample_indices(points: np.ndarray, weights: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size <= 0.0 or len(points) == 0:
        return np.arange(len(points), dtype=np.int64)

    min_corner = points.min(axis=0, keepdims=True)
    grid_coords = np.floor((points - min_corner) / voxel_size).astype(np.int64)

    best_indices: dict[tuple[int, int, int], int] = {}
    for idx, key in enumerate(map(tuple, grid_coords)):
        best_idx = best_indices.get(key)
        if best_idx is None or weights[idx] > weights[best_idx]:
            best_indices[key] = idx

    keep = np.fromiter(best_indices.values(), dtype=np.int64)
    keep.sort()
    return keep


def prepare_particles_from_gaussians(
    gaussians: GaussianData,
    config: SamplingConfig,
    plane_height: float,
    transform: GaussianAssetTransform | None = None,
) -> tuple[np.ndarray, float]:
    xyz = np.asarray(gaussians.xyz, dtype=np.float32)
    opacity = np.asarray(gaussians.opacity, dtype=np.float32).reshape(-1)
    if transform is None:
        transform = compute_asset_transform(xyz, config, plane_height)
    points = transform_points(xyz, transform)

    opacity_quantile = float(np.clip(config.opacity_quantile, 0.0, 0.99))
    opacity_cutoff = float(np.quantile(opacity, opacity_quantile))
    keep_mask = opacity >= opacity_cutoff

    if keep_mask.sum() >= min(config.max_particles, len(points)):
        points = points[keep_mask]
        opacity = opacity[keep_mask]

    if len(points) > config.max_particles:
        scaled_span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-3)
        voxel_size = float(0.45 * np.cbrt(np.prod(scaled_span) / config.max_particles))
        sampled_idx = np.arange(len(points), dtype=np.int64)

        for _ in range(8):
            sampled_idx = voxel_downsample_indices(points, opacity, voxel_size)
            if len(sampled_idx) <= config.max_particles:
                break
            voxel_size *= 1.35

        if len(sampled_idx) < config.max_particles:
            rng = np.random.default_rng(0)
            remaining_idx = np.setdiff1d(np.arange(len(points), dtype=np.int64), sampled_idx, assume_unique=True)
            n_extra = min(config.max_particles - len(sampled_idx), len(remaining_idx))
            if n_extra > 0:
                extra_idx = rng.choice(remaining_idx, size=n_extra, replace=False)
                sampled_idx = np.concatenate([sampled_idx, extra_idx])
                sampled_idx.sort()

        points = points[sampled_idx]
        opacity = opacity[sampled_idx]

    if len(points) > config.max_particles:
        keep = np.argsort(opacity)[-config.max_particles :]
        keep.sort()
        points = points[keep]

    final_span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-3)
    particle_size = float(np.clip(np.cbrt(np.prod(final_span) / len(points)), 3e-3, 1.2e-2))
    return points.astype(np.float32), particle_size


def prepare_flower_particles(
    ply_path: Path,
    max_particles: int,
    target_extent: float,
    drop_height: float | None,
    plane_height: float,
    drop_clearance: float | None,
    opacity_quantile: float,
) -> tuple[np.ndarray, float]:
    config = SamplingConfig(
        max_particles=max_particles,
        target_extent=target_extent,
        opacity_quantile=opacity_quantile,
        drop_height=drop_height,
        drop_clearance=drop_clearance,
    )
    gaussians = load_ply(ply_path)
    return prepare_particles_from_gaussians(gaussians, config, plane_height)
