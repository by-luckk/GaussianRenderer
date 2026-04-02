from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply

from .config import SamplingConfig


@dataclass
class GaussianAssetTransform:
    center: np.ndarray
    scale: float
    translation: np.ndarray


def compute_asset_transform(points: np.ndarray, config: SamplingConfig, plane_height: float) -> GaussianAssetTransform:
    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)

    centered = points - center
    span = np.maximum(centered.max(axis=0) - centered.min(axis=0), 1e-6)
    scale = float(config.target_extent / np.max(span))
    scaled = centered * scale

    z_shift = float(-scaled[:, 2].min())
    scaled_height = float(scaled[:, 2].max() - scaled[:, 2].min())
    if config.drop_height is None:
        clearance = config.drop_clearance
        if clearance is None:
            clearance = max(0.06, 0.6 * scaled_height)
        bottom_height = plane_height + float(clearance)
    else:
        bottom_height = float(config.drop_height)

    translation = np.array([0.0, 0.0, z_shift + bottom_height], dtype=np.float32)
    return GaussianAssetTransform(center=center.astype(np.float32), scale=scale, translation=translation)


def transform_points(points: np.ndarray, transform: GaussianAssetTransform) -> np.ndarray:
    return ((points - transform.center) * transform.scale + transform.translation).astype(np.float32)


def prepare_gaussian_asset(
    ply_path: str | Path,
    sampling: SamplingConfig,
    plane_height: float,
) -> tuple[GaussianData, GaussianAssetTransform]:
    gaussians = load_ply(ply_path)
    xyz = np.asarray(gaussians.xyz, dtype=np.float32)
    transform = compute_asset_transform(xyz, sampling, plane_height)

    return (
        GaussianData(
            xyz=transform_points(xyz, transform),
            rot=np.asarray(gaussians.rot, dtype=np.float32).copy(),
            scale=np.asarray(gaussians.scale, dtype=np.float32).copy() * transform.scale,
            opacity=np.asarray(gaussians.opacity, dtype=np.float32).copy(),
            sh=np.asarray(gaussians.sh, dtype=np.float32).copy(),
        ),
        transform,
    )
