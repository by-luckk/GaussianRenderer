from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class PathsConfig:
    input_ply: Path
    output_dir: Path


@dataclass
class SamplingConfig:
    max_particles: int
    target_extent: float
    opacity_quantile: float
    drop_height: float | None
    drop_clearance: float | None


@dataclass
class SimulationConfig:
    steps: int
    plane_height: float
    gravity_z: float
    youngs_modulus: float
    poisson_ratio: float
    density: float
    dt: float
    substeps: int


@dataclass
class BindingConfig:
    knn_k: int


@dataclass
class RenderConfig:
    width: int
    height: int
    fovy_deg: float
    lookat: tuple[float, float, float]
    distance: float
    elevation_deg: float
    azimuth_deg: float
    fps: int
    render_every: int
    save_depth: bool


@dataclass
class MPMGaussianConfig:
    paths: PathsConfig
    sampling: SamplingConfig
    simulation: SimulationConfig
    binding: BindingConfig
    render: RenderConfig


def _resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_mpm_gaussian_config(path: str | Path) -> MPMGaussianConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    base_dir = path.parent

    paths = PathsConfig(
        input_ply=_resolve_path(base_dir, data["paths"]["input_ply"]),
        output_dir=_resolve_path(base_dir, data["paths"]["output_dir"]),
    )
    sampling = SamplingConfig(**data["sampling"])
    simulation = SimulationConfig(**data["simulation"])
    binding = BindingConfig(**data["binding"])
    render_data = dict(data["render"])
    render_data["lookat"] = tuple(render_data["lookat"])
    render = RenderConfig(**render_data)
    return MPMGaussianConfig(paths=paths, sampling=sampling, simulation=simulation, binding=binding, render=render)
