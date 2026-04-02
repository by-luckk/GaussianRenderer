from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from gaussian_renderer.core.gs_renderer import GSRenderer

from .binding import GaussianParticleBinding
from .config import RenderConfig
from .deformation import deform_gaussian_xyz


def make_camera_pose(render: RenderConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_rmat = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = camera_rmat @ Rotation.from_euler(
        "xyz",
        [np.deg2rad(render.elevation_deg), np.deg2rad(render.azimuth_deg), 0.0],
    ).as_matrix()
    lookat = np.asarray(render.lookat, dtype=np.float32)
    position = lookat + float(render.distance) * rotation[:3, 2]
    return position.reshape(1, 3), rotation.reshape(1, 9).astype(np.float32), np.array([render.fovy_deg], dtype=np.float32)


class MPMGaussianRendererBridge:
    def __init__(self, input_ply: str | Path, gaussian_data, binding: GaussianParticleBinding, render: RenderConfig):
        self.binding = binding
        self.render = render
        self.renderer = GSRenderer({"flower": str(input_ply)})
        self.renderer.apply_gaussian_deformation(xyz=gaussian_data.xyz, scale=gaussian_data.scale)
        self.cam_pos, self.cam_xmat, self.fovy = make_camera_pose(render)

    def update_from_particles(self, particle_pos):
        xyz = deform_gaussian_xyz(particle_pos, self.binding)
        self.renderer.apply_gaussian_deformation(xyz=xyz)
        return xyz

    def render_frame(self):
        return self.renderer.render_batch(
            self.cam_pos,
            self.cam_xmat,
            self.render.height,
            self.render.width,
            self.fovy,
        )
