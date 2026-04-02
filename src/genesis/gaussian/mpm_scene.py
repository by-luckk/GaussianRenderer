import numpy as np

from .config import MPMGaussianConfig


class MPMGaussianScene:
    def __init__(self, config: MPMGaussianConfig, points: np.ndarray, particle_size: float, show_viewer: bool = False):
        import genesis as gs

        xy_half_extent = max(0.2, float(np.max(np.abs(points[:, :2]))) + 0.08)
        z_lower = min(0.0, config.simulation.plane_height)
        z_upper = max(0.8, float(points[:, 2].max()) + 0.35)

        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=config.simulation.dt,
                substeps=config.simulation.substeps,
                gravity=(0.0, 0.0, config.simulation.gravity_z),
                floor_height=config.simulation.plane_height,
            ),
            mpm_options=gs.options.MPMOptions(
                particle_size=particle_size,
                lower_bound=(-xy_half_extent, -xy_half_extent, z_lower),
                upper_bound=(xy_half_extent, xy_half_extent, z_upper),
            ),
            vis_options=gs.options.VisOptions(
                visualize_mpm_boundary=show_viewer,
            ),
            show_viewer=show_viewer,
        )

        scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=1.0,
            ),
            morph=gs.morphs.Plane(pos=(0.0, 0.0, config.simulation.plane_height)),
        )

        flower = scene.add_entity(
            material=gs.materials.MPM.Elastic(
                E=config.simulation.youngs_modulus,
                nu=config.simulation.poisson_ratio,
                rho=config.simulation.density,
            ),
            morph=gs.morphs.Nowhere(n_particles=len(points)),
            surface=gs.surfaces.Default(
                color=(0.92, 0.37, 0.46),
                vis_mode="particle",
            ),
        )

        scene.build()
        flower.set_particles_pos(points[None, ...])
        flower.set_particles_vel(np.zeros_like(points)[None, ...])
        flower.set_particles_active(np.ones((1, len(points)), dtype=bool))

        self.scene = scene
        self.flower = flower

    def step(self) -> None:
        self.scene.step()

    def get_particles_pos(self):
        return self.flower.get_particles_pos()
