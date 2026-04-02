import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


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


def prepare_flower_particles(
    ply_path: Path,
    max_particles: int,
    target_extent: float,
    drop_height: float | None,
    plane_height: float,
    drop_clearance: float | None,
    opacity_quantile: float,
) -> tuple[np.ndarray, float]:
    from gaussian_renderer.core.util_gau import load_ply

    flower = load_ply(ply_path)

    points = np.asarray(flower.xyz, dtype=np.float32)
    opacity = np.asarray(flower.opacity, dtype=np.float32).reshape(-1)

    opacity_quantile = float(np.clip(opacity_quantile, 0.0, 0.99))
    opacity_cutoff = float(np.quantile(opacity, opacity_quantile))
    keep_mask = opacity >= opacity_cutoff

    if keep_mask.sum() >= min(max_particles, len(points)):
        points = points[keep_mask]
        opacity = opacity[keep_mask]

    bbox_min = points.min(axis=0)
    bbox_max = points.max(axis=0)
    center = 0.5 * (bbox_min + bbox_max)
    points = points - center

    span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-6)
    scale = float(target_extent / np.max(span))
    points *= scale

    if len(points) > max_particles:
        scaled_span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-3)
        # 3DGS points mainly lie on thin surfaces, so we start from a finer voxel size
        # than a full-volume estimate and only grow it if we still exceed the budget.
        voxel_size = float(0.45 * np.cbrt(np.prod(scaled_span) / max_particles))
        sampled_idx = np.arange(len(points), dtype=np.int64)

        for _ in range(8):
            sampled_idx = voxel_downsample_indices(points, opacity, voxel_size)
            if len(sampled_idx) <= max_particles:
                break
            voxel_size *= 1.35

        if len(sampled_idx) < max_particles:
            rng = np.random.default_rng(0)
            remaining_idx = np.setdiff1d(np.arange(len(points), dtype=np.int64), sampled_idx, assume_unique=True)
            n_extra = min(max_particles - len(sampled_idx), len(remaining_idx))
            if n_extra > 0:
                extra_idx = rng.choice(remaining_idx, size=n_extra, replace=False)
                sampled_idx = np.concatenate([sampled_idx, extra_idx])
                sampled_idx.sort()

        points = points[sampled_idx]
        opacity = opacity[sampled_idx]

    if len(points) > max_particles:
        keep = np.argsort(opacity)[-max_particles:]
        keep.sort()
        points = points[keep]

    points[:, 2] -= points[:, 2].min()
    scaled_height = float(points[:, 2].max() - points[:, 2].min())

    if drop_height is None:
        clearance = drop_clearance
        if clearance is None:
            clearance = max(0.06, 0.6 * scaled_height)
        bottom_height = plane_height + float(clearance)
    else:
        bottom_height = float(drop_height)

    points += np.array([0.0, 0.0, bottom_height], dtype=np.float32)

    final_span = np.maximum(points.max(axis=0) - points.min(axis=0), 1e-3)
    particle_size = float(np.clip(np.cbrt(np.prod(final_span) / len(points)), 3e-3, 1.2e-2))

    return points.astype(np.float32), particle_size


def build_scene(
    particle_size: float,
    points: np.ndarray,
    youngs_modulus: float,
    poisson_ratio: float,
    density: float,
    plane_height: float,
    gravity_z: float,
    show_viewer: bool,
):
    import genesis as gs

    xy_half_extent = max(0.2, float(np.max(np.abs(points[:, :2]))) + 0.08)
    z_lower = min(0.0, plane_height)
    z_upper = max(0.8, float(points[:, 2].max()) + 0.35)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=(0.0, 0.0, gravity_z),
            floor_height=plane_height,
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
        morph=gs.morphs.Plane(pos=(0.0, 0.0, plane_height)),
    )

    flower = scene.add_entity(
        material=gs.materials.MPM.Elastic(
            E=youngs_modulus,
            nu=poisson_ratio,
            rho=density,
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

    return scene, flower


def compute_plot_bounds(trajectory: np.ndarray, plane_height: float) -> tuple[np.ndarray, np.ndarray]:
    mins = trajectory.reshape(-1, 3).min(axis=0)
    maxs = trajectory.reshape(-1, 3).max(axis=0)

    mins[2] = min(plane_height, mins[2] - 0.02)
    maxs[2] = maxs[2] + 0.02

    span = np.max(maxs - mins)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span

    plot_min = center - half
    plot_max = center + half
    plot_min[2] = min(plot_min[2], plane_height)

    return plot_min, plot_max


def render_point_trajectory(
    trajectory_path: Path,
    render_dir: Path,
    gif_path: Path | None,
    render_every: int,
    point_size: float,
    fps: int,
    elev: float,
    azim: float,
    plane_height: float,
):
    import matplotlib

    matplotlib.use("Agg")

    import imageio.v2 as imageio
    import matplotlib.pyplot as plt

    trajectory_path = Path(trajectory_path)
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_path}")

    trajectory = np.load(trajectory_path)
    if trajectory.ndim != 3 or trajectory.shape[-1] != 3:
        raise ValueError(f"Expected trajectory shape (T, N, 3), got {trajectory.shape}")

    render_every = max(1, int(render_every))
    render_dir.mkdir(parents=True, exist_ok=True)

    plot_min, plot_max = compute_plot_bounds(trajectory, plane_height=plane_height)
    plane_z = plane_height

    gif_writer = imageio.get_writer(gif_path, mode="I", fps=fps) if gif_path is not None else None
    saved_frames = 0

    try:
        for frame_idx in range(0, trajectory.shape[0], render_every):
            frame = trajectory[frame_idx]

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)

            colors = frame[:, 2]
            ax.scatter(
                frame[:, 0],
                frame[:, 1],
                frame[:, 2],
                c=colors,
                cmap="viridis",
                s=point_size,
                alpha=0.9,
                linewidths=0.0,
            )

            plane_x = np.linspace(plot_min[0], plot_max[0], 2)
            plane_y = np.linspace(plot_min[1], plot_max[1], 2)
            plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
            plane_zz = np.full_like(plane_xx, plane_z)
            ax.plot_surface(plane_xx, plane_yy, plane_zz, color="#d9d9d9", alpha=0.25, shade=False)

            ax.set_xlim(plot_min[0], plot_max[0])
            ax.set_ylim(plot_min[1], plot_max[1])
            ax.set_zlim(plot_min[2], plot_max[2])
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Flower MPM Point Render | frame {frame_idx:04d}")
            plt.tight_layout()

            frame_path = render_dir / f"frame_{frame_idx:04d}.png"
            fig.savefig(frame_path, dpi=180)
            plt.close(fig)

            if gif_writer is not None:
                gif_writer.append_data(imageio.imread(frame_path))

            saved_frames += 1
    finally:
        if gif_writer is not None:
            gif_writer.close()

    print(
        f"Rendered {saved_frames} point-cloud frames from {trajectory_path} to {render_dir}"
        + (f" and {gif_path}" if gif_path is not None else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--render-only", action="store_true", default=False)
    parser.add_argument("--render-points", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-particles", type=int, default=8192)
    parser.add_argument("--target-extent", type=float, default=0.12)
    parser.add_argument("--drop-height", type=float, default=None)
    parser.add_argument("--drop-clearance", type=float, default=None)
    parser.add_argument("--plane-height", type=float, default=0.0)
    parser.add_argument("--gravity-z", type=float, default=-9.8)
    parser.add_argument("--opacity-quantile", type=float, default=0.3)
    parser.add_argument("--youngs-modulus", type=float, default=8e4)
    parser.add_argument("--poisson-ratio", type=float, default=0.28)
    parser.add_argument("--density", type=float, default=500.0)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "results" / "flower1_mpm_pos.npy")
    parser.add_argument("--trajectory-input", type=Path, default=ROOT / "tests" / "results" / "flower1_mpm_pos.npy")
    parser.add_argument("--render-dir", type=Path, default=ROOT / "tests" / "results" / "flower1_mpm_points")
    parser.add_argument("--render-gif", type=Path, default=ROOT / "tests" / "results" / "flower1_mpm_points.gif")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=24)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--camera-elev", type=float, default=22.0)
    parser.add_argument("--camera-azim", type=float, default=-62.0)
    parser.add_argument("--input", type=Path, default=ROOT / "tests" / "assets" / "flower1.ply")
    args = parser.parse_args()

    if args.render_only:
        render_point_trajectory(
            trajectory_path=args.trajectory_input,
            render_dir=args.render_dir,
            gif_path=args.render_gif,
            render_every=args.render_every,
            point_size=args.point_size,
            fps=args.render_fps,
            elev=args.camera_elev,
            azim=args.camera_azim,
            plane_height=args.plane_height,
        )
        return

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    points, particle_size = prepare_flower_particles(
        ply_path=args.input,
        max_particles=args.max_particles,
        target_extent=args.target_extent,
        drop_height=args.drop_height,
        plane_height=args.plane_height,
        drop_clearance=args.drop_clearance,
        opacity_quantile=args.opacity_quantile,
    )

    scene, flower = build_scene(
        particle_size=particle_size,
        points=points,
        youngs_modulus=args.youngs_modulus,
        poisson_ratio=args.poisson_ratio,
        density=args.density,
        plane_height=args.plane_height,
        gravity_z=args.gravity_z,
        show_viewer=args.viewer,
    )

    horizon = args.steps if "PYTEST_VERSION" not in os.environ else min(args.steps, 5)
    trajectory = [flower.get_particles_pos().cpu().numpy()]

    for _ in range(horizon):
        scene.step()
        trajectory.append(flower.get_particles_pos().cpu().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(trajectory))

    print(
        f"Saved MPM trajectory with {points.shape[0]} particles, "
        f"particle_size={particle_size:.5f}, gravity_z={args.gravity_z:.2f}, "
        f"plane_height={args.plane_height:.3f}, initial_bottom_z={points[:, 2].min():.3f} "
        f"to {args.output}"
    )

    if args.render_points:
        render_point_trajectory(
            trajectory_path=args.output,
            render_dir=args.render_dir,
            gif_path=args.render_gif,
            render_every=args.render_every,
            point_size=args.point_size,
            fps=args.render_fps,
            elev=args.camera_elev,
            azim=args.camera_azim,
            plane_height=args.plane_height,
        )


if __name__ == "__main__":
    main()
