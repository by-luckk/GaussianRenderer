import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_scene(show_viewer: bool):
    import genesis as gs

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=(0.0, 0.0, -9.8),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=8e-3,
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=1.0,
        ),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, -0.01),
            size=(2.0, 2.0, 0.02),
            fixed=True,
        ),
    )

    flower = scene.add_entity(
        material=gs.materials.PBD.Elastic(
            rho=1000.0,
            volume_compliance=0.0,
            stretch_compliance=0.0,
            bending_compliance=0.0,
            volume_relaxation=0.3,
            stretch_relaxation=0.2,
            bending_relaxation=0.2,
        ),
        morph=gs.morphs.Mesh(
            file=str(ROOT / "tests" / "assets" / "flower1.obj"),
            scale=0.12,
            pos=(0.0, 0.0, 0.25),
            euler=(0.0, 0.0, 0.0),
        ),
        surface=gs.surfaces.Default(),
    )

    scene.build()
    return scene, flower


def get_visual_mesh_state(flower):
    from genesis.utils.misc import qd_to_numpy

    vverts_pos, _, _ = flower.solver.get_state_render()
    if vverts_pos is None:
        raise RuntimeError("PBD solver did not expose render mesh state.")

    vertices = qd_to_numpy(
        vverts_pos,
        row_mask=range(flower.vvert_start, flower.vvert_end),
        col_mask=0,
        keepdim=False,
    ).astype(np.float32, copy=False)
    faces = np.asarray(flower.vmesh.faces, dtype=np.int32)
    return vertices, faces


def compute_plot_bounds(trajectory: np.ndarray, floor_z: float) -> tuple[np.ndarray, np.ndarray]:
    mins = trajectory.reshape(-1, 3).min(axis=0)
    maxs = trajectory.reshape(-1, 3).max(axis=0)

    mins[2] = min(mins[2], floor_z)
    maxs[2] = maxs[2] + 0.02

    center = 0.5 * (mins + maxs)
    half_extent = 0.5 * np.max(maxs - mins)
    plot_min = center - half_extent
    plot_max = center + half_extent
    plot_min[2] = min(plot_min[2], floor_z)
    return plot_min, plot_max


def render_mesh_trajectory(
    mesh_input: Path,
    render_dir: Path,
    gif_path: Path | None,
    render_every: int,
    fps: int,
    elev: float,
    azim: float,
    floor_z: float,
):
    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    data = np.load(mesh_input)
    vertices_traj = np.asarray(data["vertices"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int32)

    render_dir.mkdir(parents=True, exist_ok=True)
    plot_min, plot_max = compute_plot_bounds(vertices_traj, floor_z=floor_z)

    gif_writer = imageio.get_writer(gif_path, mode="I", fps=fps) if gif_path is not None else None
    saved_frames = 0

    try:
        for frame_idx in range(0, vertices_traj.shape[0], max(1, render_every)):
            vertices = vertices_traj[frame_idx]

            fig = plt.figure(figsize=(7, 7), dpi=180)
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=elev, azim=azim)

            mesh = Poly3DCollection(
                vertices[faces],
                facecolor=(0.88, 0.42, 0.50, 0.92),
                edgecolor=(0.18, 0.18, 0.18, 0.15),
                linewidth=0.12,
            )
            ax.add_collection3d(mesh)

            plane_x = np.linspace(plot_min[0], plot_max[0], 2)
            plane_y = np.linspace(plot_min[1], plot_max[1], 2)
            plane_xx, plane_yy = np.meshgrid(plane_x, plane_y)
            plane_zz = np.full_like(plane_xx, floor_z)
            ax.plot_surface(plane_xx, plane_yy, plane_zz, color="#d9d9d9", alpha=0.22, shade=False)

            ax.set_xlim(plot_min[0], plot_max[0])
            ax.set_ylim(plot_min[1], plot_max[1])
            ax.set_zlim(plot_min[2], plot_max[2])
            ax.set_box_aspect((1.0, 1.0, 1.0))
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(f"Flower PBD Mesh | frame {frame_idx:04d}")
            plt.tight_layout()

            frame_path = render_dir / f"frame_{frame_idx:04d}.png"
            fig.savefig(frame_path, bbox_inches="tight")
            plt.close(fig)

            if gif_writer is not None:
                gif_writer.append_data(imageio.imread(frame_path))

            saved_frames += 1
    finally:
        if gif_writer is not None:
            gif_writer.close()

    print(
        f"Rendered {saved_frames} mesh frames from {mesh_input} to {render_dir}"
        + (f" and {gif_path}" if gif_path is not None else "")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_pos.npy")
    parser.add_argument("--mesh-output", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_mesh.npz")
    parser.add_argument("--render-mesh", action="store_true", default=False)
    parser.add_argument("--render-only", action="store_true", default=False)
    parser.add_argument("--mesh-input", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_mesh.npz")
    parser.add_argument("--render-dir", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_mesh_frames")
    parser.add_argument("--render-gif", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_mesh.gif")
    parser.add_argument("--render-every", type=int, default=1)
    parser.add_argument("--render-fps", type=int, default=24)
    parser.add_argument("--camera-elev", type=float, default=22.0)
    parser.add_argument("--camera-azim", type=float, default=-58.0)
    parser.add_argument("--floor-z", type=float, default=0.0)
    args = parser.parse_args()

    if args.render_only:
        render_mesh_trajectory(
            mesh_input=args.mesh_input,
            render_dir=args.render_dir,
            gif_path=args.render_gif,
            render_every=args.render_every,
            fps=args.render_fps,
            elev=args.camera_elev,
            azim=args.camera_azim,
            floor_z=args.floor_z,
        )
        return

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene, flower = build_scene(show_viewer=args.viewer)

    horizon = args.steps if "PYTEST_VERSION" not in os.environ else min(args.steps, 5)
    particle_traj = []
    mesh_vertices_traj = []

    vertices, faces = get_visual_mesh_state(flower)
    particle_traj.append(flower.get_particles_pos().cpu().numpy())
    mesh_vertices_traj.append(vertices)

    for _ in range(horizon):
        scene.step()
        particle_traj.append(flower.get_particles_pos().cpu().numpy())
        vertices, faces = get_visual_mesh_state(flower)
        mesh_vertices_traj.append(vertices)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(particle_traj))

    args.mesh_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.mesh_output,
        vertices=np.asarray(mesh_vertices_traj, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int32),
    )

    print(f"Saved particle trajectory to {args.output}")
    print(f"Saved mesh trajectory to {args.mesh_output}")

    if args.render_mesh:
        render_mesh_trajectory(
            mesh_input=args.mesh_output,
            render_dir=args.render_dir,
            gif_path=args.render_gif,
            render_every=args.render_every,
            fps=args.render_fps,
            elev=args.camera_elev,
            azim=args.camera_azim,
            floor_z=args.floor_z,
        )


if __name__ == "__main__":
    main()
