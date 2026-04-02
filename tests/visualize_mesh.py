import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def compute_equal_axes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = 0.5 * (vmin + vmax)
    half_extent = 0.5 * np.max(vmax - vmin)
    mins = center - half_extent
    maxs = center + half_extent
    return mins, maxs


def render_mesh(mesh: trimesh.Trimesh, output: Path, elev: float, azim: float) -> None:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    mins, maxs = compute_equal_axes(vertices)

    fig = plt.figure(figsize=(7, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    triangles = vertices[faces]
    collection = Poly3DCollection(
        triangles,
        facecolor=(0.88, 0.42, 0.50, 0.92),
        edgecolor=(0.20, 0.20, 0.20, 0.18),
        linewidth=0.15,
    )
    ax.add_collection3d(collection)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)
    ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(output.stem)
    ax.grid(False)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="PNG output path. Defaults to <mesh_stem>.png next to the mesh.",
    )
    parser.add_argument("--elev", type=float, default=20.0)
    parser.add_argument("--azim", type=float, default=45.0)
    args = parser.parse_args()

    mesh = trimesh.load_mesh(args.mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    output = args.output
    if output is None:
        output = args.mesh.with_suffix(".png")

    render_mesh(mesh, output, elev=args.elev, azim=args.azim)
    print(output)


if __name__ == "__main__":
    main()
