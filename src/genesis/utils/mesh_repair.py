import numpy as np
import trimesh
from pathlib import Path

import genesis as gs


def _mesh_state(mesh: trimesh.Trimesh) -> tuple[bool, bool, float, int, int]:
    return (
        bool(mesh.is_watertight),
        bool(mesh.is_winding_consistent),
        float(abs(mesh.volume)),
        int(len(mesh.vertices)),
        int(len(mesh.faces)),
    )


def _export_repaired_mesh(mesh: trimesh.Trimesh, source_path: str | Path | None) -> None:
    if source_path is None:
        return

    source_path = Path(source_path)
    export_path = source_path.with_name(f"{source_path.stem}_repaired{source_path.suffix}")
    mesh.export(export_path)
    gs.logger.info(f"Exported repaired mesh to: {export_path}")


def _default_thickness(mesh: trimesh.Trimesh, thickness: float | None) -> float:
    if thickness is not None and thickness > 0.0:
        return float(thickness)

    diag = float(np.linalg.norm(mesh.extents))
    return max(diag * 0.005, 1e-5)


def _solidify_shell(
    mesh: trimesh.Trimesh,
    thickness: float | None = None,
    aggressive: bool = False,
) -> trimesh.Trimesh:
    shell = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))
    if len(shell.vertices) == 0 or len(shell.faces) == 0:
        return shell

    shell.remove_unreferenced_vertices()
    shell.merge_vertices()

    vertex_normals = np.array(shell.vertex_normals, dtype=np.float64, copy=True)
    if vertex_normals.shape != shell.vertices.shape or not np.isfinite(vertex_normals).all():
        trimesh.repair.fix_normals(shell, multibody=True)
        vertex_normals = np.array(shell.vertex_normals, dtype=np.float64, copy=True)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    invalid = ~np.isfinite(norms[:, 0]) | (norms[:, 0] < 1e-12)
    vertex_normals[invalid] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    vertex_normals /= np.maximum(np.linalg.norm(vertex_normals, axis=1, keepdims=True), 1e-12)

    solid_thickness = _default_thickness(shell, thickness)
    offset = vertex_normals * (solid_thickness * 0.5)

    outer = np.asarray(shell.vertices, dtype=np.float64) + offset
    inner = np.asarray(shell.vertices, dtype=np.float64) - offset
    n_verts = len(shell.vertices)

    outer_faces = np.asarray(shell.faces, dtype=np.int64)
    inner_faces = outer_faces[:, ::-1] + n_verts

    edge_counts = np.bincount(shell.edges_unique_inverse, minlength=len(shell.edges_unique))
    boundary_edges = shell.edges_unique[edge_counts == 1]
    side_faces = np.empty((len(boundary_edges) * 2, 3), dtype=np.int64)
    for i, (u, v) in enumerate(boundary_edges):
        side_faces[2 * i] = np.array([u, v, v + n_verts], dtype=np.int64)
        side_faces[2 * i + 1] = np.array([u, v + n_verts, u + n_verts], dtype=np.int64)

    solid = trimesh.Trimesh(
        vertices=np.vstack([outer, inner]),
        faces=np.vstack([outer_faces, inner_faces, side_faces]),
        process=False,
    )
    solid.process(validate=True)
    solid.remove_unreferenced_vertices()
    solid.merge_vertices()
    trimesh.repair.fix_normals(solid, multibody=True)
    trimesh.repair.fix_winding(solid)
    if solid.volume < 0.0:
        solid.invert()
    if solid.is_watertight and has_nonzero_volume(solid):
        return solid

    if not aggressive:
        return solid

    pitch = _default_thickness(shell, thickness)
    voxels = shell.voxelized(pitch=pitch).fill()
    solid = voxels.marching_cubes
    solid.apply_transform(voxels.transform)
    solid.process(validate=True)
    solid.remove_unreferenced_vertices()
    solid.merge_vertices()
    trimesh.repair.fix_normals(solid, multibody=True)
    trimesh.repair.fix_winding(solid)
    if solid.volume < 0.0:
        solid.invert()
    return solid


def repair_volume_mesh(
    mesh: trimesh.Trimesh,
    context: str = "mesh",
    solidify: bool = False,
    thickness: float | None = None,
    aggressive_solidify: bool = False,
    source_path: str | Path | None = None,
) -> trimesh.Trimesh:
    repaired = mesh.copy(**(dict(include_cache=True) if isinstance(mesh, trimesh.Trimesh) else {}))
    before = _mesh_state(repaired)

    repaired.process(validate=True)
    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()

    if len(repaired.faces) > 0:
        repaired.update_faces(repaired.unique_faces())
        repaired.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(repaired, multibody=True)
    trimesh.repair.fix_winding(repaired)
    trimesh.repair.fill_holes(repaired)
    repaired.remove_unreferenced_vertices()
    repaired.merge_vertices()

    if repaired.volume < 0.0:
        repaired.invert()

    if solidify and (not repaired.is_watertight or not has_nonzero_volume(repaired)):
        repaired = _solidify_shell(repaired, thickness=thickness, aggressive=aggressive_solidify)
        repaired = repair_volume_mesh(
            repaired,
            context=f"{context} after solidify",
            solidify=False,
            aggressive_solidify=aggressive_solidify,
        )

    after = _mesh_state(repaired)
    if before != after:
        gs.logger.info(
            f"Repaired {context}: watertight {before[0]} -> {after[0]}, "
            f"winding {before[1]} -> {after[1]}, "
            f"volume {before[2]:.6e} -> {after[2]:.6e}, "
            f"verts {before[3]} -> {after[3]}, faces {before[4]} -> {after[4]}"
        )
        _export_repaired_mesh(repaired, source_path)

    return repaired


def has_nonzero_volume(mesh: trimesh.Trimesh, eps: float | None = None) -> bool:
    threshold = gs.EPS if eps is None else eps
    return bool(np.isfinite(mesh.volume)) and abs(float(mesh.volume)) > threshold
