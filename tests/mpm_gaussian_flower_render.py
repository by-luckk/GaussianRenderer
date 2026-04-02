import argparse
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gaussian_renderer.core.util_gau import load_ply
from genesis.gaussian.binding import build_gaussian_particle_binding
from genesis.gaussian.config import load_mpm_gaussian_config
from genesis.gaussian.gaussian_asset import prepare_gaussian_asset
from genesis.gaussian.mpm_scene import MPMGaussianScene
from genesis.gaussian.particle_sampling import prepare_particles_from_gaussians
from genesis.gaussian.renderer_bridge import MPMGaussianRendererBridge


def save_rgb_frame(rgb_tensor, frame_path: Path) -> None:
    rgb = rgb_tensor[0].detach().cpu().numpy()
    rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(frame_path, rgb)


def save_depth_frame(depth_tensor, frame_path: Path) -> None:
    depth = depth_tensor[0, ..., 0].detach().cpu().numpy().astype(np.float32)
    np.save(frame_path, depth)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "tests" / "config" / "mpm_gaussian_flower.yaml",
    )
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--viewer", action="store_true", default=False)
    args = parser.parse_args()

    config = load_mpm_gaussian_config(args.config)

    import genesis as gs

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    gaussian_data, transform = prepare_gaussian_asset(
        config.paths.input_ply,
        sampling=config.sampling,
        plane_height=config.simulation.plane_height,
    )
    gaussian_asset_raw = load_ply(config.paths.input_ply)
    particle_points, particle_size = prepare_particles_from_gaussians(
        gaussian_asset_raw,
        config=config.sampling,
        plane_height=config.simulation.plane_height,
        transform=transform,
    )

    scene = MPMGaussianScene(config, particle_points, particle_size, show_viewer=args.viewer)
    binding = build_gaussian_particle_binding(
        np.asarray(gaussian_data.xyz, dtype=np.float32),
        particle_points,
        k=config.binding.knn_k,
        device=gs.device,
    )
    bridge = MPMGaussianRendererBridge(config.paths.input_ply, gaussian_data, binding, config.render)

    output_dir = config.paths.output_dir
    rgb_dir = output_dir / "rgb"
    depth_dir = output_dir / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    if config.render.save_depth:
        depth_dir.mkdir(parents=True, exist_ok=True)

    frame_paths: list[Path] = []
    horizon = config.simulation.steps if "PYTEST_VERSION" not in os.environ else min(config.simulation.steps, 5)

    for frame_idx in range(horizon + 1):
        if frame_idx > 0:
            scene.step()

        particle_pos = scene.get_particles_pos()
        bridge.update_from_particles(particle_pos)
        rgb_tensor, depth_tensor = bridge.render_frame()

        if frame_idx % max(1, config.render.render_every) != 0:
            continue

        frame_path = rgb_dir / f"rgb_{frame_idx:04d}.png"
        save_rgb_frame(rgb_tensor, frame_path)
        frame_paths.append(frame_path)

        if config.render.save_depth:
            save_depth_frame(depth_tensor, depth_dir / f"depth_{frame_idx:04d}.npy")

    gif_path = output_dir / "render.gif"
    with imageio.get_writer(gif_path, mode="I", fps=config.render.fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))

    print(f"Rendered {len(frame_paths)} frames to {rgb_dir}")
    print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":
    main()
