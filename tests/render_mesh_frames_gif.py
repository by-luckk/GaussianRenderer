import argparse
from pathlib import Path

import imageio.v2 as imageio


ROOT = Path(__file__).resolve().parents[1]


def collect_frames(frame_dir: Path) -> list[Path]:
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"No PNG frames found in: {frame_dir}")
    return frames


def build_gif(frame_dir: Path, output: Path, fps: int) -> None:
    frames = collect_frames(frame_dir)
    output.parent.mkdir(parents=True, exist_ok=True)

    with imageio.get_writer(output, mode="I", fps=fps) as writer:
        for frame_path in frames:
            writer.append_data(imageio.imread(frame_path))

    print(f"Rendered GIF with {len(frames)} frames from {frame_dir} to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "frame_dir",
        type=Path,
        nargs="?",
        default=ROOT / "tests" / "results" / "flower1_pbd_mesh_frames",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tests" / "results" / "flower1_pbd_mesh_from_frames.gif",
    )
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    build_gif(frame_dir=args.frame_dir, output=args.output, fps=args.fps)


if __name__ == "__main__":
    main()
