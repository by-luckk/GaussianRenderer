# Gaussian Renderer

A Gaussian Splatting Renderer and Tools package.

[中文文档](README_zh.md)

This repository primarily provides rendering capabilities for Gaussian Splatting models. It is developed as a component of the **DISCOVERSE** project.

For detailed usage within the simulation environment, see: [https://github.com/TATP-233/DISCOVERSE](https://github.com/TATP-233/DISCOVERSE)

## Requirements

Python >= 3.10

## Installation

```bash
uv add gaussian-renderer
# or: pip install gaussian-renderer
```

From source:
```bash
git clone https://github.com/TATP-233/GaussainRenderer.git
cd GaussainRenderer
uv pip install .
# or: pip install .
```

### Optional extras

```bash
uv add "gaussian-renderer[viewer]"   # OpenGL viewer (glfw, PyOpenGL)
uv add "gaussian-renderer[mujoco]"   # MuJoCo integration
uv add "gaussian-renderer[motrix]"   # MotrixSim integration

# Combine as needed
uv add "gaussian-renderer[viewer,mujoco]"
# or: pip install ".[viewer,mujoco]"
```

## Usage

### Command-line tools

**`gs-viewer`** — OpenGL viewer for `.ply` models
```bash
gs-viewer path/to/model.ply
```
Controls: Left mouse = rotate, Right/Middle = pan, Scroll = zoom, Up/Down = SH degree, Drag & drop = load file

**`gs-compress`** — Compress 3DGS PLY to SuperSplat format
```bash
gs-compress input.ply
gs-compress input.ply -o output.ply
gs-compress models/          # batch
```

**`gs-transform`** — Apply translation/rotation/scale to a model
```bash
gs-transform input.ply -o output.ply -t 0 1 0 -s 2.0
gs-transform input.ply -r 0 0 0 1   # rotation quaternion xyzw
# --compress: save as compressed PLY
```

### Python API

```bash
uv run python -m gaussian_renderer.simple_viewer path/to/model.ply
uv run python -m gaussian_renderer.supersplat_compress input.ply
uv run python -m gaussian_renderer.transform_gs_model input.ply
```

### Local simulation examples

The repository also includes local Genesis-based simulation scripts under `tests/`.

For MPM-driven Gaussian rendering, use the YAML config at `tests/config/mpm_gaussian_flower.yaml`:

```bash
uv run python -m tests.mpm_gaussian_flower_render
```

The config supports two deformation modes:

- `knn_xyz`: keep the original KNN position update and only deform Gaussian centers
- `knn_rigid`: keep the same KNN position update, then fit a local rigid rotation from the same KNN neighborhood and update Gaussian `rot`

Run the PBD flower example:

```bash
uv run python -m tests.pbd_fall_flower
```

Common options:

```bash
uv run python -m tests.pbd_fall_flower --cpu
uv run python -m tests.pbd_fall_flower --steps 500
uv run python -m tests.pbd_fall_flower --output tests/results/flower1_pbd_pos.npy
uv run python -m tests.pbd_fall_flower --render-mesh
```

This script simulates `tests/assets/flower1.obj` as a PBD elastic body and saves:

- particle trajectory to `tests/results/flower1_pbd_pos.npy` with shape `(T, N, 3)`
- mesh trajectory to `tests/results/flower1_pbd_mesh.npz` with keys `vertices` and `faces`

To render the full surface mesh motion after simulation:

```bash
uv run python -m tests.pbd_fall_flower --render-mesh
```

This writes mesh frames to `tests/results/flower1_pbd_mesh_frames/` and a GIF to `tests/results/flower1_pbd_mesh.gif`.

To render an existing saved mesh trajectory without rerunning simulation:

```bash
uv run python -m tests.pbd_fall_flower --render-only
```

### Mesh visualization

For a simple offline mesh preview, render a mesh file to PNG:

```bash
uv run python -m tests.visualize_mesh tests/assets/flower1.obj
```

If mesh repair exports a repaired mesh, you can preview that file directly:

```bash
uv run python -m tests.visualize_mesh tests/assets/flower1_repaired.obj
```

By default the script writes a PNG next to the input mesh. You can also choose the output path and camera view:

```bash
uv run python -m tests.visualize_mesh tests/assets/flower1_repaired.obj \
  --output tests/results/flower1_repaired_view.png \
  --elev 25 --azim 35
```

## Development

```bash
uv pip install ".[dev]"
# or: pip install ".[dev]"
make lint       # ruff check
make format     # ruff format
make typecheck  # mypy
make test       # pytest
make ci         # all of the above
```

## Citation

```bibtex
@article{jia2025discoverse,
      title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
      author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
      journal={arXiv preprint arXiv:2507.21981},
      year={2025},
      url={https://arxiv.org/abs/2507.21981}
}
```
