import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--output", type=Path, default=ROOT / "tests" / "results" / "flower1_pbd_pos.npy")
    args = parser.parse_args()

    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=2e-3,
            substeps=10,
            gravity=(0.0, 0.0, -9.8),
        ),
        pbd_options=gs.options.PBDOptions(
            particle_size=8e-3,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        material=gs.materials.Rigid(
            needs_coup=True,
            coup_friction=1.0,
        ),
        morph=gs.morphs.Plane(),
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

    horizon = args.steps if "PYTEST_VERSION" not in os.environ else min(args.steps, 5)
    all_pos = []

    for _ in range(horizon):
        scene.step()
        all_pos.append(flower.get_particles_pos().cpu().numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, np.asarray(all_pos))


if __name__ == "__main__":
    main()
