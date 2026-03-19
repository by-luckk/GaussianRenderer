import tempfile
from pathlib import Path

import pytest
from conftest import BANANA_PLY
from plyfile import PlyData, PlyElement
import numpy as np

from gaussian_renderer import is_super_splat_format, load_ply, save_ply
from gaussian_renderer.core.super_splat_loader import save_super_splat_ply


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_is_super_splat_format_true():
    plydata = PlyData.read(BANANA_PLY)
    assert is_super_splat_format(plydata) is True


def test_is_super_splat_format_false():
    vertex = PlyElement.describe(
        np.array([(0.0, 0.0, 0.0)], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")]),
        "vertex",
    )
    plydata = PlyData([vertex])
    assert is_super_splat_format(plydata) is False


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_save_super_splat_roundtrip(tmp_path):
    gd = load_ply(BANANA_PLY)
    out = tmp_path / "out.ply"
    save_super_splat_ply(gd, out)
    gd2 = load_ply(out)
    assert len(gd2) == len(gd)
    assert gd2.xyz.shape == gd.xyz.shape


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_save_3dgs_roundtrip(tmp_path):
    gd = load_ply(BANANA_PLY)
    out = tmp_path / "out_3dgs.ply"
    save_ply(gd, out)
    gd2 = load_ply(out)
    assert len(gd2) == len(gd)
    assert gd2.xyz.shape == gd.xyz.shape
