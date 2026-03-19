import tempfile
from pathlib import Path

import pytest
from conftest import BANANA_PLY

from gaussian_renderer import load_ply, save_ply


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_load_supersplat_ply():
    gd = load_ply(BANANA_PLY)
    assert len(gd) > 0
    assert gd.xyz.shape[1] == 3
    assert gd.rot.shape[1] == 4
    assert gd.scale.shape[1] == 3


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_save_load_roundtrip():
    gd = load_ply(BANANA_PLY)
    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        tmp = Path(f.name)
    try:
        save_ply(gd, tmp)
        gd2 = load_ply(tmp)
        assert len(gd2) == len(gd)
        assert gd2.xyz.shape == gd.xyz.shape
    finally:
        tmp.unlink(missing_ok=True)


def test_version():
    from gaussian_renderer import __version__

    assert __version__ == "0.1.10"
