from unittest.mock import patch

import numpy as np
import pytest
from conftest import BANANA_PLY


@pytest.mark.skipif(not BANANA_PLY.exists(), reason="banana.ply not found")
def test_apply_gaussian_deformation_updates_xyz_and_scale():
    with (
        patch("gaussian_renderer.core.gs_renderer.GSPLAT_AVAILABLE", True),
        patch("torch.Tensor.cuda", lambda self: self),
    ):
        from gaussian_renderer.core.gs_renderer import GSRenderer

        renderer = GSRenderer({"banana": str(BANANA_PLY)})
        xyz_new = renderer.gaussians.xyz.detach().cpu().numpy() + 0.1
        scale_new = renderer.gaussians.scale.detach().cpu().numpy() * 1.2

        renderer.apply_gaussian_deformation(xyz=xyz_new, scale=scale_new)

        np.testing.assert_allclose(renderer.gaussians.xyz.detach().cpu().numpy(), xyz_new)
        np.testing.assert_allclose(renderer.gaussians.scale.detach().cpu().numpy(), scale_new)
