import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock all optional/heavy deps before any project import
gsplat_mock = MagicMock()
sys.modules["gsplat"] = gsplat_mock
sys.modules["gsplat.rendering"] = gsplat_mock.rendering

mujoco_mock = MagicMock()
mujoco_mock.mjtObj.mjOBJ_BODY = 1
sys.modules.setdefault("mujoco", mujoco_mock)
sys.modules.setdefault("motrixsim", MagicMock())

BANANA_PLY = Path(__file__).parent / "banana.ply"
