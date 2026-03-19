import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock gsplat before any project imports
gsplat_mock = MagicMock()
sys.modules["gsplat"] = gsplat_mock
sys.modules["gsplat.rendering"] = gsplat_mock.rendering

BANANA_PLY = Path(__file__).parent / "banana.ply"
