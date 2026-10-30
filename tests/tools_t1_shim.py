"""Import the `tools/t1` scripts under a module name, since `tools/` is not a package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "tools" / "t1" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rank_swap = _load("rank_swap")
gate_margins = _load("gate_margins")
