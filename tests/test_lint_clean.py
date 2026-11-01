"""`ruff check .` passes over the whole repository, so the gate cannot quietly rot.

Three one-off scripts had been failing lint for a week without anyone noticing, because the gate
that was actually run was scoped to `src tests tools/t1`. A scoped gate reports success while the
rest of the tree drifts, so the scope is the repository and the exemptions are declared in
`pyproject.toml` with a reason each.

`ruff format` is deliberately NOT asserted. Twenty-four files would be reformatted, and the
documentation cites code by `file:line` in dozens of places; a mechanical reflow would silently
falsify those citations in exchange for no behavioural gain.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUFF = ROOT / ".venv311" / "bin" / "ruff"


@pytest.mark.skipif(not RUFF.exists() and not shutil.which("ruff"), reason="ruff not installed")
def test_ruff_check_is_clean_over_the_whole_repo() -> None:
    ruff = str(RUFF) if RUFF.exists() else "ruff"
    done = subprocess.run(
        [ruff, "check", "."],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
