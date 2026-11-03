"""The spend guard is the one process allowed to delete running work, so it is pinned by name.

The incident these tests exist for: the first version of this guard was written inline over a
launch with the box start time hardcoded as an epoch, and the epoch was a year out --
1761612220 is 2025-10-28, not 2026-10-28. It computed 26,280 box-hours, read $17,871 against a
$20 cap, and deleted three boxes that were 14M steps into training at 100% GPU and passing every
check (results/owed_answers.json).

The bug was not the arithmetic; it was that a process allowed to delete running work had no
opinion about whether its own numbers made sense. Each test below is one of those opinions, and
the first replays the incident's exact input.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "brev" / "cr_cap_guard.sh"
#: The epoch from the incident: 2025-10-28, a year before the boxes it judged.
INCIDENT_EPOCH = 1761612220
BOXES = ("v2d-sw-cr-b01", "v2d-sw-cr-b02", "v2d-sw-cr-b03")


def fake_brev(tmp_path: Path, boxes=BOXES) -> Path:
    """A `brev` that lists the boxes and records every delete instead of performing one."""
    path = tmp_path / "brev"
    listing = "\n".join(f"{b} RUNNING hyperstack_A6000" for b in boxes)
    path.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = ls ]; then\n'
        f'  printf "%s\\n" "{listing}"\n'
        "  exit 0\n"
        "fi\n"
        'if [ "$1" = delete ]; then\n'
        f'  echo "$2" >> "{tmp_path}/deleted.txt"\n'
        "  exit 0\n"
        "fi\n"
        "exit 0\n"
    )
    path.chmod(0o755)
    return path


def run_guard(tmp_path: Path, ledger: dict[str, int], *, cap="20", boxes=BOXES):
    ledger_path = tmp_path / "seen.txt"
    ledger_path.write_text("".join(f"{b} {t}\n" for b, t in ledger.items()))
    env = dict(os.environ, V2D_BREV=str(fake_brev(tmp_path, boxes)))
    done = subprocess.run(
        [
            "bash",
            str(GUARD),
            "--prefix",
            "v2d-sw-cr-",
            "--cap",
            cap,
            "--ledger",
            str(ledger_path),
            "--keep-ledger",
            "--once",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    deleted = tmp_path / "deleted.txt"
    return done, deleted.read_text().split() if deleted.exists() else []


def test_the_incident_input_deletes_nothing_and_says_it_is_a_bug(tmp_path: Path) -> None:
    """A year-old start time must be read as a broken guard, not as a $17,871 bill."""
    done, deleted = run_guard(tmp_path, dict.fromkeys(BOXES, INCIDENT_EPOCH))
    assert deleted == [], f"the guard deleted {deleted} on the exact input from the incident"
    assert done.returncode == 3, f"expected the plausibility refusal, got rc={done.returncode}"
    assert "GUARD BUG" in done.stdout


def test_a_future_dated_start_time_deletes_nothing(tmp_path: Path) -> None:
    """A clock skew the other way must not read as a negative bill that silently passes, either."""
    import time

    ahead = int(time.time()) + 86_400
    done, deleted = run_guard(tmp_path, dict.fromkeys(BOXES, ahead))
    assert deleted == [], f"the guard deleted {deleted} on a future-dated ledger"
    assert done.returncode in (0, 3), done.stdout


def test_a_box_under_the_age_floor_survives_even_over_cap(tmp_path: Path) -> None:
    """A guard that somehow fires in a box's first minutes cannot take down a fresh box."""
    import time

    now = int(time.time())
    # One box old enough to be over a $1 cap, but younger than the 0.25 h floor.
    done, deleted = run_guard(tmp_path, {BOXES[0]: now - 300}, cap="0.01", boxes=(BOXES[0],))
    assert deleted == [], f"deleted a {5}-minute-old box: {deleted}"
    assert "under the" in done.stdout and "floor" in done.stdout


def test_a_real_overspend_is_still_acted_on(tmp_path: Path) -> None:
    """The guard must remain a guard: doubting itself is not the same as never firing."""
    import time

    now = int(time.time())
    # Three boxes, 5 h each = 15 box-hours, under the 60 h plausibility bound, $10.20 at $0.68.
    done, deleted = run_guard(tmp_path, dict.fromkeys(BOXES, now - 5 * 3600), cap="10")
    assert sorted(deleted) == sorted(BOXES), f"expected all three deleted, got {deleted}"
    assert done.returncode == 0
    assert "CAP:" in done.stdout


@pytest.mark.parametrize("hours", [0.5, 2.0, 4.9])
def test_a_normal_run_is_left_alone(tmp_path: Path, hours: float) -> None:
    """The whole arm is 5 h/box at $0.68; nothing inside a $30 cap should be touched."""
    import time

    now = int(time.time())
    done, deleted = run_guard(tmp_path, {b: now - int(hours * 3600) for b in BOXES}, cap="30")
    assert deleted == [], done.stdout
    assert done.returncode == 0
