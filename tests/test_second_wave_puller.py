"""Guards on the watcher that decides what goes on the board.

Three failures here would be silent and expensive: a box mapped to the wrong arm (a clip scored
against another arm's stage), a shell variable that bash owns (`GROUPS=` printed two errors and
skipped the adoption decision entirely on the first run of this script), and the e01 caveat
dropped from the adopt-check call, which would flatter B5 by scoring it against its own easier
reference.

It also pins what the puller must never do: create, restart or delete a box.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
PULLER = REPO / "brev/second_wave_puller.sh"
SCRIPTS = sorted((REPO / "brev").glob("*.sh"))

# Names bash owns. Assigning to one of these does not fail loudly: `GROUPS=(...)` prints
# "variable may not be assigned value" to stderr and carries on with the caller's group ids.
BASH_RESERVED = (
    "GROUPS",
    "UID",
    "EUID",
    "PPID",
    "BASHPID",
    "RANDOM",
    "SECONDS",
    "LINENO",
    "FUNCNAME",
    "BASH_SOURCE",
    "SHELLOPTS",
    "BASH_ARGV0",
    "HISTCMD",
)
ASSIGNMENT = re.compile(
    r"^\s*(?:local\s+|declare\s+(?:-\w+\s+)*|export\s+)?("
    + "|".join(BASH_RESERVED)
    + r")(?:=|\+=|\[)"
)


def arm_of(name: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(PULLER), "--arm-of", name], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    ("box", "arm"),
    [
        ("v2d-sw-a-b01", "A"),
        ("v2d-sw-a-b07", "A"),
        ("v2d-sw-b1-b01", "B1"),
        ("v2d-sw-b2-b01", "B2"),
        ("v2d-sw-b3-b01", "B3"),
        ("v2d-sw-b4-b02", "B4"),
        ("v2d-sw-b5-b01", "B5"),
        # B4x keeps its small x everywhere -- stage dir, map, results dir -- so the mapping must
        # return the canonical spelling and not the upper-cased box name.
        ("v2d-sw-b4x-b01", "B4x"),
        ("v2d-sw-b4x-b08", "B4x"),
    ],
)
def test_every_live_box_maps_to_its_own_arm(box: str, arm: str):
    done = arm_of(box)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == arm


@pytest.mark.parametrize(
    "box", ["v2d-t2-b01", "v2d-ab-raw-e03", "", "v2d-sw-c-b01", "v2d-sw-b4y-b01"]
)
def test_a_box_from_another_run_is_refused_rather_than_guessed(box: str):
    done = arm_of(box)
    assert done.returncode == 1
    assert "no arm owns" in done.stderr


def test_the_puller_parses_under_bash():
    done = subprocess.run(["bash", "-n", str(PULLER)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_no_brev_script_assigns_a_name_bash_owns(script: Path):
    offenders = [
        f"{script.name}:{number}: {line.strip()}"
        for number, line in enumerate(script.read_text().splitlines(), start=1)
        if ASSIGNMENT.match(line)
    ]
    assert not offenders, "\n".join(offenders)


def test_the_puller_never_touches_an_instance():
    """'Do not relaunch or create boxes' is the instruction this watcher runs under."""
    text = PULLER.read_text()
    for forbidden in ("brev create", "brev delete", "brev start", "brev stop", "gate_launch.sh"):
        assert forbidden not in text, forbidden
    # The one brev subcommand it may use is the read-only listing.
    assert "brev ls" in text or "$BREV ls" in text


def test_the_e01_caveat_is_in_the_adoption_call():
    assert "--reference-from-old e01" in PULLER.read_text()


def test_it_scores_through_the_shared_pass_so_selection_cannot_see_a_score():
    text = PULLER.read_text()
    assert "score_pulled.sh" in text
    # Same --results dir as the launchers, which is what shares score_pulled's flock.
    assert '--results "$RESULTS"' in text


def test_the_clip_count_comes_from_the_staged_maps_not_a_hand_written_number():
    """B4x was added to the watch mid-run; a hardcoded 28 would have ended it eight clips early."""
    text = PULLER.read_text()
    assert "EXPECTED_CLIPS=\"$(expected_clips)\"" in text
    assert "map_$arm.txt" in text


def test_every_arm_of_the_wave_is_watched():
    text = PULLER.read_text()
    for arm in ("A", "B1", "B2", "B3", "B4", "B5", "B4x"):
        assert f" {arm}" in text.split("ARMS=", 1)[1].split("\n", 1)[0] or f'"{arm} ' in text
