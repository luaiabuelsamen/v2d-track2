"""The cap has to read high, never low, because a low read authorises spend nobody approved.

Two ways it read low in one afternoon: a `brev ls` that answered empty made every live box count
as zero minutes ($94.08 against a true $134.64), and a ledger path that depended on whether
V2D_SCRATCH happened to be exported pointed at a file that did not exist. A box that cannot be
reached is still billing, so it now counts as a full expected cell.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO = Path(__file__).parents[1]
GUARD = REPO / "brev/budget_guard.sh"


def run(script: str, *, path: Path, ledger: Path | None = None) -> subprocess.CompletedProcess:
    """Source the guard with a fake `brev` on PATH, and run `script` against it."""
    env = {
        "PATH": f"{path}:/usr/bin:/bin",
        "HOME": str(path),
        "V2D_SCRATCH": str(path),
    }
    if ledger:
        env["LEDGER"] = str(ledger)
    return subprocess.run(
        ["bash", "-c", f". {GUARD}\n{script}"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        cwd=REPO,
    )


def fake_brev(path: Path, rows: str) -> None:
    """A `brev` that prints `rows` as `brev ls`, under ~/.local/bin where the guard looks."""
    binary = path / ".local/bin"
    binary.mkdir(parents=True, exist_ok=True)
    script = binary / "brev"
    script.write_text(f'#!/bin/bash\n[ "$1" = ls ] && cat <<"EOF"\n{rows}\nEOF\nexit 0\n')
    script.chmod(0o755)


def test_comments_and_malformed_lines_are_skipped_not_counted(tmp_path: Path):
    """The ledger's header comments contain a date; counting its second field read 2026 minutes."""
    fake_brev(tmp_path, "head\nhead2")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text(
        "# 2026-10-26 10:05-10:43 first launch attempt: all boxes died at the md5 gate\n"
        "# Minutes are creation->deletion, which is what brev bills\n"
        "v2d-t2-b01 100\n"
        "v2d-t2-b02 50\n"
        "garbage line with no minutes\n"
    )
    done = run('echo "$(ab4_committed_minutes)"', path=tmp_path, ledger=ledger)
    assert done.stdout.strip().splitlines()[-1] == "150", done.stdout + done.stderr


def test_an_empty_brev_listing_does_not_make_live_boxes_free(tmp_path: Path):
    """What produced the $94.08: no rows, so every running box counted as zero minutes."""
    fake_brev(tmp_path, "header one\nheader two")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("v2d-t2-b01 600\n")
    done = run('echo "$(ab4_committed_minutes)"', path=tmp_path, ledger=ledger)
    # Nothing live to find, so the ledger alone -- but the listing is read three times and unioned,
    # so a transient empty answer cannot hide a box that the other two reads see.
    assert done.stdout.strip().splitlines()[-1] == "600"
    assert "brev ls" not in done.stderr


def test_a_box_that_cannot_be_reached_counts_as_a_full_cell(tmp_path: Path):
    """An unreachable box is still billing. Counting it as zero is the one error a cap must not make."""
    fake_brev(tmp_path, "header one\nheader two\nv2d-sw-ghost RUNNING COMPLETED READY")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("")
    done = run(
        'UNREADABLE_BOX_MINUTES=300; echo "$(ab4_committed_minutes)"', path=tmp_path, ledger=ledger
    )
    assert done.stdout.strip().splitlines()[-1] == "300", done.stdout + done.stderr
    assert "did not answer" in done.stderr


def test_the_ledger_path_does_not_depend_on_an_exported_variable():
    """It defaulted to $HOME/spend_ab4.txt, which does not exist, and read zero closed minutes."""
    text = GUARD.read_text()
    assert "V2D_LEDGER_DEFAULT=" in text
    assert '$(dirname "${BASH_SOURCE[0]}")/../../../spend_ab4.txt' not in text


def test_the_listing_is_read_more_than_once():
    assert "sort -u" in GUARD.read_text()


def test_a_box_over_the_cap_is_refused(tmp_path: Path):
    fake_brev(tmp_path, "header one\nheader two")
    ledger = tmp_path / "ledger.txt"
    ledger.write_text("v2d-t2-b01 1200\n")  # 20 h = $13.60 at $0.68
    refused = run(
        'NEW_BUDGET_USD=15 budget_ok_for_hours 5 "a box"; echo "rc=$?"',
        path=tmp_path,
        ledger=ledger,
    )
    assert "rc=1" in refused.stdout
    assert "BUDGET REFUSAL" in refused.stderr
    allowed = run(
        'NEW_BUDGET_USD=30 budget_ok_for_hours 5 "a box"; echo "rc=$?"',
        path=tmp_path,
        ledger=ledger,
    )
    assert "rc=0" in allowed.stdout
