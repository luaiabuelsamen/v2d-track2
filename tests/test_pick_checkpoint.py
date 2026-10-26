"""Known answers for the checkpoint selection, because it decides what gets submitted.

The rule reads only the noisy input reference, so it cannot be checked against a score -- which is
the whole point of it. What can be checked is that it picks the checkpoint we can work out by hand
from the trajectory, that it refuses the ones that have lost the object, and that the lift-lag
artefact which made the bare z-line rule throw away 6 of e24's 7 checkpoints is not a drop.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.select.checkpoints import first_z_drop, main, read_cell

FRAMES = 100


def write_checkpoint(
    cell: Path,
    steps: int,
    achieved: np.ndarray,
    reference: np.ndarray,
    *,
    auc: float = 0.2,
    mppe_cm: float = 5.0,
    sha: str | None = None,
    policy: bytes | None = None,
    valid: np.ndarray | None = None,
) -> Path:
    """One `policy_<steps>/` directory, the way the evaluator writes it."""
    directory = cell / f"policy_{steps}"
    directory.mkdir(parents=True)
    frames = len(achieved)
    columns = {"environment_valid": np.ones(frames, dtype=bool) if valid is None else valid}
    for name, data in (("achieved", achieved), ("reference", reference)):
        for index, axis in enumerate("xyz"):
            columns[f"{name}_position_{axis}"] = data[:, index].astype(np.float64)
    pq.write_table(pa.table(columns), directory / "objects.parquet")
    if policy is not None:
        (directory / "policy.safetensors").write_bytes(policy)
        sha = sha or hashlib.sha256(policy).hexdigest()
    (directory / "metrics.json").write_text(
        json.dumps(
            {
                "environment_steps": steps,
                "checkpoint_sha256": sha or f"sha-{steps}",
                "metrics": {"add_auc": auc, "mppe_cm": mppe_cm, "chord_sr": 0.0},
            }
        )
    )
    return directory


def held(offset_m: float, frames: int = FRAMES) -> tuple[np.ndarray, np.ndarray]:
    """A reference on a table, and an achieved pose a fixed distance from it."""
    reference = np.tile(np.array([0.3, 0.1, 0.8]), (frames, 1))
    achieved = reference + np.array([offset_m, 0.0, 0.0])
    return achieved, reference


def run(tmp_path: Path, *cells: str, extra: list[str] | None = None) -> dict:
    out = tmp_path / "selection.json"
    arguments = []
    for cell in cells:
        arguments += ["--cell", cell]
    main(arguments + ["-o", str(out)] + (extra or []))
    return json.loads(out.read_text())


def test_picks_the_closest_checkpoint_that_never_diverges(tmp_path: Path):
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.05), auc=0.10)  # no drop, 5 cm
    write_checkpoint(cell, 200, *held(0.02), auc=0.20)  # no drop, 2 cm  <- the pick
    # Closest on average, but it leaves the 10 cm band, so it is not holding the object.
    achieved, reference = held(0.01)
    achieved[40:60, 0] += 0.30
    write_checkpoint(cell, 300, achieved, reference, auc=0.90)
    record = run(tmp_path, f"e03={cell}")
    episode = record["episodes"]["e03"]
    assert episode["drop_free_checkpoints"] == [100, 200]
    assert episode["selected_environment_steps"] == 200
    assert "no drop" in episode["selection_reason"]
    # The highest-AUC checkpoint was available and was not chosen: the rule never reads the score.
    assert record["summary"]["mean_evaluator_add_auc_selected"] == 0.2


def test_when_every_checkpoint_drops_the_longest_hold_wins(tmp_path: Path):
    cell = tmp_path / "e09"
    for steps, drop_at, offset in ((100, 10, 0.01), (200, 50, 0.02)):
        achieved, reference = held(offset)
        achieved[drop_at:, 0] += 0.4
        write_checkpoint(cell, steps, achieved, reference)
    record = run(tmp_path, f"e09={cell}")
    episode = record["episodes"]["e09"]
    assert episode["drop_free_checkpoints"] == []
    assert episode["selected_environment_steps"] == 200
    assert "held longest (50 frames)" in episode["selection_reason"]
    assert record["summary"]["episodes_with_no_drop_free_checkpoint"] == 1


def test_ties_break_on_distance_then_on_the_earlier_checkpoint(tmp_path: Path):
    cell = tmp_path / "e18"
    write_checkpoint(cell, 100, *held(0.03))
    write_checkpoint(cell, 200, *held(0.03))  # identical: the earlier one wins
    write_checkpoint(cell, 300, *held(0.04))
    assert run(tmp_path, f"e18={cell}")["episodes"]["e18"]["selected_environment_steps"] == 100


def test_invalid_frames_are_not_scored(tmp_path: Path):
    cell = tmp_path / "e24"
    achieved, reference = held(0.02)
    achieved[80:, 0] += 1.0  # would be a gross divergence...
    valid = np.ones(FRAMES, dtype=bool)
    valid[80:] = False  # ...but the evaluator marked those frames invalid
    write_checkpoint(cell, 100, achieved, reference, valid=valid)
    episode = run(tmp_path, f"e24={cell}")["episodes"]["e24"]
    assert episode["checkpoints"][0]["frames"] == 80
    assert episode["checkpoints"][0]["frames_invalid"] == 20
    assert episode["drop_free_checkpoints"] == [100]


def test_the_divergence_limit_is_the_evaluators_own_threshold(tmp_path: Path):
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.11))  # 11 cm: past the evaluator's 10 cm
    write_checkpoint(cell, 200, *held(0.09))  # 9 cm: inside it
    record = run(tmp_path, f"e03={cell}")
    assert record["rule"]["divergence_limit_m"] == 0.10
    assert record["episodes"]["e03"]["drop_free_checkpoints"] == [200]


def test_a_two_frame_lift_lag_is_not_a_drop_under_the_z_rule():
    """The e24 artefact: a 7 mm, 2-frame lag as the object crosses the line, not a lost object."""
    reference = np.full(FRAMES, 0.16)
    achieved = reference.copy()
    achieved[50:52] = 0.149  # 11 mm under the reference, 1 mm under the line, for 2 frames
    assert first_z_drop(achieved, reference) is None
    # A real fall: 30 cm below the reference, for 20 frames.
    achieved = reference.copy()
    achieved[60:] = 0.02
    assert first_z_drop(achieved, reference) == 60


def test_the_z_drop_criterion_is_selectable_and_recorded(tmp_path: Path):
    cell = tmp_path / "e24"
    # Diverges far in x, but never falls: a drop under `divergence`, not under `z-drop`.
    achieved, reference = held(0.0)
    achieved[50:, 0] += 0.5
    write_checkpoint(cell, 100, achieved, reference)
    assert run(tmp_path, f"e24={cell}")["episodes"]["e24"]["drop_free_checkpoints"] == []
    record = run(tmp_path, f"e24={cell}", extra=["--criterion", "z-drop"])
    assert record["rule"]["criterion"] == "z-drop"
    assert record["episodes"]["e24"]["drop_free_checkpoints"] == [100]


def test_each_episode_emits_its_own_checkpoint_sha_for_the_packer(tmp_path: Path):
    first, second = tmp_path / "e03", tmp_path / "e09"
    write_checkpoint(first, 100, *held(0.05), sha="a" * 64)
    write_checkpoint(first, 200, *held(0.02), sha="b" * 64)
    write_checkpoint(second, 100, *held(0.01), sha="c" * 64)
    write_checkpoint(second, 200, *held(0.04), sha="d" * 64)
    record = run(tmp_path, f"e03={first}", f"9={second}")
    # Provenance is stamped per episode, and with this rule the two episodes differ.
    assert record["per_episode_checkpoint_sha256"] == {"3": "b" * 64, "9": "c" * 64}
    assert record["rule"]["ground_truth_read"] is False


def test_a_policy_that_does_not_match_its_recorded_sha_is_refused(tmp_path: Path):
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.02), policy=b"weights", sha="f" * 64)
    with pytest.raises(SystemExit) as caught:
        run(tmp_path, f"e03={cell}")
    assert "must not claim a provenance it cannot reproduce" in str(caught.value)


def test_a_policy_that_matches_is_hashed_and_recorded(tmp_path: Path):
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.02), policy=b"weights")
    row = read_cell(cell)[0]
    assert row["checkpoint_sha256_of_local_policy"] == hashlib.sha256(b"weights").hexdigest()
    assert row["checkpoint_sha256"] == row["checkpoint_sha256_of_local_policy"]


def test_a_cell_with_nothing_evaluated_is_an_error_not_an_empty_pick(tmp_path: Path):
    cell = tmp_path / "e03"
    (cell / "policy_100").mkdir(parents=True)
    with pytest.raises(SystemExit) as caught:
        run(tmp_path, f"e03={cell}")
    assert "no evaluated checkpoints" in str(caught.value)


def test_a_label_that_is_not_an_episode_is_refused(tmp_path: Path):
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.02))
    with pytest.raises(SystemExit) as caught:
        run(tmp_path, f"table={cell}")
    assert "is not an episode" in str(caught.value)
