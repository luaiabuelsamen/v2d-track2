"""The adoption rule decides what goes on the board, so the costly mistakes are pinned here.

Three of them would be invisible in the output: deciding on the kit score instead of the input
reference (choosing on the answer), comparing two rollouts that tracked different references as
though they were commensurable, and letting a tie move a clip off a rollout that already scored.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.select.adopt import (
    PART_B_BARS,
    compare,
    load_trajectory,
    main,
    render,
    selected_checkpoint,
)
from v2d.select.checkpoints import read_checkpoint

STEPS = 349999104


def write_cell(
    directory: Path,
    *,
    achieved_x: list[float],
    reference_x: list[float],
    steps: int = STEPS,
    auc: float | None = None,
    maniptrans: float = 0.0,
    dropped: bool = False,
) -> Path:
    """One clip's cell: a rollout, the evaluator's metrics, a selection and maybe a kit score."""
    checkpoint = directory / f"policy_{steps}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    frames = len(achieved_x)
    columns = {
        "achieved_position_x": achieved_x,
        "achieved_position_y": [0.0] * frames,
        "achieved_position_z": [0.9] * frames,
        "reference_position_x": reference_x,
        "reference_position_y": [0.0] * frames,
        "reference_position_z": [0.9] * frames,
        "environment_valid": [True] * frames,
    }
    pq.write_table(pa.table(columns), checkpoint / "objects.parquet")
    (checkpoint / "metrics.json").write_text(
        json.dumps({"environment_steps": steps, "checkpoint_sha256": "a" * 64, "metrics": {}})
    )
    distance_cm = round(float(np.abs(np.array(achieved_x) - np.array(reference_x)).mean()) * 100, 4)
    (directory / "selection.json").write_text(
        json.dumps(
            {
                "episodes": {
                    directory.name: {
                        "episode": int(directory.name[1:]),
                        "checkpoints": [
                            {
                                "environment_steps": steps,
                                "dropped": dropped,
                                "mean_distance_to_input_reference_cm": distance_cm,
                            }
                        ],
                        "drop_free_checkpoints": [] if dropped else [steps],
                        "selected_environment_steps": steps,
                        "selected_checkpoint_sha256": "a" * 64,
                        "selection_reason": "no drop, then the smallest mean distance",
                    }
                }
            }
        )
    )
    if auc is not None:
        (checkpoint / "score.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "add_auc": auc,
                        "spider_sr": 1.0,
                        "maniptrans_sr": maniptrans,
                        "mppe_cm": 4.0,
                    }
                }
            )
        )
    return directory


def two_waves(tmp_path: Path, *, new_x, old_x, reference_x, new_reference_x=None, **kw):
    old_results, new_results = tmp_path / "old", tmp_path / "new"
    write_cell(old_results / "e10", achieved_x=old_x, reference_x=reference_x, **kw)
    write_cell(
        new_results / "e10",
        achieved_x=new_x,
        reference_x=new_reference_x if new_reference_x is not None else reference_x,
        **kw,
    )
    return new_results, old_results


def test_closer_to_the_input_reference_is_adopted(tmp_path):
    new, old = two_waves(tmp_path, new_x=[0.01] * 5, old_x=[0.05] * 5, reference_x=[0.0] * 5)
    record = compare({"A": new}, old, reference_from_old=set())
    decision = record["clips"]["e10"]["decision"]
    assert decision["verdict"] == "adopt"
    assert decision["new_mean_distance_to_input_cm"] == 1.0
    assert decision["old_mean_distance_to_input_cm"] == 5.0


def test_further_from_the_input_reference_keeps_the_board(tmp_path):
    new, old = two_waves(tmp_path, new_x=[0.05] * 5, old_x=[0.01] * 5, reference_x=[0.0] * 5)
    assert (
        compare({"A": new}, old, reference_from_old=set())["clips"]["e10"]["decision"]["verdict"]
        == "keep"
    )


def test_a_tie_keeps_the_first_wave(tmp_path):
    new, old = two_waves(tmp_path, new_x=[0.03] * 5, old_x=[0.03] * 5, reference_x=[0.0] * 5)
    assert (
        compare({"A": new}, old, reference_from_old=set())["clips"]["e10"]["decision"]["verdict"]
        == "keep"
    )


def test_the_verdict_ignores_the_kit_score(tmp_path):
    """The new rollout scores far better and is still kept, because it sits further from the input.

    This is the whole point of the rule: a kit score is Tier 1 truth, and an entry chosen on it
    would be chosen on the answer.
    """
    old_results, new_results = tmp_path / "old", tmp_path / "new"
    write_cell(old_results / "e10", achieved_x=[0.01] * 5, reference_x=[0.0] * 5, auc=0.10)
    write_cell(new_results / "e10", achieved_x=[0.05] * 5, reference_x=[0.0] * 5, auc=0.90)
    record = compare({"A": new_results}, old_results, reference_from_old=set())
    clip = record["clips"]["e10"]
    assert clip["decision"]["verdict"] == "keep"
    assert clip["decision"]["verdict_reads_ground_truth"] is False
    assert clip["new_kit_score"]["kit_add_auc"] == 0.90


def test_a_changed_reference_is_undecidable_unless_registered(tmp_path):
    new, old = two_waves(
        tmp_path,
        new_x=[0.01] * 5,
        old_x=[0.05] * 5,
        reference_x=[0.0] * 5,
        new_reference_x=[0.40] * 5,
    )
    record = compare({"B5": new}, old, reference_from_old=set())
    decision = record["clips"]["e10"]["decision"]
    assert decision["verdict"] == "undecidable"
    assert decision["references_agree"] is False
    assert decision["reference_max_gap_cm"] == 40.0
    assert record["summary"]["undecidable"] == ["e10"]


def test_a_registered_clip_is_measured_against_the_shipped_reference(tmp_path):
    """B5: the new rollout tracks its own easier target and must not be flattered by it."""
    new, old = two_waves(
        tmp_path,
        new_x=[0.41] * 5,
        old_x=[0.05] * 5,
        reference_x=[0.0] * 5,
        new_reference_x=[0.40] * 5,
    )
    decision = compare({"B5": new}, old, reference_from_old={"e10"})["clips"]["e10"]["decision"]
    # 1 cm from its own reference, 41 cm from the shipped one. The rule reads the shipped one.
    assert decision["new_mean_distance_to_input_cm"] == 41.0
    assert decision["verdict"] == "keep"
    assert decision["reference_used"] == "the shipped rollout's input reference"


def test_references_within_a_millimetre_count_as_the_same_input(tmp_path):
    """e01's two references differ by a rigid 1.09 mm after retargeting; that is the same input."""
    new, old = two_waves(
        tmp_path,
        new_x=[0.01] * 5,
        old_x=[0.05] * 5,
        reference_x=[0.0] * 5,
        new_reference_x=[0.00109] * 5,
    )
    decision = compare({"B5": new}, old, reference_from_old=set())["clips"]["e10"]["decision"]
    assert decision["references_agree"] is True
    assert decision["verdict"] == "adopt"


def test_distance_matches_what_the_selector_recorded(tmp_path):
    """The rule must decide on the same number pick-checkpoint chose on, not a second opinion."""
    cell = write_cell(tmp_path / "e10", achieved_x=[0.02, 0.04, 0.06], reference_x=[0.0] * 3)
    from_selector = read_checkpoint(cell / f"policy_{STEPS}")
    picked = selected_checkpoint(tmp_path, "e10")
    achieved, reference = load_trajectory(cell / f"policy_{STEPS}")
    assert picked["mean_distance_to_own_reference_cm"] == pytest.approx(
        from_selector["mean_distance_to_input_reference_cm"]
    )
    assert np.allclose(achieved[:, 0], [0.02, 0.04, 0.06])
    assert np.allclose(reference[:, 0], 0.0)


def test_invalid_frames_are_dropped_as_the_selector_drops_them(tmp_path):
    checkpoint = tmp_path / "e10" / f"policy_{STEPS}"
    checkpoint.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "achieved_position_x": [0.0, 10.0],
                "achieved_position_y": [0.0, 0.0],
                "achieved_position_z": [0.9, 0.9],
                "reference_position_x": [0.0, 0.0],
                "reference_position_y": [0.0, 0.0],
                "reference_position_z": [0.9, 0.9],
                "environment_valid": [True, False],
            }
        ),
        checkpoint / "objects.parquet",
    )
    achieved, reference = load_trajectory(checkpoint)
    assert len(achieved) == 1 and len(reference) == 1


def test_a_clip_with_no_first_wave_rollout_is_fatal(tmp_path):
    new_results = tmp_path / "new"
    write_cell(new_results / "e10", achieved_x=[0.0] * 3, reference_x=[0.0] * 3)
    (tmp_path / "old").mkdir()
    with pytest.raises(SystemExit, match="no first-wave selection"):
        compare({"A": new_results}, tmp_path / "old", reference_from_old=set())


def test_a_clip_still_training_is_waited_for_not_decided(tmp_path):
    new_results, old_results = tmp_path / "new", tmp_path / "old"
    (new_results / "e10").mkdir(parents=True)
    write_cell(old_results / "e10", achieved_x=[0.0] * 3, reference_x=[0.0] * 3)
    record = compare({"A": new_results}, old_results, reference_from_old=set())
    assert record["clips"] == {}
    assert record["clips_awaiting_results"] == {"A": ["e10"]}


def test_part_a_bar_counts_maniptrans_passes_and_regressions(tmp_path):
    new_results, old_results = tmp_path / "new", tmp_path / "old"
    # e10 passes MP-SR; e11 collapses more than 0.05 below its record.
    write_cell(
        new_results / "e10", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.5, maniptrans=1.0
    )
    write_cell(old_results / "e10", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.4)
    write_cell(new_results / "e11", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.2)
    write_cell(old_results / "e11", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.4)
    record = {
        "clips": {
            "e10": {"selected_kit_score": {"kit_add_auc": 0.4, "kit_maniptrans_sr": 0.0}},
            "e11": {"selected_kit_score": {"kit_add_auc": 0.4, "kit_maniptrans_sr": 0.0}},
        }
    }
    bars = compare({"A": new_results}, old_results, reference_from_old=set(), record=record)[
        "bars"
    ]["part_a"]
    assert bars["maniptrans_passes_now"] == 1
    assert bars["clips_regressed"] == ["e11"]
    # Not every Part A clip has landed, so the bar is not yet met or missed.
    assert bars["met"] is None


def test_part_b_bars_are_the_registered_numbers():
    assert PART_B_BARS["e01"] == 0.200
    assert PART_B_BARS["e28"] == 0.089
    assert len(PART_B_BARS) == 10


def test_a_record_that_disagrees_with_the_registered_bars_is_fatal(tmp_path):
    new_results, old_results = tmp_path / "new", tmp_path / "old"
    write_cell(new_results / "e01", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.3)
    write_cell(old_results / "e01", achieved_x=[0.0] * 3, reference_x=[0.0] * 3, auc=0.3)
    record_path = tmp_path / "record.json"
    record_path.write_text(
        json.dumps({"clips": {"e01": {"selected_kit_score": {"kit_add_auc": 0.777}}}})
    )
    with pytest.raises(SystemExit, match="registered in"):
        main(
            [
                "--group",
                f"B5={new_results}",
                "--old",
                str(old_results),
                "--record",
                str(record_path),
                "-o",
                str(tmp_path / "out.json"),
            ]
        )


def test_main_writes_the_json_and_the_table(tmp_path):
    new_results, old_results = tmp_path / "new", tmp_path / "old"
    write_cell(
        new_results / "e10", achieved_x=[0.01] * 3, reference_x=[0.0] * 3, auc=0.5, maniptrans=1.0
    )
    write_cell(old_results / "e10", achieved_x=[0.05] * 3, reference_x=[0.0] * 3, auc=0.4)
    out, markdown = tmp_path / "out.json", tmp_path / "table.md"
    main(
        [
            "--group",
            f"A={new_results}",
            "--old",
            str(old_results),
            "-o",
            str(out),
            "--markdown",
            str(markdown),
        ]
    )
    record = json.loads(out.read_text())
    assert record["clips"]["e10"]["decision"]["verdict"] == "adopt"
    assert record["rule"]["ground_truth_read"] is False
    text = markdown.read_text()
    assert "| e10 | A |" in text and "**adopt**" in text
    # Rewritten whole each pass, so a second run leaves one table rather than two.
    main(
        [
            "--group",
            f"A={new_results}",
            "--old",
            str(old_results),
            "-o",
            str(out),
            "--markdown",
            str(markdown),
        ]
    )
    assert markdown.read_text().count("| clip | arm |") == 1


def test_the_table_survives_a_clip_that_is_not_scored_yet(tmp_path):
    new_results, old_results = tmp_path / "new", tmp_path / "old"
    write_cell(new_results / "e10", achieved_x=[0.01] * 3, reference_x=[0.0] * 3)
    write_cell(old_results / "e10", achieved_x=[0.05] * 3, reference_x=[0.0] * 3)
    text = render(compare({"A": new_results}, old_results, reference_from_old=set()))
    assert "| e10 | A | — | — | — | — |" in text
