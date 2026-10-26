"""The selection-rule comparison must not flatter any rule, least of all by accident.

Two ways this could mislead and both are cheap to rule out: scoring a clip whose checkpoints are
only partly scored (each rule then picks from a different menu), and mixing up which rule is
allowed to read the kit score -- the oracle row reads Tier 1 truth and the other three must not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from v2d.select.rules import READS_GROUND_TRUTH, compare, main


def write_clip(
    results: Path,
    tag: str,
    checkpoints: list[tuple[int, bool, int]],
    scores: dict[int, float],
    selected: int,
) -> None:
    """One clip: (steps, dropped, held_frames) per checkpoint, plus whichever scores exist."""
    cell = results / tag
    cell.mkdir(parents=True)
    (cell / "selection.json").write_text(
        json.dumps(
            {
                "episodes": {
                    tag: {
                        "episode": int(tag[1:]),
                        "checkpoints": [
                            {"environment_steps": steps, "dropped": dropped, "held_frames": held}
                            for steps, dropped, held in checkpoints
                        ],
                        "selected_environment_steps": selected,
                    }
                }
            }
        )
    )
    for steps, auc in scores.items():
        (cell / f"policy_{steps}").mkdir()
        (cell / f"policy_{steps}/score.json").write_text(json.dumps({"overall": {"add_auc": auc}}))


def test_each_rule_picks_what_it_says_it_picks(tmp_path: Path):
    """e05's shape: the registered rule takes an early checkpoint, the end of training is better."""
    write_clip(
        tmp_path,
        "e05",
        [(100, False, 800), (200, False, 800), (300, False, 800)],
        {100: 0.27, 200: 0.17, 300: 0.39},
        selected=100,
    )
    picks = compare(tmp_path)["per_clip"]["e05"]["picks"]
    assert picks == {
        "z_drop_registered": 100,
        "latest_drop_free": 300,
        "final_checkpoint": 300,
        "oracle_reads_truth": 300,
    }


def test_the_last_drop_free_is_not_the_last_checkpoint_when_the_end_drops(tmp_path: Path):
    write_clip(
        tmp_path,
        "e07",
        [(100, False, 800), (200, False, 800), (300, True, 120)],
        {100: 0.20, 200: 0.30, 300: 0.90},
        selected=200,
    )
    picks = compare(tmp_path)["per_clip"]["e07"]["picks"]
    assert picks["latest_drop_free"] == 200
    assert picks["final_checkpoint"] == 300
    # The oracle is allowed to take the dropping checkpoint; it is an upper bound, not a candidate.
    assert picks["oracle_reads_truth"] == 300


def test_a_clip_with_nothing_drop_free_falls_back_to_the_longest_hold(tmp_path: Path):
    """e04's shape. pick-checkpoint's own fallback, so the comparison must use the same one."""
    write_clip(
        tmp_path,
        "e04",
        [(100, True, 130), (200, True, 386), (300, True, 310)],
        {100: 0.12, 200: 0.15, 300: 0.14},
        selected=200,
    )
    assert compare(tmp_path)["per_clip"]["e04"]["picks"]["latest_drop_free"] == 200


def test_a_partly_scored_clip_is_skipped_not_scored_on_what_it_has(tmp_path: Path):
    write_clip(
        tmp_path,
        "e11",
        [(100, False, 800), (200, False, 800)],
        {100: 0.33},  # 200 never scored: the race that cost e11 its score on 2026-10-26
        selected=200,
    )
    record = compare(tmp_path)
    assert record["clips"] == []
    assert record["clips_skipped_not_fully_scored"] == ["e11"]
    assert record["mean_kit_add_auc"] == {}


def test_only_the_oracle_is_declared_to_read_truth():
    assert READS_GROUND_TRUTH == {
        "z_drop_registered": False,
        "latest_drop_free": False,
        "final_checkpoint": False,
        "oracle_reads_truth": True,
    }


def test_the_means_are_over_the_same_clips_for_every_rule(tmp_path: Path):
    write_clip(
        tmp_path, "e05", [(100, False, 800), (200, False, 800)], {100: 0.20, 200: 0.40}, selected=100
    )
    write_clip(
        tmp_path, "e06", [(100, False, 800), (200, False, 800)], {100: 0.60, 200: 0.80}, selected=100
    )
    means = compare(tmp_path)["mean_kit_add_auc"]
    assert means["z_drop_registered"] == pytest.approx(0.4)
    assert means["final_checkpoint"] == pytest.approx(0.6)


def test_nothing_scored_at_all_is_fatal_not_an_empty_table(tmp_path: Path):
    write_clip(tmp_path, "e05", [(100, False, 800)], {}, selected=100)
    with pytest.raises(SystemExit) as caught:
        main(["--results", str(tmp_path)])
    assert "no clip with every checkpoint scored" in str(caught.value)
    assert "e05" in str(caught.value)


def test_it_writes_the_record_where_it_is_told(tmp_path: Path):
    write_clip(
        tmp_path, "e05", [(100, False, 800), (200, False, 800)], {100: 0.20, 200: 0.40}, selected=200
    )
    out = tmp_path / "out.json"
    assert main(["--results", str(tmp_path), "-o", str(out)]) == 0
    record = json.loads(out.read_text())
    assert record["clips"] == ["e05"]
    assert record["prior_measurement"]["z_drop_registered"] == 0.36635
