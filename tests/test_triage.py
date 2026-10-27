"""The buckets have to be wrong-proof, because each one sends money somewhere different.

The two that would cost most if they were silently wrong: a clip bucketed `held_drifting` when it
actually loses the object (no fix gets proposed), and the before/after split on the error means --
averaging over frames after the object is gone makes every clip look equally broken.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.analyze.triage import (
    ceiling_auc,
    classify,
    geodesic_deg,
    hand_count,
    main,
    read_rollout,
    sustained_to_end,
    triage,
)

IDENTITY = (0.0, 0.0, 0.0, 1.0)


def rollout(achieved_z: list[float], reference_z: list[float], offset_x: list[float] | None = None):
    frames = len(achieved_z)
    across = offset_x if offset_x is not None else [0.0] * frames
    return {
        "achieved": np.stack(
            [np.array(across), np.zeros(frames), np.array(achieved_z, dtype=float)], axis=1
        ),
        "reference": np.stack(
            [np.zeros(frames), np.zeros(frames), np.array(reference_z, dtype=float)], axis=1
        ),
        "distance": np.abs(np.array(across, dtype=float)),
        "angle_deg": np.zeros(frames),
    }


def bucket_of(data, drop_free=3, **kw):
    options = {"drop_z": 0.15, "margin": 0.05, "min_frames": 3, "lost_limit": 0.10, **kw}
    return classify(data, drop_free=drop_free, **options)[0]


def test_a_clip_that_never_leaves_the_gate_is_held_drifting():
    assert bucket_of(rollout([0.9] * 20, [0.9] * 20, [0.02] * 20)) == "held_drifting"


def test_an_excursion_that_lasts_to_the_end_is_lost_mid_clip():
    data = rollout([0.9] * 20, [0.9] * 20, [0.02] * 10 + [0.5] * 10)
    assert bucket_of(data) == "lost_mid_clip"
    assert classify(data, drop_free=3, drop_z=0.15, margin=0.05, min_frames=3,
                    lost_limit=0.10)[3] == 10


def test_an_excursion_that_recovers_is_not_lost_however_big_it_was():
    """The hole a to-the-end rule leaves, and the bucket that catches it: e01's shape."""
    data = rollout([0.9] * 20, [0.9] * 20, [0.5] * 9 + [0.02] * 11)
    assert bucket_of(data) == "never_locks_on"
    # Same excursion, but short enough to be a blip rather than a pattern.
    brief = rollout([0.9] * 20, [0.9] * 20, [0.5] * 2 + [0.02] * 18)
    assert bucket_of(brief) == "held_drifting"


def test_a_real_loss_of_the_object_outranks_an_excursion():
    data = rollout([0.9] * 5 + [0.02] * 15, [0.9] * 20, [0.02] * 10 + [0.5] * 10)
    assert bucket_of(data) == "dropped"


def test_no_drop_free_checkpoint_outranks_everything_the_rollout_shows():
    """`fell_back` is a fact about training, not about this one rollout."""
    assert bucket_of(rollout([0.9] * 20, [0.9] * 20, [0.02] * 20), drop_free=0) == "fell_back"


def test_the_frame_it_goes_wrong_is_the_drop_when_there_is_one():
    data = rollout([0.9] * 8 + [0.02] * 12, [0.9] * 20, [0.5] * 20)
    _, goes_wrong, drop, lost = classify(
        data, drop_free=3, drop_z=0.15, margin=0.05, min_frames=3, lost_limit=0.10
    )
    assert drop == 8 and lost == 0 and goes_wrong == 8


def test_sustained_only_counts_a_run_that_reaches_the_end():
    assert sustained_to_end(np.array([0, 1, 1, 1, 0, 0], bool), 3) is None
    assert sustained_to_end(np.array([0, 0, 1, 1, 1], bool), 3) == 2
    assert sustained_to_end(np.array([1, 1], bool), 3) is None


def test_the_caption_decides_the_hand_count():
    assert hand_count("using both hands, tip the stool") == "two-hand"
    assert hand_count("Use two hands to lift") == "two-hand"
    assert hand_count("push the luggage with your foot") == "foot"
    assert hand_count("pick up the mug") == "one-hand"
    # A foot clip that also says "both hands" is still the foot clip.
    assert hand_count("with both hands behind you, push with your foot") == "foot"


def test_the_ceiling_follows_the_reference_the_clip_actually_trained_on():
    assert ceiling_auc(
        {"add_auc_before": 0.38, "add_auc_after": 0.35, "fall_back_to_raw": True}
    ) == 0.38
    assert ceiling_auc(
        {"add_auc_before": 0.21, "add_auc_after": 0.39, "fall_back_to_raw": False}
    ) == 0.39


def test_the_angle_is_a_real_angle_and_ignores_the_double_cover():
    quarter = np.array([[0.0, 0.0, np.sin(np.pi / 8), np.cos(np.pi / 8)]])
    assert geodesic_deg(np.array([IDENTITY]), quarter) == pytest.approx(45.0)
    assert geodesic_deg(np.array([IDENTITY]), -quarter) == pytest.approx(45.0)


def write_cell(results: Path, tag: str, steps: int, *, drop_free: list[int], auc: float,
               achieved_x: list[float]) -> None:
    cell = results / tag / f"policy_{steps}"
    cell.mkdir(parents=True)
    frames = len(achieved_x)
    pq.write_table(
        pa.table(
            {
                "environment_valid": np.ones(frames, dtype=bool),
                **{f"achieved_position_{a}": np.asarray(v, dtype=float)
                   for a, v in zip("xyz", [achieved_x, [0.0] * frames, [0.9] * frames], strict=True)},
                **{f"reference_position_{a}": np.zeros(frames) if a != "z"
                   else np.full(frames, 0.9) for a in "xyz"},
                **{f"achieved_quaternion_{a}": np.full(frames, v)
                   for a, v in zip("xyzw", IDENTITY, strict=True)},
                **{f"reference_quaternion_{a}": np.full(frames, v)
                   for a, v in zip("xyzw", IDENTITY, strict=True)},
            }
        ),
        cell / "objects.parquet",
    )
    (cell / "score.json").write_text(
        json.dumps({"overall": {"add_auc": auc, "mppe_cm": 3.0, "spider_sr": 1.0,
                                "maniptrans_sr": 0.0}})
    )
    (results / tag / "selection.json").write_text(
        json.dumps(
            {
                "episodes": {
                    tag: {
                        "episode": int(tag[1:]),
                        "checkpoints": [{"environment_steps": steps, "dropped": not drop_free}],
                        "drop_free_checkpoints": drop_free,
                        "selected_environment_steps": steps,
                    }
                }
            }
        )
    )


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    meta = tmp_path / "data/track_2/tier_1_multiview_caption/meta"
    meta.mkdir(parents=True)
    (meta / "episodes_metadata.jsonl").write_text(
        json.dumps(
            {
                "episode_index": 5,
                "object": "big_black_box",
                "task": "Using both hands, lift the box",
                "support_surface": "support_surface/missing.usda",
            }
        )
        + "\n"
    )
    return tmp_path / "data"


def test_end_to_end_on_one_clip(tmp_path: Path, dataset: Path):
    results = tmp_path / "results"
    write_cell(results, "e05", 100, drop_free=[100], auc=0.26,
               achieved_x=[0.02] * 10 + [0.5] * 10)
    record = triage(results, dataset, min_frames=3)
    clip = record["clips"][0]
    assert clip["bucket"] == "lost_mid_clip"
    assert clip["goes_wrong_frame"] == 10
    assert clip["hands"] == "two-hand"
    # No support USDA on disk, so no primitives, so a floor scene -- not a crash.
    assert clip["scene"] == "floor"
    # The error mean covers only the frames before it goes wrong.
    assert clip["mean_distance_cm_while_holding"] == pytest.approx(2.0)
    assert clip["max_distance_cm"] == pytest.approx(50.0)
    assert record["buckets"] == {"lost_mid_clip": ["e05"]}


def test_an_unscored_clip_is_skipped_rather_than_half_reported(tmp_path: Path, dataset: Path):
    results = tmp_path / "results"
    write_cell(results, "e05", 100, drop_free=[100], auc=0.26, achieved_x=[0.02] * 10)
    (results / "e05/policy_100/score.json").unlink()
    assert triage(results, dataset)["clips"] == []


def test_clips_filter_and_out_file(tmp_path: Path, dataset: Path):
    results = tmp_path / "results"
    write_cell(results, "e05", 100, drop_free=[100], auc=0.26, achieved_x=[0.02] * 10)
    out = tmp_path / "triage.json"
    assert main(["--results", str(results), "--dataset-root", str(dataset),
                 "--clips", "e05", "-o", str(out)]) == 0
    assert json.loads(out.read_text())["clips"][0]["tag"] == "e05"
    with pytest.raises(SystemExit, match="no scored clip"):
        main(["--results", str(results), "--dataset-root", str(dataset), "--clips", "e09"])


def test_read_rollout_keeps_only_valid_frames(tmp_path: Path):
    cell = tmp_path / "policy_1"
    cell.mkdir()
    pq.write_table(
        pa.table(
            {
                "environment_valid": np.array([True, False, True]),
                **{f"achieved_position_{a}": np.array([0.0, 99.0, 0.0]) for a in "xyz"},
                **{f"reference_position_{a}": np.zeros(3) for a in "xyz"},
                **{f"achieved_quaternion_{a}": np.full(3, v) for a, v in zip("xyzw", IDENTITY, strict=True)},
                **{f"reference_quaternion_{a}": np.full(3, v) for a, v in zip("xyzw", IDENTITY, strict=True)},
            }
        ),
        cell / "objects.parquet",
    )
    assert len(read_rollout(cell / "objects.parquet")["distance"]) == 2
