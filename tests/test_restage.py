"""`v2d restage-entry`: rebuilding the entry from what survived the scratchpad loss.

The recovery's whole value is that it does not re-decide anything and does not assume the files it
finds are the right ones, so that is what these pin: the digest check that proves a surviving
checkpoint is the one the entry registered, the redirect of the wiped dataset path to wherever the
released data lives now, and the refusal when an episode cannot be rebuilt.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.submit.restage import (
    WIPED_DATA_MARKER,
    cell_for,
    check,
    reference_root_for,
    stage,
)

WIPED = f"/tmp/claude-1000/-home-jetson3{WIPED_DATA_MARKER}"


def write_released(root: Path, episode: int, frames: int) -> None:
    """A released-format episode parquet with just the columns `object_pose` reads."""
    path = root / "track_2/tier_2_synthetic_noise/data/chunk-000" / f"episode_{episode:06d}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = [[0.1 * i, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0] for i in range(frames)]
    pq.write_table(
        pa.table(
            {
                "observation.object.pose": pose,
                "observation.object.visible": [True] * frames,
            }
        ),
        path,
    )


def write_cell(root: Path, tag: str, steps: int, *, frames: int = 4, policy: bytes = b"weights") -> str:
    cell = root / tag / f"policy_{steps}"
    cell.mkdir(parents=True)
    pq.write_table(pa.table({"achieved_position_x": [0.0] * frames}), cell / "objects.parquet")
    (cell / "metrics.json").write_text(json.dumps({"add_auc": 0.5}))
    (cell / "policy.safetensors").write_bytes(policy)
    return hashlib.sha256(policy).hexdigest()


@pytest.fixture
def world(tmp_path: Path):
    """An entry record of two episodes, their surviving cells, and a released root."""
    checkpoints = tmp_path / "ship_ckpts_wave2"
    released = tmp_path / "data"
    reference = tmp_path / "bridge_v1"
    digests = {
        "e00": write_cell(checkpoints, "e00", 349999104),
        "e01": write_cell(checkpoints, "e01", 199999488, policy=b"other"),
    }
    write_released(reference, 0, 6)
    write_released(released, 1, 7)
    record = {
        "episodes": [
            {
                "episode": 0,
                "tag": "e00",
                "ships_from": "A",
                "selected_environment_steps": 349999104,
                "checkpoint_sha256": digests["e00"],
                "frame_reference_root": str(reference),
                "frames": 6,
            },
            {
                "episode": 1,
                "tag": "e01",
                "ships_from": "first wave",
                "selected_environment_steps": 199999488,
                "checkpoint_sha256": digests["e01"],
                "frame_reference_root": WIPED,
                "frames": 7,
            },
        ]
    }
    return record, checkpoints, released, tmp_path


def test_it_stages_both_trees_from_the_surviving_cells(world) -> None:
    record, checkpoints, released, tmp = world
    staged = stage(record, tmp / "out", checkpoints=checkpoints, released=released)
    assert staged["episodes_staged"] == 2
    assert staged["partial"] is True  # two episodes is not a Tier 2 entry
    assert sorted(p.name for p in (tmp / "out/rollouts").glob("*.parquet")) == [
        "episode_000000.parquet",
        "episode_000001.parquet",
    ]
    assert (tmp / "out/frame_refs/episode_000000.npz").is_file()
    assert staged["checkpoint_digests_verified"] == 2
    assert staged["frame_counts_matching_the_record"] == 2


def test_the_frame_reference_carries_the_reference_pose(world) -> None:
    record, checkpoints, released, tmp = world
    stage(record, tmp / "out", checkpoints=checkpoints, released=released)
    with np.load(tmp / "out/frame_refs/episode_000000.npz") as data:
        assert data["pose_xyzw"].shape == (6, 1, 7)
        assert int(data["episode_index"]) == 0
        assert str(data["coordinate_frame"]) == "released_track2_world"


def test_the_wiped_dataset_path_is_redirected_to_the_released_root(world) -> None:
    """The record was written before the reboot, so e01's root points into the wiped scratchpad."""
    record, _checkpoints, released, _tmp = world
    root, is_released = reference_root_for(record["episodes"][1], released)
    assert root == released
    assert is_released is True


def test_a_reference_root_on_the_home_disk_is_used_as_recorded(world) -> None:
    record, _checkpoints, released, tmp = world
    root, is_released = reference_root_for(record["episodes"][0], released)
    assert root == tmp / "bridge_v1"
    assert is_released is False


def test_a_checkpoint_that_is_not_the_registered_one_is_refused(world) -> None:
    """The digest is what distinguishes the entry's checkpoint from any other file of that name."""
    record, checkpoints, released, tmp = world
    (checkpoints / "e00/policy_349999104/policy.safetensors").write_bytes(b"a different policy")
    ready, problems = check(record, checkpoints, released, "tier2")
    assert [e["tag"] for e in ready] == ["e01"]
    assert "not the checkpoint that trained the entry" in problems[0]
    with pytest.raises(SystemExit, match="refusing to restage"):
        stage(record, tmp / "out", checkpoints=checkpoints, released=released)


def test_a_missing_episode_is_refused_by_default_and_named(world) -> None:
    record, checkpoints, released, tmp = world
    import shutil

    shutil.rmtree(checkpoints / "e00")
    with pytest.raises(SystemExit) as refusal:
        stage(record, tmp / "out", checkpoints=checkpoints, released=released)
    assert "e00: no surviving checkpoint" in str(refusal.value)
    assert not (tmp / "out").exists()


def test_allow_missing_stages_the_rest_and_records_the_gap(world) -> None:
    record, checkpoints, released, tmp = world
    import shutil

    shutil.rmtree(checkpoints / "e00")
    staged = stage(
        record, tmp / "out", checkpoints=checkpoints, released=released, allow_missing=True
    )
    assert staged["episodes_staged"] == 1
    assert staged["partial"] is True
    assert any("e00" in problem for problem in staged["missing"])


def test_a_cell_missing_its_rollout_is_refused(world) -> None:
    record, checkpoints, released, _tmp = world
    (checkpoints / "e00/policy_349999104/objects.parquet").unlink()
    _ready, problems = check(record, checkpoints, released, "tier2")
    assert "missing objects.parquet" in problems[0]


def test_cell_for_follows_the_backup_layout(world) -> None:
    record, checkpoints, _released, _tmp = world
    assert cell_for(checkpoints, record["episodes"][0]).name == "policy_349999104"
    assert cell_for(checkpoints, record["episodes"][0]).parent.name == "e00"


def test_nothing_is_re_decided(world) -> None:
    """Every shipped field comes from the record, not from a fresh selection."""
    record, checkpoints, released, tmp = world
    staged = stage(record, tmp / "out", checkpoints=checkpoints, released=released)
    for before, after in zip(record["episodes"], staged["episodes"], strict=True):
        assert after["ships_from"] == before["ships_from"]
        assert after["checkpoint_sha256"] == before["checkpoint_sha256"]
        assert after["selected_environment_steps"] == before["selected_environment_steps"]
    assert "re-decided" in staged["selection_rule"]
