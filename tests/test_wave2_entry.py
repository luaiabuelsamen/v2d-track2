"""Staging the final-candidate entry: the mistakes here would ship a wrong or short CSV.

Three are pinned because none of them shows up in the packer's own output: packing a clip's
rollout against a reference its policy never tracked (every arm has its own staged tree), packing
fewer than 30 episodes because an arm had not landed, and re-deciding which rollout ships instead
of reading the registered adoption record.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.dataset import episode_parquet
from v2d.submit.wave2 import FIRST_WAVE, main, reference_root, ships, stage


def write_rollout(checkpoint: Path, frames: int = 4) -> None:
    checkpoint.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "achieved_position_x": [0.0] * frames,
                "reference_position_x": [0.0] * frames,
                "environment_valid": [True] * frames,
            }
        ),
        checkpoint / "objects.parquet",
    )
    (checkpoint / "metrics.json").write_text(json.dumps({"environment_steps": 349999104}))


def write_dataset(root: Path, episode: int, *, x: float, frames: int = 4) -> None:
    """A minimal released-layout tier 2 parquet: object pose is all this reads."""
    path = episode_parquet(root, "tier2", episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    pose = [[x, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0] for _ in range(frames)]
    pq.write_table(
        pa.table(
            {
                "observation.object.pose": pose,
                "observation.object.visible": [True] * frames,
            }
        ),
        path,
    )


def adoption_record(tmp_path: Path, *, arm: str, episodes: int = 1, awaiting: dict | None = None):
    clips = {}
    for episode in range(episodes):
        tag = f"e{episode:02d}"
        shipped = tmp_path / "first_wave" / tag / "policy_299999232"
        write_rollout(shipped)
        clip = {
            "episode": episode,
            "shipped": {
                "checkpoint_dir": str(shipped),
                "selected_environment_steps": 299999232,
                "selected_checkpoint_sha256": "a" * 64,
            },
        }
        if arm != FIRST_WAVE:
            new = tmp_path / f"run_{arm}" / tag / "policy_349999104"
            write_rollout(new)
            clip["ships"] = {
                "arm": arm,
                "checkpoint_dir": str(new),
                "selected_environment_steps": 349999104,
                "selected_checkpoint_sha256": "b" * 64,
            }
        clips[tag] = clip
    path = tmp_path / "adoption.json"
    path.write_text(json.dumps({"clips": clips, "clips_awaiting_results": awaiting or {}}))
    return path


def test_ships_falls_back_to_the_shipped_rollout_when_no_arm_replaced_the_clip():
    clip = {
        "shipped": {
            "checkpoint_dir": "/cell/policy_1",
            "selected_environment_steps": 1,
            "selected_checkpoint_sha256": "c" * 64,
        }
    }
    assert ships(clip)["arm"] == FIRST_WAVE
    assert ships(clip)["checkpoint_dir"] == "/cell/policy_1"


def test_ships_reads_the_record_and_never_recomputes_it():
    clip = {
        "shipped": {"checkpoint_dir": "/old"},
        "ships": {"arm": "B4x", "checkpoint_dir": "/new"},
    }
    assert ships(clip) == {"arm": "B4x", "checkpoint_dir": "/new"}


def test_reference_root_is_the_shipping_arms_own_tree():
    roots = {"A": {"e03": "/clean/bridge_v1"}, FIRST_WAVE: {"e03": "/other"}}
    root, is_released = reference_root(roots, "A", "e03", Path("/released"))
    assert (root, is_released) == (Path("/clean/bridge_v1"), False)


def test_reference_root_falls_back_to_the_release_for_a_clip_the_arm_does_not_list():
    root, is_released = reference_root({"A": {}}, "A", "e01", Path("/released"))
    assert (root, is_released) == (Path("/released"), True)


def test_a_clip_still_awaiting_its_arm_is_a_refusal_not_a_short_csv(tmp_path):
    adoption = adoption_record(tmp_path, arm="B4x", awaiting={"B4x": ["e26"]})
    with pytest.raises(SystemExit, match="awaiting B4x e26"):
        stage(
            adoption,
            tmp_path / "out",
            stage_root=tmp_path,
            released=tmp_path / "released",
        )


def test_fewer_than_thirty_decided_clips_is_a_refusal(tmp_path):
    adoption = adoption_record(tmp_path, arm="B4x")
    with pytest.raises(SystemExit, match="a Tier 2 entry needs 30"):
        stage(adoption, tmp_path / "out", stage_root=tmp_path, released=tmp_path / "released")


def test_allow_partial_stages_and_stamps_the_record(tmp_path):
    released = tmp_path / "released"
    write_dataset(released, 0, x=0.5)
    adoption = adoption_record(tmp_path, arm="B4x", awaiting={"B4x": ["e26"]})
    staged = stage(
        adoption,
        tmp_path / "out",
        stage_root=tmp_path,
        released=released,
        allow_partial=True,
    )
    assert staged["partial"] is True
    assert staged["awaiting"] == {"B4x": ["e26"]}


def test_a_missing_rollout_names_the_clip_and_the_arm(tmp_path):
    adoption = adoption_record(tmp_path, arm="B4x")
    record = json.loads(adoption.read_text())
    Path(record["clips"]["e00"]["ships"]["checkpoint_dir"], "objects.parquet").unlink()
    with pytest.raises(SystemExit, match="e00 ships B4x"):
        stage(
            adoption,
            tmp_path / "out",
            stage_root=tmp_path,
            released=tmp_path / "released",
            allow_partial=True,
        )


def test_the_staged_frame_reference_comes_from_the_shipping_arms_provenance(tmp_path):
    """The whole point: B4x's clip is converted against bridge_v1, not the released input."""
    released, clean = tmp_path / "released", tmp_path / "clean"
    write_dataset(released, 0, x=0.5)
    write_dataset(clean, 0, x=0.25)
    (tmp_path / "sw_B4x").mkdir()
    (tmp_path / "sw_B4x" / "provenance.json").write_text(
        json.dumps({"ref_dataset_by_clip": {"e00": str(clean)}})
    )
    adoption = adoption_record(tmp_path, arm="B4x")

    staged = stage(
        adoption,
        tmp_path / "out",
        stage_root=tmp_path,
        released=released,
        allow_partial=True,
    )

    entry = staged["episodes"][0]
    assert entry["frame_reference_root"] == str(clean)
    assert entry["frame_reference_is_released_input"] is False
    pose = np.load(tmp_path / "out" / "frame_refs" / "episode_000000.npz")["pose_xyzw"]
    assert pose[0, 0, 0] == pytest.approx(0.25)


def test_a_first_wave_clip_keeps_the_reference_the_first_wave_recorded(tmp_path):
    released, clean = tmp_path / "released", tmp_path / "clean"
    write_dataset(released, 0, x=0.5)
    write_dataset(clean, 0, x=0.25)
    first_wave = tmp_path / "first_wave_provenance.json"
    first_wave.write_text(json.dumps({"ref_dataset_by_clip": {"e00": str(clean)}}))
    adoption = adoption_record(tmp_path, arm=FIRST_WAVE)

    staged = stage(
        adoption,
        tmp_path / "out",
        stage_root=tmp_path,
        released=released,
        first_wave_provenance=first_wave,
        allow_partial=True,
    )

    assert staged["episodes"][0]["ships_from"] == FIRST_WAVE
    assert staged["episodes"][0]["frame_reference_root"] == str(clean)


def test_the_rollout_tree_is_named_the_way_the_kit_packer_reads_it(tmp_path):
    released = tmp_path / "released"
    write_dataset(released, 0, x=0.5)
    adoption = adoption_record(tmp_path, arm="B4x")
    stage(
        adoption,
        tmp_path / "out",
        stage_root=tmp_path,
        released=released,
        allow_partial=True,
    )
    rollouts = tmp_path / "out" / "rollouts"
    assert (rollouts / "episode_000000.parquet").exists()
    assert (rollouts / "episode_000000.json").exists()


def test_staging_twice_does_not_leave_a_stale_episode_behind(tmp_path):
    released = tmp_path / "released"
    write_dataset(released, 0, x=0.5)
    write_dataset(released, 1, x=0.5)
    adoption = adoption_record(tmp_path, arm="B4x", episodes=2)
    kwargs = {"stage_root": tmp_path, "released": released, "allow_partial": True}
    stage(adoption, tmp_path / "out", **kwargs)

    record = json.loads(adoption.read_text())
    del record["clips"]["e01"]
    adoption.write_text(json.dumps(record))
    stage(adoption, tmp_path / "out", **kwargs)

    assert not (tmp_path / "out" / "rollouts" / "episode_000001.parquet").exists()


def test_the_cli_reports_the_arm_split_and_the_digest_count(tmp_path, capsys):
    released = tmp_path / "released"
    write_dataset(released, 0, x=0.5)
    adoption = adoption_record(tmp_path, arm="B4x")
    main(
        [
            "--adoption",
            str(adoption),
            "--out",
            str(tmp_path / "out"),
            "--stage-root",
            str(tmp_path),
            "--released",
            str(released),
            "--allow-partial",
        ]
    )
    out = capsys.readouterr().out
    assert "B4x 1" in out
    assert "1 distinct checkpoint digests" in out
    assert "PARTIAL" in out
