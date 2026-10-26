"""Known answers for the per-clip reference match, because the convert root is silent when wrong.

A rollout converted against the wrong reference still produces a number. On B3 the wrong root moved
MPPE by 0.36 cm and only `rollout-to-npz`'s own frame check noticed. interp's finger tree carries
two references in one tree -- cleaned object poses on the clips tetool cleaned, released ones on the
clip tetool flagged -- so this is what decides which root each of the 30 clips is scored in.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from v2d.stage.object_reference import OBJECT_COLUMNS, main, trajectory_delta

FRAMES = 8


def trajectory(offset: float) -> dict[str, list]:
    """One clip's object columns, displaced by `offset` metres so trees can be told apart."""
    frames = np.arange(FRAMES, dtype=np.float64)
    position = np.stack([frames * 0.01 + offset, frames * 0.0, frames * 0.0], axis=1)
    return {
        "object_root_position": position.tolist(),
        "object_root_axis_angle": np.zeros((FRAMES, 3)).tolist(),
        "object_body_position": position[:, None, :].tolist(),
        "object_body_wxyz": np.tile([1.0, 0.0, 0.0, 0.0], (FRAMES, 1))[:, None, :].tolist(),
    }


def write_clip(tree: Path, sequence: str, offset: float, *, extra_column: bool = False) -> Path:
    clip = tree / f"whole_body/soma/sequence_id={sequence}/robot_name=g1"
    clip.mkdir(parents=True, exist_ok=True)
    row = trajectory(offset)
    if extra_column:
        # A finger solve changes columns outside the object trajectory; those must not count.
        row["robot_joint_positions"] = np.full((FRAMES, 2), offset).tolist()
    table = pa.table({name: [value] for name, value in row.items()})
    pq.write_table(table, clip / "data.parquet")
    return clip / "data.parquet"


@pytest.fixture
def world(tmp_path: Path) -> dict[str, Path]:
    """Three clips; the candidate takes clip 0 and 1 from clean and clip 2 from raw."""
    sequences = {0: "seq_a", 1: "seq_b", 2: "seq_c"}
    (tmp_path / "map.txt").write_text(
        "".join(f"{episode} {sequence}\n" for episode, sequence in sequences.items())
    )
    for episode, sequence in sequences.items():
        write_clip(tmp_path / "clean", sequence, 0.0 if episode < 2 else 7.0)
        write_clip(tmp_path / "raw", sequence, 9.0 if episode < 2 else 5.0)
        write_clip(tmp_path / "fing", sequence, 0.0 if episode < 2 else 5.0, extra_column=True)
    return {"root": tmp_path, "map": tmp_path / "map.txt"}


def run(world: dict[str, Path], tree: str, *extra: str) -> dict:
    out = world["root"] / f"{tree}_matched.json"
    assert (
        main(
            [
                str(world["root"] / tree),
                "--map",
                str(world["map"]),
                "--against",
                f"clean={world['root'] / 'clean'}",
                "--against",
                f"raw={world['root'] / 'raw'}",
                "--ref",
                f"clean={world['root'] / 'bridge_v1'}",
                "-o",
                str(out),
                *extra,
            ]
        )
        == 0
    )
    return json.loads(out.read_text())


def test_each_clip_inherits_the_reference_of_the_trajectory_it_carries(world):
    report = run(world, "fing")
    assert report["by_label"] == {"clean": 2, "raw": 1}
    assert report["per_episode"]["0"]["ref_dataset"] == str(world["root"] / "bridge_v1")
    assert report["per_episode"]["1"]["ref_dataset"] == str(world["root"] / "bridge_v1")
    # raw has no --ref: it is the released root, which is the scorer's default.
    assert "ref_dataset" not in report["per_episode"]["2"]
    assert report["unmatched"] == []


def test_columns_outside_the_object_trajectory_do_not_break_a_match(world):
    """The whole point of the finger tree: 14 joint columns differ, the object does not."""
    report = run(world, "fing")
    assert report["per_episode"]["0"]["matches"] == "clean"
    assert report["per_episode"]["0"]["max_object_delta_m"]["clean"] == 0.0


def test_a_clip_that_matches_nothing_is_fatal(world):
    """A finger solve against the noisy object would put the fingers where the noise was."""
    write_clip(world["root"] / "fing", "seq_b", 3.0, extra_column=True)
    with pytest.raises(SystemExit) as caught:
        run(world, "fing")
    assert "matches none of the sources: [1]" in str(caught.value)
    assert "could not be converted against the reference the policy tracked" in str(caught.value)


def test_a_tie_resolves_to_the_higher_priority_source(world):
    """A clip the cleaner declined is identical in both trees; the pick must not be arbitrary."""
    write_clip(world["root"] / "raw", "seq_a", 0.0)
    report = run(world, "fing")
    assert report["per_episode"]["0"]["max_object_delta_m"] == {"clean": 0.0, "raw": 0.0}
    assert report["per_episode"]["0"]["matches"] == "clean"


def test_an_exact_match_is_the_default_and_a_tolerance_is_explicit(world):
    """1 mm of drift means the trajectory was re-solved, which is not safe to assume away."""
    write_clip(world["root"] / "fing", "seq_a", 0.001, extra_column=True)
    with pytest.raises(SystemExit):
        run(world, "fing")
    report = run(world, "fing", "--tolerance", "0.002")
    assert report["per_episode"]["0"]["matches"] == "clean"


def test_a_clip_the_candidate_does_not_have_is_left_to_a_later_source(world):
    (world["root"] / "fing/whole_body/soma/sequence_id=seq_c").rename(
        world["root"] / "fing/whole_body/soma/absent"
    )
    report = run(world, "fing")
    assert set(report["per_episode"]) == {"0", "1"}
    assert report["unmatched"] == []


def test_a_shape_mismatch_is_not_a_match(world):
    """Different frame counts are a different motion, not a reference to inherit."""
    a = {name: np.zeros((4, 3)) for name in OBJECT_COLUMNS}
    b = {name: np.zeros((5, 3)) for name in OBJECT_COLUMNS}
    assert trajectory_delta(a, b) is None


def test_a_ref_for_a_source_that_is_not_compared_is_refused():
    with pytest.raises(SystemExit) as caught:
        main(["tree", "--map", "map.txt", "--against", "clean=x", "--ref", "fing=y"])
    assert "--ref names a source that is not --against" in str(caught.value)
