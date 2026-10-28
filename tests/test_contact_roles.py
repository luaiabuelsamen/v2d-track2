"""The contact roles arm CR trains on: the two filters, and the file the kernel side reads back.

The filters are the arm's real content -- a role that flickers teaches a reward that flickers, and
a role set on a hand 15 cm from the object teaches the policy to reach with a hand the reference
keeps away -- so each one is pinned on hand-made arrays. The last two tests close the loop: the
npz `v2d contact-roles` writes is fed to the loader `brev/patch_contact_role.py` injects into
flash_chord, because that pair is the one place where a schema mistake would be invisible until a
box had been billed for it.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from v2d.arms.contact_roles import fill_gaps, roles_for_clip, write_roles

REPO = Path(__file__).parents[1]
UPSTREAM = Path(__file__).parent / "data/upstream"


def loader():
    """`_v2d_contact_roles`, lifted out of the patch exactly as it is injected."""
    spec = importlib.util.spec_from_file_location("cr", REPO / "brev/patch_contact_role.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace: dict = {"np": np}
    exec(module.ROLE_HELPER, namespace)  # the point is to run the shipped text itself
    return namespace["_v2d_contact_roles"]


def test_a_short_dropout_mid_grasp_is_filled() -> None:
    flag = np.array([0, 1, 1, 0, 0, 0, 1, 1, 0], dtype=float)
    assert fill_gaps(flag, 10).tolist() == [False, True, True, True, True, True, True, True, False]


def test_a_long_gap_is_two_separate_grasps() -> None:
    flag = np.zeros(40)
    flag[0:5] = 1
    flag[25:30] = 1
    filled = fill_gaps(flag, 10)
    assert filled[5:25].sum() == 0
    assert filled.sum() == 10


def test_reaching_and_releasing_are_not_dropouts() -> None:
    """Leading and trailing zeros stay zero however short they are."""
    flag = np.array([0, 0, 1, 1, 0, 0], dtype=float)
    assert fill_gaps(flag, 10).tolist() == [False, False, True, True, False, False]


def test_a_gap_of_exactly_the_limit_is_filled_and_one_more_is_not() -> None:
    def gap(size: int) -> bool:
        flag = np.zeros(size + 2)
        flag[0] = flag[-1] = 1
        return bool(fill_gaps(flag, 10).all())

    assert gap(10)
    assert not gap(11)


def test_nothing_to_fill_is_left_alone() -> None:
    assert fill_gaps(np.zeros(10), 10).sum() == 0
    assert fill_gaps(np.ones(10), 10).all()


def scene(distance_cm: float, frames: int = 20) -> tuple[dict, np.ndarray, np.ndarray]:
    """One hand flagged throughout, its palm a chosen distance from a unit-vertex object."""
    palms = np.zeros((frames, 2, 3))
    palms[:, 0, 0] = distance_cm / 100.0
    state = {
        "contact_active": np.column_stack([np.ones(frames), np.zeros(frames)]),
        "palms": palms,
    }
    pose = np.zeros((frames, 7))
    pose[:, 6] = 1.0
    vertices = np.zeros((1, 3))  # the object's surface is the origin
    return state, pose, vertices


def test_a_hand_on_the_object_keeps_its_contact_role() -> None:
    state, pose, vertices = scene(distance_cm=2.0)
    roles, record = roles_for_clip(state, pose, vertices, fill_gap=10, max_distance_cm=8.0)
    assert roles[:, 0].all()
    assert record["hands"][0]["contact_role_frames"] == 20
    assert record["hands"][0]["frames_dropped_as_too_far"] == 0


def test_a_hand_flagged_but_far_from_the_object_loses_it() -> None:
    """e04's right hand is flagged at 10-17 cm; rewarding it teaches the wrong reach."""
    state, pose, vertices = scene(distance_cm=15.0)
    roles, record = roles_for_clip(state, pose, vertices, fill_gap=10, max_distance_cm=8.0)
    assert roles.sum() == 0
    assert record["hands"][0]["flagged_frames"] == 20
    assert record["hands"][0]["frames_dropped_as_too_far"] == 20
    assert record["hands"][0]["palm_to_surface_cm_while_flagged"]["median"] == 15.0


def test_a_hand_the_reference_never_flags_is_never_in_the_contact_role() -> None:
    state, pose, vertices = scene(distance_cm=1.0)
    roles, _ = roles_for_clip(state, pose, vertices, fill_gap=10, max_distance_cm=8.0)
    assert roles[:, 1].sum() == 0


def test_the_gate_is_on_the_surface_not_the_centre() -> None:
    """A 30 cm object held at the end is 0 cm from its surface and 30 from its origin."""
    frames = 5
    palms = np.zeros((frames, 2, 3))
    palms[:, 0, 0] = 0.30
    state = {
        "contact_active": np.column_stack([np.ones(frames), np.zeros(frames)]),
        "palms": palms,
    }
    pose = np.zeros((frames, 7))
    pose[:, 6] = 1.0
    vertices = np.array([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])  # a bar from the origin to the palm
    roles, _ = roles_for_clip(state, pose, vertices, fill_gap=10, max_distance_cm=8.0)
    assert roles[:, 0].all()


def test_the_record_counts_what_each_filter_changed(tmp_path: Path) -> None:
    frames = 30
    palms = np.zeros((frames, 2, 3))
    palms[:, 0, 0] = 0.01
    contact = np.zeros((frames, 2))
    contact[0:10, 0] = 1
    contact[15:20, 0] = 1  # a 5-frame dropout at 10..15
    pose = np.zeros((frames, 7))
    pose[:, 6] = 1.0
    _roles, record = roles_for_clip(
        {"contact_active": contact, "palms": palms},
        pose,
        np.zeros((1, 3)),
        fill_gap=10,
        max_distance_cm=8.0,
    )
    assert record["hands"][0]["flagged_frames"] == 15
    assert record["hands"][0]["gap_frames_filled"] == 5
    assert record["hands"][0]["contact_role_frames"] == 20
    assert record["frames_with_any_contact_role"] == 20
    assert record["frames_with_both_hands"] == 0


def test_the_file_we_write_is_the_file_the_kernel_side_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roles = np.zeros((7, 2), dtype=np.float32)
    roles[2:5, 1] = 1.0
    write_roles(tmp_path / "e04.npz", 4, roles, {"tag": "e04"})

    read = loader()
    monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path / "e04.npz"))
    got = read(np.zeros((7, 2), dtype=np.float32), 7, 2)
    assert got.shape == (7, 2)
    assert got.dtype == np.float32
    assert np.array_equal(got, roles)

    # and by directory plus tag, which is how the runner passes it
    monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path))
    monkeypatch.setenv("V2D_CLIP_TAG", "e04")
    assert np.array_equal(read(np.zeros((7, 2), dtype=np.float32), 7, 2), roles)


def test_the_loader_falls_back_to_the_labels_only_when_nothing_is_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read = loader()
    monkeypatch.delenv("V2D_CONTACT_ROLES", raising=False)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert np.array_equal(read(labels, 2, 2), labels)


@pytest.mark.parametrize(
    "setup",
    ["missing file", "directory without a tag", "wrong keys", "wrong shape", "not finite"],
)
def test_an_unreadable_roles_file_is_fatal_not_a_silent_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setup: str
) -> None:
    """Falling back here would put the stub on the record under the arm's name."""
    read = loader()
    monkeypatch.delenv("V2D_CLIP_TAG", raising=False)
    if setup == "missing file":
        monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path / "nope.npz"))
    elif setup == "directory without a tag":
        monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path))
    else:
        payload = {
            "wrong keys": {"something_else": np.zeros((7, 2), dtype=np.float32)},
            "wrong shape": {"role": np.zeros((7, 3), dtype=np.float32)},
            "not finite": {"role": np.full((7, 2), np.nan, dtype=np.float32)},
        }[setup]
        np.savez(tmp_path / "e04.npz", **payload)
        monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path / "e04.npz"))
    with pytest.raises(ValueError):
        read(np.zeros((7, 2), dtype=np.float32), 7, 2)


def test_a_transposed_role_array_is_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`hand_contact_active` ships (hand, frame), so interp's file may well arrive that way."""
    roles = np.zeros((2, 9), dtype=np.float32)
    roles[1, 3:6] = 1.0
    np.savez(tmp_path / "e04.npz", role=roles)
    monkeypatch.setenv("V2D_CONTACT_ROLES", str(tmp_path / "e04.npz"))
    got = loader()(np.zeros((9, 2), dtype=np.float32), 9, 2)
    assert got.shape == (9, 2)
    assert np.array_equal(got, roles.T)


def test_the_written_file_carries_its_provenance(tmp_path: Path) -> None:
    write_roles(tmp_path / "e09.npz", 9, np.zeros((4, 2), dtype=np.float32), {"fill_gap_frames": 10})
    with np.load(tmp_path / "e09.npz", allow_pickle=False) as data:
        assert int(data["episode_index"]) == 9
        assert [str(s) for s in data["sides"]] == ["left", "right"]
        assert json.loads(str(data["provenance"]))["fill_gap_frames"] == 10


def test_every_clip_the_arm_runs_has_a_roles_file() -> None:
    """The invariant, not a clip list: a CR clip without roles would train the stub silently.

    This asserted the original eight and so failed the moment the arm was cut to six, which is the
    wrong shape for a test -- the clip list is Luai's to change, while "every clip in the row has
    roles on disk" is what must hold whatever he picks.
    """
    table = (REPO / "brev/second_wave.sh").read_text()
    row = next(line for line in table.splitlines() if line.startswith("CR|"))
    clips = {f"e{int(ep):02d}" for ep in row.split("|")[1].split()}
    assert clips, "the CR row names no clips"
    for tag in clips:
        assert (REPO / f"results/contact_roles/{tag}.npz").is_file(), f"{tag} has no roles file"
    record = REPO / "results/contact_roles.json"
    if record.is_file():
        assert clips <= set(json.loads(record.read_text())["clips"])
    assert row.split("|")[3] == "V2D_CONTACT_ROLES=/v2d/assets/cr_refs/contact_roles"
    assert re.search(r"\|cr_refs\|", row)
