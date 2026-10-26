"""Guards on how a handover is read, because a misread one trains on the wrong references.

The two handovers arrive in different shapes: interp's is a retargeted G1 tree, tetool's is a
drop-in `--dataset-root` whose object poses are cleaned. The stage has to tell them apart by what
is in them, retarget the second kind first, score its rollouts against that same root, and take
the clips a source says not to trust from the source's own JSON rather than from its prose. Each
of those being wrong is quiet: the run trains and scores and reports a number that means something
else.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[1]
STAGER = REPO / "brev/stage_t2_combined.sh"
EPISODES = [(episode, f"2026-01-0{episode % 10}_seq_{episode}") for episode in range(3)]


def briefs(root: Path, **files: str) -> Path:
    directory = root / "briefs"
    directory.mkdir(parents=True, exist_ok=True)
    for name, body in files.items():
        (directory / f"{name}.txt").write_text(body)
    return directory


def episode_map(root: Path) -> Path:
    mapping = root / "map.txt"
    mapping.write_text("".join(f"{episode} {sequence}\n" for episode, sequence in EPISODES))
    return mapping


def retargeted_tree(root: Path, name: str, episodes: list[tuple[int, str]]) -> Path:
    """The shape the runner reads: whole_body/soma/sequence_id=*/robot_name=g1/data.parquet."""
    tree = root / name
    for _episode, sequence in episodes:
        clip = tree / f"whole_body/soma/sequence_id={sequence}"
        (clip / "robot_name=g1").mkdir(parents=True)
        (clip / "robot_name=g1/data.parquet").write_bytes(b"not a real parquet")
        (clip / "object").mkdir()
    return tree


def dataset_root(root: Path, name: str, *, fall_back_to_raw: list[int] | None = None) -> Path:
    """The shape tetool handed over: a released dataset tree with cleaned object poses."""
    tree = root / name
    (tree / "track_2/tier_2_synthetic_noise").mkdir(parents=True)
    if fall_back_to_raw is not None:
        (tree / "ceiling_per_clip.json").write_text(
            json.dumps({"fall_back_to_raw": fall_back_to_raw})
        )
    return tree


def stage(root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "bash",
            str(STAGER),
            "--from-briefs",
            "--briefs",
            str(root / "briefs"),
            "--map",
            str(episode_map(root)),
            "--mount",
            "t2_test",
            "--out",
            str(root / "out"),
            *extra,
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": str(root), "V2D_SCRATCH": str(root)},
    )


def test_an_absent_handover_is_a_skipped_source_not_an_error(tmp_path: Path):
    briefs(tmp_path)
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    done = stage(tmp_path)
    assert "no finger_refs_path.txt -- skipping the fing source" in done.stdout
    assert "no clean_refs_path.txt -- skipping the clean source" in done.stdout


def test_a_dataset_root_without_its_retarget_names_the_command_to_run(tmp_path: Path):
    tree = dataset_root(tmp_path, "bridge_v1")
    briefs(tmp_path, clean_refs_path=f"{tree}\n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    done = stage(tmp_path)
    assert done.returncode == 2
    assert "names a dataset root, whose retarget is not on disk yet" in done.stderr
    assert "v2d retarget --dataset-root" in done.stderr
    assert not (tmp_path / "out").exists()


def test_a_retarget_still_running_is_refused_rather_than_filled_with_raw(tmp_path: Path):
    tree = dataset_root(tmp_path, "bridge_v1")
    briefs(tmp_path, clean_refs_path=f"{tree}\n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    retargeted_tree(tmp_path, "t2_clean/tier2", EPISODES[:1])  # 1 of 3 done
    done = stage(tmp_path)
    assert done.returncode == 2
    assert "has 1 of 3 clips -- it is still running" in done.stderr
    assert "silently fall back to raw for the other 2" in done.stderr


def test_the_mix_can_be_accepted_deliberately(tmp_path: Path):
    tree = dataset_root(tmp_path, "bridge_v1")
    briefs(tmp_path, clean_refs_path=f"{tree}\n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    retargeted_tree(tmp_path, "t2_clean/tier2", EPISODES[:1])
    done = stage(tmp_path, "--allow-partial")
    # It gets as far as staging, which then fails on these fake parquets -- but not on the count.
    assert "still running" not in done.stderr
    assert "scored against" in done.stdout


def test_flagged_clips_come_from_the_sources_own_json_not_its_prose(tmp_path: Path):
    tree = dataset_root(tmp_path, "bridge_v1", fall_back_to_raw=[1])
    briefs(tmp_path, clean_refs_path=f"{tree}\nFALL BACK TO RAW ON EPISODE 2 (prose, ignored)\n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    retargeted_tree(tmp_path, "t2_clean/tier2", EPISODES)
    done = stage(tmp_path)
    assert "falls back to raw on 1" in done.stdout
    assert "falls back to raw on 2" not in done.stdout


def test_a_handover_of_neither_shape_stops_the_stage(tmp_path: Path):
    tree = tmp_path / "something_else"
    (tree / "results").mkdir(parents=True)
    briefs(tmp_path, clean_refs_path=f"{tree}\n")
    done = stage(tmp_path)
    assert done.returncode == 2
    assert "neither a retargeted tree" in done.stderr


def test_a_handover_file_that_names_no_path_stops_the_stage(tmp_path: Path):
    briefs(tmp_path, clean_refs_path="\n   \n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    done = stage(tmp_path)
    # A file that exists but says nothing means the handover went wrong. Treating it as "no source"
    # would quietly produce a raw-only run that looks like a successful stage.
    assert done.returncode == 2
    assert "names no path" in done.stderr


def test_a_retargeted_tree_is_used_as_handed_over(tmp_path: Path):
    tree = retargeted_tree(tmp_path, "interp_fingers", EPISODES)
    briefs(tmp_path, finger_refs_path=f"{tree}\n")
    retargeted_tree(tmp_path, "t2_raw/tier2", EPISODES)
    done = stage(tmp_path)
    assert f"fing -> {tree}" in done.stdout
    # No retarget is implied for this shape, so nothing is said about scoring against a root.
    assert "scored against" not in done.stdout
