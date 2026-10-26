"""Prove the start-curriculum patch applies to the real upstream curriculum, before a box runs it.

The patch moves two reset probabilities between curriculum stages, and the stage a probability
lands in is what decides the step count it takes effect at. A silent off-by-one-stage here is not
a crash; it is a 4.8 h run that measured something nobody designed. So the tests check the stage
structure, not just the text, and check that stages 1-6 come out byte-identical.

`brev/arm_flags.sh` is tested here too, because the patch is only half the change: the other half
is the reset.reset_voc_scale override that the flag carries.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

UPSTREAM = Path(__file__).parent / "data/upstream"
REPO = Path(__file__).parents[1]
CURRICULUM = "configs/curriculum/recon_body_flash_sac.yaml"


def _load(name: str):
    """brev/ holds the box-side scripts and is not an importable package, so load by path."""
    path = REPO / "brev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


start_curriculum = _load("patch_start_curriculum")
object_orientation = _load("patch_object_orientation")
main = start_curriculum.main


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A flash_chord source tree with the real curriculum in it."""
    root = tmp_path / "flash_chord"
    (root / "configs/curriculum").mkdir(parents=True)
    shutil.copy(UPSTREAM / "configs_curriculum_recon_body_flash_sac.yaml", root / CURRICULUM)
    return root


def stages(tree: Path) -> list[dict]:
    return yaml.safe_load((tree / CURRICULUM).read_text())["stages"]


def thresholds(tree: Path) -> list[int]:
    return yaml.safe_load((tree / CURRICULUM).read_text())["thresholds"]


def test_upstream_is_the_curriculum_the_patch_was_derived_against(tree: Path):
    # If this fails, every step count quoted in the patch and in docs/AB4_DESIGN.md is wrong.
    assert thresholds(tree)[5] == 209999872  # zero VOC begins, 60% of the budget
    assert thresholds(tree)[6] == 288749568  # deployment-exact starts begin upstream, 82.5%
    assert [s["voc_scale"] for s in stages(tree)] == [
        1.0,
        0.75,
        0.5,
        0.25,
        0.1,
        0.05,
        0.0,
        0.0,
        0.0,
    ]
    upstream = [s.get("immediate_first_frame_probability") for s in stages(tree)]
    assert upstream == [None] * 7 + [0.25, 0.50]


def test_probabilities_move_one_stage_earlier(tree: Path):
    assert main([str(tree)]) == 0
    moved = [s.get("immediate_first_frame_probability") for s in stages(tree)]
    # Stage 7 is the zero-VOC stage entered at 209,999,872; stage 8 at 288,749,568.
    assert moved == [None] * 6 + [0.25, 0.50, 0.50]


def test_stages_1_to_6_and_the_thresholds_are_untouched(tree: Path):
    before = (tree / CURRICULUM).read_text().splitlines()
    main([str(tree)])
    after = (tree / CURRICULUM).read_text().splitlines()
    # Everything up to and including the last early stage is byte-identical, and so is the
    # threshold list: moving a probability must not move a stage boundary.
    early = before.index("    voc_scale: 0.05") + 2
    assert after[:early] == before[:early]
    assert thresholds(tree) == [
        35000320,
        70000640,
        104998912,
        139999232,
        174999552,
        209999872,
        288749568,
        306249728,
        349999104,
    ]
    assert [s["voc_scale"] for s in stages(tree)] == [
        1.0,
        0.75,
        0.5,
        0.25,
        0.1,
        0.05,
        0.0,
        0.0,
        0.0,
    ]


def test_weights_are_still_the_shared_alias_in_every_stage(tree: Path):
    main([str(tree)])
    text = (tree / CURRICULUM).read_text()
    # set_weights() compares key sets, so a stage that lost the alias would raise mid-run.
    assert text.count("objective_weights: *recon_body_weights") == 8
    assert text.count("objective_weights: &recon_body_weights") == 1
    assert all("object_position" in s["objective_weights"] for s in stages(tree))


def test_idempotent(tree: Path):
    main([str(tree)])
    once = (tree / CURRICULUM).read_text()
    assert main([str(tree)]) == 0
    assert (tree / CURRICULUM).read_text() == once


def test_check_writes_nothing(tree: Path):
    before = (tree / CURRICULUM).read_text()
    assert main([str(tree), "--check"]) == 0
    assert (tree / CURRICULUM).read_text() == before


def test_refuses_a_curriculum_with_different_thresholds(tree: Path):
    path = tree / CURRICULUM
    path.write_text(path.read_text().replace("209999872", "200000000"))
    before = path.read_text()
    with pytest.raises(SystemExit) as raised:
        main([str(tree)])
    assert "not the curriculum this patch was derived against" in str(raised.value)
    assert path.read_text() == before


def test_refuses_a_curriculum_with_a_different_stage_count(tree: Path):
    path = tree / CURRICULUM
    extra = (
        "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
        "    voc_scale: 0.0\n"
        "    objective_weights: *recon_body_weights\n"
    )
    path.write_text(path.read_text() + extra)
    before = path.read_text()
    with pytest.raises(SystemExit) as raised:
        main([str(tree)])
    # 10 stages, 4 of them zero-VOC: the anchor would now be ambiguous, and the patch says so
    # instead of moving a probability into whichever pair it matched first.
    assert "expected 9" in str(raised.value) or "expected 3" in str(raised.value)
    assert path.read_text() == before


def test_refuses_a_missing_curriculum(tmp_path: Path):
    with pytest.raises(SystemExit) as raised:
        main([str(tmp_path / "flash_chord")])
    assert "does not exist" in str(raised.value)


def test_composes_with_the_object_orientation_patch_in_either_order(tmp_path: Path):
    results = []
    for order in ("start-first", "orientation-first"):
        root = tmp_path / order
        (root / "objectives").mkdir(parents=True)
        (root / "configs/objective").mkdir(parents=True)
        (root / "configs/curriculum").mkdir(parents=True)
        shutil.copy(UPSTREAM / "objectives_recon_body.py", root / "objectives/recon_body.py")
        shutil.copy(
            UPSTREAM / "configs_objective_recon_body.yaml",
            root / "configs/objective/recon_body.yaml",
        )
        shutil.copy(UPSTREAM / "configs_curriculum_recon_body_flash_sac.yaml", root / CURRICULUM)
        steps = (
            [main, object_orientation.main]
            if order == "start-first"
            else [object_orientation.main, main]
        )
        for step in steps:
            assert step([str(root)]) == 0
        results.append((root / CURRICULUM).read_text())
    # The two patches touch the same file; applying them in either order must give one curriculum.
    assert results[0] == results[1]
    parsed = yaml.safe_load(results[0])
    assert [s.get("immediate_first_frame_probability") for s in parsed["stages"]] == [None] * 6 + [
        0.25,
        0.50,
        0.50,
    ]
    assert parsed["stages"][0]["objective_weights"]["object_orientation"] == 1.0


# --- brev/arm_flags.sh: the other half of the change, and the arm plumbing ------------------


def flags(script: str) -> subprocess.CompletedProcess:
    """Run a snippet with arm_flags.sh sourced, the way a runner sources it."""
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\n. "{REPO}/brev/arm_flags.sh"\n{script}'],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
    )


def test_stock_recipe_is_no_flags_and_no_overrides():
    done = flags('arm_flags_normalise ""; arm_flags_overrides ""; arm_flags_slug ""')
    assert done.returncode == 0
    assert done.stdout == ""


def test_start_curriculum_carries_the_reset_voc_override():
    # The patch moves the deployment-exact starts; without this, 75% of late resets still have the
    # object placed for them by the VOC wrench.
    done = flags('arm_flags_overrides "start_curriculum"')
    assert done.returncode == 0
    assert done.stdout == "reset.reset_voc_scale=0.0"


def test_flag_order_does_not_change_the_arm():
    canonical = flags('arm_flags_normalise "start_curriculum,orientation_reward,finger_targets"')
    shuffled = flags('arm_flags_normalise "finger_targets,orientation_reward,start_curriculum"')
    assert canonical.stdout == shuffled.stdout
    assert canonical.stdout.split() == ["start_curriculum", "orientation_reward", "finger_targets"]
    assert flags('arm_flags_slug "orientation_reward,start_curriculum"').stdout == "-sc-or"
    assert flags('arm_flags_slug "start_curriculum,orientation_reward"').stdout == "-sc-or"


def test_repeated_flag_is_named_once():
    done = flags('arm_flags_normalise "start_curriculum,start_curriculum"')
    assert done.stdout.split() == ["start_curriculum"]


def test_unknown_flag_is_fatal():
    done = flags('arm_flags_normalise "orientation_rewrad"')
    assert done.returncode == 2
    assert "unknown arm flag" in done.stderr
    assert done.stdout == ""


def test_the_launcher_ships_the_patch_behind_each_flag():
    # The launcher derives its shipping list from this, so a flag whose patch is not listed here is
    # a cell that dies in a second on a box that then idles.
    done = flags('arm_flags_files "start_curriculum,orientation_reward"')
    assert done.stdout.split() == ["patch_start_curriculum.py", "patch_object_orientation.py"]
    for name in done.stdout.split():
        assert (REPO / "brev" / name).is_file()


def test_a_flag_without_its_patch_refuses_to_train():
    done = flags('DRY_RUN=1 arm_flags_apply "finger_targets" "cell"')
    assert done.returncode == 1
    assert "patch_finger_targets.py" in done.stderr
    assert done.stdout == ""


def test_apply_reports_every_patch_and_emits_the_overrides():
    done = flags('DRY_RUN=1 arm_flags_apply "start_curriculum,orientation_reward" "cell e03"')
    assert done.returncode == 0
    assert done.stdout == "reset.reset_voc_scale=0.0"
    assert "would apply patch_start_curriculum.py" in done.stderr
    assert "would apply patch_object_orientation.py" in done.stderr


def test_the_runner_and_launcher_agree_on_the_cell_name():
    # Different slugs on the two sides would pull an empty directory for 4.8 h and report nothing.
    runner = (REPO / "brev/run_ab350.sh").read_text()
    launcher = (REPO / "brev/launch_ab350.sh").read_text()
    assert 'CELL="${ARM}_e$(printf \'%02d\' "$EP")${SLUG}"' in runner
    assert 'CELL_DIR="${ARM}_e$(printf \'%02d\' "$EP")${SLUG}"' in launcher
    assert 'SLUG="$(arm_flags_slug "$ARM_FLAGS")"' in runner
    assert 'SLUG="$(arm_flags_slug "$ARM_FLAGS")"' in launcher
    assert "ARM_FLAGS='$ARM_FLAGS'" in launcher
