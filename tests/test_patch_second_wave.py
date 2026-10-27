"""The two second-wave patches, proven against the real upstream text before a box runs them.

Both change numbers that decide an arm, so the failure that matters is a patch that reports
success having changed the wrong thing -- or nothing. `tests/data/upstream/` holds verbatim copies.
"""

from __future__ import annotations

import ast
import importlib.util
import itertools
import shutil
from pathlib import Path

import pytest

UPSTREAM = Path(__file__).parent / "data/upstream"
BUDGET = 349_999_104


def load(name: str):
    path = Path(__file__).parents[1] / f"brev/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


position = load("patch_object_position")
voc = load("patch_voc_curriculum")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "flash_chord"
    (root / "objectives").mkdir(parents=True)
    (root / "configs/objective").mkdir(parents=True)
    (root / "configs/curriculum").mkdir(parents=True)
    shutil.copy(UPSTREAM / "objectives_recon_body.py", root / "objectives/recon_body.py")
    shutil.copy(UPSTREAM / "configs_objective_recon_body.yaml",
                root / "configs/objective/recon_body.yaml")
    shutil.copy(UPSTREAM / "configs_curriculum_recon_body_flash_sac.yaml",
                root / "configs/curriculum/recon_body_flash_sac.yaml")
    return root


def objective_py(tree: Path) -> str:
    return (tree / "objectives/recon_body.py").read_text()


def objective_yaml(tree: Path) -> str:
    return (tree / "configs/objective/recon_body.yaml").read_text()


def curriculum(tree: Path) -> str:
    return (tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text()


# ------------------------------------------------------------------ object position
def test_std_lands_in_both_files_and_leaves_valid_python(tree: Path):
    assert position.main([str(tree), "--std", "0.08"]) == 0
    ast.parse(objective_py(tree))
    assert "    object_position_std: float = 0.08\n" in objective_py(tree)
    assert "object_position_std: 0.08\n" in objective_yaml(tree)


def test_the_weight_is_left_alone_unless_it_is_asked_for(tree: Path):
    position.main([str(tree), "--std", "0.08"])
    assert "    object_position_weight: float = 1\n" not in objective_py(tree)
    assert "    object_position_weight: float = 1.0\n" in objective_py(tree)
    # And the curriculum, which is what actually applies at run time, is untouched.
    assert "      object_position: 1.0\n" in curriculum(tree)


def test_a_weight_change_reaches_the_curriculum_because_that_is_what_applies(tree: Path):
    """set_weights applies the curriculum mapping every stage; the config alone would do nothing."""
    assert position.main([str(tree), "--std", "0.08", "--weight", "5.0"]) == 0
    assert "      object_position: 5\n" in curriculum(tree)
    assert "object_position_weight: 5\n" in objective_yaml(tree)
    assert "    object_position_weight: float = 5\n" in objective_py(tree)


def test_only_the_object_position_weight_moves(tree: Path):
    position.main([str(tree), "--std", "0.08", "--weight", "5.0"])
    text = curriculum(tree)
    for untouched in ("body_joint_position: 5.0", "palm_position: 1.0", "progress: 1.0",
                      "force_closure: 5.0", "pelvis_position: 1.0"):
        assert untouched in text


def test_the_same_values_twice_is_a_no_op(tree: Path):
    position.main([str(tree), "--std", "0.08"])
    once = objective_py(tree), objective_yaml(tree)
    assert position.main([str(tree), "--std", "0.08"]) == 0
    assert (objective_py(tree), objective_yaml(tree)) == once


def test_a_tree_patched_at_other_values_is_refused(tree: Path):
    position.main([str(tree), "--std", "0.08"])
    with pytest.raises(SystemExit, match="already carries"):
        position.main([str(tree), "--std", "0.12"])
    assert "object_position_std: 0.08\n" in objective_yaml(tree)
    # A weight change on a std-only tree is a different arm too, and is refused.
    with pytest.raises(SystemExit, match="already carries"):
        position.main([str(tree), "--std", "0.08", "--weight", "5.0"])


@pytest.mark.parametrize("bad", ["0", "-0.1", "nan", "inf"])
def test_a_std_that_cannot_shape_anything_is_refused(tree: Path, bad: str):
    before = objective_yaml(tree)
    with pytest.raises(SystemExit, match="finite and positive"):
        position.main([str(tree), "--std", bad])
    assert objective_yaml(tree) == before


def test_check_mode_writes_nothing(tree: Path):
    before = objective_py(tree), objective_yaml(tree), curriculum(tree)
    assert position.main([str(tree), "--std", "0.08", "--weight", "5.0", "--check"]) == 0
    assert (objective_py(tree), objective_yaml(tree), curriculum(tree)) == before


def test_a_changed_upstream_is_refused_and_nothing_is_written(tree: Path):
    path = tree / "configs/objective/recon_body.yaml"
    path.write_text(path.read_text().replace("object_position_std: 0.2", "object_position_std: 0.3"))
    with pytest.raises(SystemExit, match="anchor occurs 0 times"):
        position.main([str(tree), "--std", "0.08"])
    # The objective .py is planned first but must not have been written.
    assert "    object_position_std: float = 0.2\n" in objective_py(tree)


# ------------------------------------------------------------------ voc curriculum
def test_the_zero_point_lands_where_it_was_asked_for():
    steps = voc.thresholds(0.85)
    assert steps[5] / BUDGET == pytest.approx(0.85, abs=1e-5)
    assert steps[-1] == BUDGET


def test_the_schedule_is_increasing_and_on_the_world_count_grid():
    for zero_at in (0.65, 0.75, 0.85, 0.95):
        steps = voc.thresholds(zero_at)
        assert all(b > a for a, b in itertools.pairwise(steps))
        assert all(step % voc.WORLD_COUNT == 0 for step in steps)


def test_the_fade_shape_is_unchanged_only_its_timescale():
    """Stages 1-6 keep their spacing ratios; that is what 'stretch, do not reshape' means."""
    stock = [f / voc.STOCK_ZERO_AT for f in voc.STOCK[:6]]
    steps = voc.thresholds(0.85)
    moved = [step / steps[5] for step in steps[:6]]
    assert moved == pytest.approx(stock, abs=1e-4)


def test_the_tail_keeps_its_stock_proportions():
    steps = voc.thresholds(0.85)
    room = BUDGET - steps[5]
    assert (steps[6] - steps[5]) / room == pytest.approx(0.225 / 0.40, abs=1e-3)
    assert (steps[7] - steps[5]) / room == pytest.approx(0.275 / 0.40, abs=1e-3)


def test_patching_changes_the_thresholds_and_nothing_else(tree: Path):
    before = curriculum(tree)
    assert voc.main([str(tree), "--zero-at", "0.85"]) == 0
    after = curriculum(tree)
    assert "thresholds: [49584128," in after
    assert "# v2d voc-curriculum zero_at=0.85" in after
    # Every stage body survives verbatim, which is what lets this compose with the start patch.
    assert before.split("stages:")[1] == after.split("stages:")[1]


def test_it_composes_with_the_start_curriculum_patch_in_that_order(tree: Path):
    """Start first, then voc: one edits stage bodies, the other the thresholds line."""
    start = load("patch_start_curriculum")
    assert start.main([str(tree)]) == 0
    assert voc.main([str(tree), "--zero-at", "0.85"]) == 0
    both = curriculum(tree)
    assert "thresholds: [49584128," in both
    # Start leaves three stages naming the probability: 0.25 at the zero-VOC stage, then 0.50.
    assert both.count("immediate_first_frame_probability") == 3
    assert both.count("immediate_first_frame_probability: 0.25") == 1
    assert both.count("immediate_first_frame_probability: 0.50") == 2
    assert "# v2d start-curriculum" in both and "# v2d voc-curriculum" in both


def test_the_other_order_is_refused_loudly_rather_than_half_applied(tree: Path):
    """start-curriculum verifies the stock thresholds, so voc-first stops the run. By design.

    The runner has to apply them start-then-voc; this pins the failure so the ordering cannot be
    quietly reversed without a test going red.
    """
    start = load("patch_start_curriculum")
    voc.main([str(tree), "--zero-at", "0.85"])
    with pytest.raises(SystemExit, match="thresholds line"):
        start.main([str(tree)])
    # And nothing of the start patch landed, so the tree is still a coherent voc-only arm.
    assert "# v2d start-curriculum" not in curriculum(tree)


def test_a_zero_point_outside_the_useful_range_is_refused(tree: Path):
    for bad in ("0.5", "0.6", "1.0", "1.2"):
        with pytest.raises(SystemExit, match="must be between"):
            voc.main([str(tree), "--zero-at", bad])
    assert voc.MARKER not in curriculum(tree)


def test_the_same_zero_point_twice_is_a_no_op(tree: Path):
    voc.main([str(tree), "--zero-at", "0.85"])
    once = curriculum(tree)
    assert voc.main([str(tree), "--zero-at", "0.85"]) == 0
    assert curriculum(tree) == once


def test_a_different_zero_point_on_a_patched_tree_is_refused(tree: Path):
    voc.main([str(tree), "--zero-at", "0.85"])
    with pytest.raises(SystemExit, match="already carries"):
        voc.main([str(tree), "--zero-at", "0.9"])
    assert "zero_at=0.85" in curriculum(tree)


def test_a_curriculum_with_other_thresholds_is_refused(tree: Path):
    path = tree / "configs/curriculum/recon_body_flash_sac.yaml"
    path.write_text(path.read_text().replace("35000320", "35000321"))
    with pytest.raises(SystemExit, match="derived against"):
        voc.main([str(tree), "--zero-at", "0.85"])


def test_voc_check_mode_writes_nothing(tree: Path):
    before = curriculum(tree)
    assert voc.main([str(tree), "--zero-at", "0.85", "--check"]) == 0
    assert curriculum(tree) == before
