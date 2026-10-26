"""Prove the object-orientation patch applies to the real upstream text, before a box runs it.

flash_chord is not installed on the Jetson -- it ships inside the training image -- so without
this the patch could only be tested by paying for a box and waiting ~13 min for the image build.
`tests/data/upstream/` holds verbatim copies of the three files it edits.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path

import pytest

UPSTREAM = Path(__file__).parent / "data/upstream"


def _load_patcher():
    """brev/ holds the box-side scripts and is not an importable package, so load by path."""
    path = Path(__file__).parents[1] / "brev/patch_object_orientation.py"
    spec = importlib.util.spec_from_file_location("patch_object_orientation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


patcher = _load_patcher()
main = patcher.main


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A flash_chord source tree laid out the way the patcher expects to find it."""
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


def objective(tree: Path) -> str:
    return (tree / "objectives/recon_body.py").read_text()


def test_patch_applies_and_leaves_valid_python(tree: Path):
    assert main([str(tree)]) == 0
    ast.parse(objective(tree))


def test_term_is_appended_so_existing_indices_are_untouched(tree: Path):
    main([str(tree)])
    text = objective(tree)
    assert "_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(13)" in text
    # The stock terms keep their indices; the new one takes 12.
    assert "terms[term_base + 11] = force_closure[world]" in text
    assert "terms[term_base + 12] = wp.exp(" in text
    names = text.split("RECON_BODY_OBJECTIVE_TERM_NAMES = (")[1].split(")")[0]
    ordered = [n.strip().strip('",') for n in names.strip().splitlines() if n.strip()]
    assert ordered[11] == "force_closure"
    assert ordered[12] == "object_orientation"
    assert len(ordered) == 13


def test_weights_tuple_order_matches_the_term_order(tree: Path):
    main([str(tree)])
    text = objective(tree)
    block = text.split("    def weights(self)")[1].split("    def build(")[0]
    weights = [line.strip().rstrip(",") for line in block.splitlines()
               if line.strip().startswith("self.") and line.strip().endswith(",")]
    assert weights[-1] == "self.object_orientation_weight"
    assert weights[-2] == "self.force_closure_weight"


def test_kernel_receives_the_reference_quaternion_and_the_new_std(tree: Path):
    main([str(tree)])
    text = objective(tree)
    assert "    object_target_quat_w: wp.array(dtype=wp.quat),\n" in text
    assert "    object_orientation_std: float,\n" in text
    # And both are actually passed at launch, in signature order.
    launch = text.split("            compose_recon_body_objective,")[1]
    assert launch.index("self.command.body_target_pos_w") < launch.index(
        "self.command.body_target_quat_w")
    assert launch.index("self.config.palm_orientation_std") < launch.index(
        "self.config.object_orientation_std")


def test_the_term_mirrors_palm_orientation(tree: Path):
    """Same geodesic angle, same exp(-e^2/s^2), its own std -- not a different formula."""
    main([str(tree)])
    text = objective(tree)
    assert "object_orientation_error = quat_geodesic_angle(" in text
    assert "object_target_quat_w[world]" in text
    assert "-(object_orientation_error * object_orientation_error)" in text
    assert "/ (object_orientation_std * object_orientation_std)" in text


def test_curriculum_names_the_new_weight(tree: Path):
    """set_weights() compares key SETS, so a curriculum short by one raises mid-run."""
    main([str(tree)])
    curriculum = (tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text()
    assert "object_orientation: 1.0" in curriculum
    objective_yaml = (tree / "configs/objective/recon_body.yaml").read_text()
    assert "object_orientation_weight: 1.0" in objective_yaml
    assert "object_orientation_std: 0.3" in objective_yaml


def test_the_default_std_is_0_3_and_the_weight_is_the_house_calibration(tree: Path):
    """Weight 1.0 is copied from palm_orientation; the std is tighter, on purpose."""
    assert patcher.DEFAULT_STD == 0.3
    main([str(tree)])
    text = objective(tree)
    assert "    object_orientation_weight: float = 1.0\n" in text
    assert "    object_orientation_std: float = 0.3\n" in text
    # palm_orientation is left exactly as upstream has it: this is a one-variable change.
    assert "    palm_orientation_weight: float = 1.0\n" in text
    assert "    palm_orientation_std: float = 0.4\n" in text


def test_std_is_a_parameter_and_reaches_both_files(tree: Path):
    assert main([str(tree), "--std", "0.45"]) == 0
    assert "    object_orientation_std: float = 0.45\n" in objective(tree)
    assert "object_orientation_std: 0.45" in (
        tree / "configs/objective/recon_body.yaml"
    ).read_text()
    # The std shapes the term; it must not have leaked into the weight.
    assert "    object_orientation_weight: float = 1.0\n" in objective(tree)


def test_a_tree_already_patched_at_another_std_is_refused(tree: Path):
    """Idempotence would otherwise keep the first std and report the second as applied."""
    main([str(tree), "--std", "0.3"])
    with pytest.raises(SystemExit) as caught:
        main([str(tree), "--std", "0.4"])
    assert "already carries object_orientation_std 0.3" in str(caught.value)
    assert "    object_orientation_std: float = 0.3\n" in objective(tree)


def test_a_tree_already_patched_at_the_same_std_is_a_no_op(tree: Path):
    main([str(tree), "--std", "0.45"])
    once = objective(tree)
    assert main([str(tree), "--std", "0.45"]) == 0
    assert objective(tree) == once


@pytest.mark.parametrize("bad", ["0", "-0.3", "nan", "inf"])
def test_a_std_that_cannot_shape_anything_is_refused(tree: Path, bad: str):
    before = objective(tree)
    with pytest.raises(SystemExit) as caught:
        main([str(tree), "--std", bad])
    assert "finite and positive" in str(caught.value)
    assert objective(tree) == before


def test_is_idempotent(tree: Path):
    main([str(tree)])
    once = objective(tree)
    main([str(tree)])
    assert objective(tree) == once


def test_check_mode_writes_nothing(tree: Path):
    before = objective(tree)
    assert main([str(tree), "--check"]) == 0
    assert objective(tree) == before


def test_refuses_a_tree_it_does_not_recognise(tree: Path):
    """A changed upstream must stop the run, not apply somewhere unintended."""
    path = tree / "objectives/recon_body.py"
    path.write_text(path.read_text().replace(
        "_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(12)", "# removed by this test"))
    with pytest.raises(SystemExit) as caught:
        main([str(tree)])
    assert "anchor occurs 0 times" in str(caught.value)
    # And nothing was written: the term name edit must not have landed either.
    assert "object_orientation" not in path.read_text()


# Verbatim shape of the curricula the container ships alongside the ReconBody ones: a different
# objective whose force_closure is an interpolation. Selecting curricula by the word
# "force_closure" made the patch refuse on these, and because it is atomic nothing was patched at
# all -- which stopped all eight boxes of the 2026-10-26 relaunch from training.
FOREIGN_CURRICULUM = """_target_: flash_chord.configuration.make_fixed_curriculum
thresholds: [2000, 3500]
stages:
  - _target_: flash_chord.configuration.make_curriculum_stage
    voc_scale: 1.0
    objective_weights: &base_weights
      object_keypoints: 0.0
      hand_keypoints: 0.25
      contact_wrench_support: 10.0
      force_closure: ${objective.force_closure.weight}
"""


def test_a_curriculum_for_another_objective_is_left_alone(tree: Path):
    """It must be skipped, said out loud, and must not stop the patch applying elsewhere."""
    foreign = tree / "configs/curriculum/reference.yaml"
    foreign.write_text(FOREIGN_CURRICULUM)
    assert main([str(tree)]) == 0
    # Untouched: adding object_orientation to another objective's key set would break set_weights
    # for THAT curriculum the moment it was used.
    assert foreign.read_text() == FOREIGN_CURRICULUM
    # And the ReconBody one did get patched.
    curriculum = (tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text()
    assert "object_orientation: 1.0" in curriculum


def test_every_reconbody_curriculum_is_patched_not_just_the_recipe_we_run(tree: Path):
    """The container ships recon_body_ppo.yaml too; a stage short one key raises mid-run."""
    other = tree / "configs/curriculum/recon_body_ppo.yaml"
    other.write_text((tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text())
    assert main([str(tree)]) == 0
    assert "object_orientation: 1.0" in other.read_text()


def test_a_tree_with_no_reconbody_curriculum_at_all_is_refused(tree: Path):
    recipe = tree / "configs/curriculum/recon_body_flash_sac.yaml"
    recipe.write_text(FOREIGN_CURRICULUM)
    with pytest.raises(SystemExit) as caught:
        main([str(tree)])
    assert "no curriculum config carries the ReconBody weight block" in str(caught.value)


def test_a_failure_anywhere_writes_nothing_anywhere(tree: Path):
    """The objective must not be left patched when the curriculum cannot be."""
    curriculum = tree / "configs/curriculum/recon_body_flash_sac.yaml"
    # Two weight blocks instead of one aliased block: the anchor is no longer unique.
    curriculum.write_text(curriculum.read_text() + "\n      force_closure: 5.0\n")
    with pytest.raises(SystemExit) as caught:
        main([str(tree)])
    assert "anchor occurs 2 times" in str(caught.value)
    assert "object_orientation" not in objective(tree)
    assert "object_orientation_weight" not in (tree / "configs/objective/recon_body.yaml").read_text()
