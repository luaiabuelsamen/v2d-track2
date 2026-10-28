"""brev/patch_contact_role.py against the real upstream text, and composed with the other arms.

The arm adds two reward terms and a frame-dependent weight to a warp kernel that only exists
inside the training image, so everything here is checked against the verbatim upstream copies in
tests/data/upstream. tests/test_arm_cr_kernel.py then compiles and launches the patched kernels.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import shutil
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[1]
UPSTREAM = Path(__file__).parent / "data/upstream"


def load(name: str):
    path = REPO / f"brev/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cr = load("patch_contact_role")
ori = load("patch_object_orientation")


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "flash_chord"
    (root / "objectives").mkdir(parents=True)
    (root / "configs/objective").mkdir(parents=True)
    (root / "configs/curriculum").mkdir(parents=True)
    shutil.copy(UPSTREAM / "objectives_recon_body.py", root / "objectives/recon_body.py")
    shutil.copy(
        UPSTREAM / "configs_objective_recon_body.yaml", root / "configs/objective/recon_body.yaml"
    )
    shutil.copy(
        UPSTREAM / "configs_curriculum_recon_body_flash_sac.yaml",
        root / "configs/curriculum/recon_body_flash_sac.yaml",
    )
    return root


def names_of(root: Path) -> tuple[str, ...]:
    source = (root / "objectives/recon_body.py").read_text()
    found = re.search(r"RECON_BODY_OBJECTIVE_TERM_NAMES = (\(.*?\))\n_RECON", source, re.S)
    return ast.literal_eval(found.group(1))


def count_of(root: Path) -> int:
    source = (root / "objectives/recon_body.py").read_text()
    return int(re.search(r"_RECON_BODY_OBJECTIVE_TERM_COUNT = wp\.constant\((\d+)\)", source).group(1))


def curriculum_keys(root: Path) -> list[set[str]]:
    loaded = yaml.safe_load((root / "configs/curriculum/recon_body_flash_sac.yaml").read_text())
    return [set(stage["objective_weights"]) for stage in loaded["stages"]]


def test_it_applies_to_the_real_upstream_text(tree: Path) -> None:
    assert cr.main([str(tree)]) == 0
    assert names_of(tree)[-2:] == ("contact_role_palm", "contact_consistency")
    assert count_of(tree) == 14
    ast.parse((tree / "objectives/recon_body.py").read_text())


def test_the_term_count_always_matches_the_number_of_names(tree: Path) -> None:
    cr.main([str(tree)])
    assert count_of(tree) == len(names_of(tree))


def test_every_curriculum_stage_names_the_new_terms(tree: Path) -> None:
    """set_weights() compares key SETS, so a stage short by one kills the run when it is entered."""
    cr.main([str(tree)])
    for keys in curriculum_keys(tree):
        assert keys == set(names_of(tree))


def test_the_weights_property_lists_them_in_term_order(tree: Path) -> None:
    cr.main([str(tree)])
    source = (tree / "objectives/recon_body.py").read_text()
    block = source[source.index("def weights(self)") :]
    block = block[: block.index("        )")]
    order = re.findall(r"self\.(\w+)_weight,", block)
    assert order[-2:] == ["contact_role_palm", "contact_consistency"]
    assert [f"{name}" for name in order] == [
        name for name in names_of(tree) if f"self.{name}_weight," in block
    ]


def test_the_yaml_carries_floats_not_ints(tree: Path) -> None:
    """A bare `2` is an int to YAML, and the config declares these fields float."""
    cr.main([str(tree), "--weight", "2", "--consistency-weight", "1"])
    loaded = yaml.safe_load((tree / "configs/objective/recon_body.yaml").read_text())
    for key in ("contact_role_palm_weight", "contact_consistency_weight"):
        assert isinstance(loaded[key], float), key


def test_it_is_idempotent(tree: Path) -> None:
    cr.main([str(tree)])
    before = (tree / "objectives/recon_body.py").read_text()
    assert cr.main([str(tree)]) == 0
    assert (tree / "objectives/recon_body.py").read_text() == before
    assert count_of(tree) == 14


def test_a_different_std_is_refused_rather_than_silently_ignored(tree: Path) -> None:
    cr.main([str(tree), "--std", "0.05"])
    with pytest.raises(SystemExit) as refusal:
        cr.main([str(tree), "--std", "0.09"])
    assert "already carries different contact-role settings" in str(refusal.value)
    assert "0.05" in str(refusal.value)


def test_a_different_body_weight_is_refused_too(tree: Path) -> None:
    cr.main([str(tree)])
    with pytest.raises(SystemExit) as refusal:
        cr.main([str(tree), "--body-weight-in-contact", "3.0"])
    assert "body_joint_position_weight_in_contact" in str(refusal.value)


@pytest.mark.parametrize(
    "argv", [["--std", "0"], ["--std", "-1"], ["--body-weight-in-contact", "-1"], ["--weight", "nan"]]
)
def test_impossible_parameters_are_refused(tree: Path, argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        cr.main([str(tree), *argv])
    assert "contact_role_palm" not in (tree / "objectives/recon_body.py").read_text()


def test_changed_upstream_is_a_refusal_and_writes_nothing(tree: Path) -> None:
    objective = tree / "objectives/recon_body.py"
    objective.write_text(
        objective.read_text().replace('    sides: tuple[str, ...] = ("left", "right")\n', "")
    )
    before = objective.read_text()
    with pytest.raises(SystemExit) as refusal:
        cr.main([str(tree)])
    assert "anchor occurs 0 times" in str(refusal.value)
    assert objective.read_text() == before
    assert "contact_role_palm" not in (tree / "configs/objective/recon_body.yaml").read_text()


def test_check_writes_nothing(tree: Path) -> None:
    before = (tree / "objectives/recon_body.py").read_text()
    assert cr.main([str(tree), "--check"]) == 0
    assert (tree / "objectives/recon_body.py").read_text() == before


def test_a_foreign_curriculum_is_left_alone(tree: Path) -> None:
    """Three other objectives name force_closure; adding our terms to them would break them."""
    foreign = tree / "configs/curriculum/tracking_250m_mixed_reset.yaml"
    foreign.write_text(
        "stages:\n  - objective_weights:\n      object_keypoints: 1.0\n"
        "      force_closure: ${objective.force_closure.weight}\n"
    )
    before = foreign.read_text()
    assert cr.main([str(tree)]) == 0
    assert foreign.read_text() == before


def test_it_refuses_when_no_curriculum_carries_the_weight_block(tree: Path) -> None:
    (tree / "configs/curriculum/recon_body_flash_sac.yaml").unlink()
    with pytest.raises(SystemExit) as refusal:
        cr.main([str(tree)])
    assert "no curriculum config carries the ReconBody weight block" in str(refusal.value)
    assert "contact_role_palm" not in (tree / "objectives/recon_body.py").read_text()


def test_it_composes_with_the_orientation_arm_in_the_canonical_order(tree: Path) -> None:
    """ARM_FLAGS_ORDER runs orientation_reward before contact_role; this is that order."""
    assert ori.main([str(tree)]) == 0
    assert cr.main([str(tree)]) == 0
    names = names_of(tree)
    assert count_of(tree) == len(names) == 15
    assert set(names[-3:]) == {"object_orientation", "contact_role_palm", "contact_consistency"}
    assert names.index("contact_role_palm") < names.index("contact_consistency")
    for keys in curriculum_keys(tree):
        assert keys == set(names)
    ast.parse((tree / "objectives/recon_body.py").read_text())


def test_the_kernel_reads_the_scale_from_the_config_pair(tree: Path) -> None:
    """The launch divides by the stock weight, so the effective in-contact weight is exact."""
    cr.main([str(tree), "--body-weight-in-contact", "1.0"])
    source = (tree / "objectives/recon_body.py").read_text()
    assert "self.config.body_joint_position_weight_in_contact" in source
    assert "/ max(self.config.body_joint_position_weight, 1e-12)" in source


def test_the_force_closure_reducer_still_gates_on_the_label(tree: Path) -> None:
    """`covered` is now computed for every hand, but only labelled hands reach `reward`."""
    cr.main([str(tree)])
    source = (tree / "objectives/recon_body.py").read_text()
    body = source[source.index("def reduce_recon_body_force_closure") :]
    body = body[: body.index("@wp.kernel", 1)] if "@wp.kernel" in body[1:] else body
    assert "if contact_active[frame * num_hands + hand] > 0.5:" in body
    assert body.index("hand_touching[world * num_hands + hand] = touching") < body.index(
        "if contact_active[frame * num_hands + hand] > 0.5:"
    )
    assert "reward += covered / float(num_basis)" in body


def test_the_reverse_order_refuses_instead_of_half_patching(tree: Path) -> None:
    """CR before ORI is not supported, and fails loudly rather than quietly.

    `patch_object_orientation.py` anchors its term names on the tuple ending in `force_closure`,
    which CR has by then appended to, so ORI refuses -- atomically, having written nothing. The
    cell then dies at the patch step instead of training a recipe nobody designed, and
    ARM_FLAGS_ORDER is what keeps it from happening through the launcher: `contact_role` comes
    after `orientation_reward` there, and brev/arm_flags.sh says why.
    """
    assert cr.main([str(tree)]) == 0
    before = (tree / "objectives/recon_body.py").read_text()
    with pytest.raises(SystemExit) as refusal:
        ori.main([str(tree)])
    assert "anchor occurs 0 times" in str(refusal.value)
    assert (tree / "objectives/recon_body.py").read_text() == before
    assert "object_orientation" not in (tree / "configs/objective/recon_body.yaml").read_text()


def test_the_canonical_flag_order_puts_orientation_before_contact_role() -> None:
    """The constraint above, pinned where it is actually enforced."""
    order = (REPO / "brev/arm_flags.sh").read_text()
    names = re.search(r'ARM_FLAGS_ORDER="([^"]+)"', order).group(1).split()
    assert names.index("orientation_reward") < names.index("contact_role")
