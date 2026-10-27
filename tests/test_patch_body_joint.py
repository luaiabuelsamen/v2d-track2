"""The body-joint weight patch, proven against the real upstream text before a box runs it.

The weight that decides training is the curriculum's -- `set_weights` applies the curriculum
mapping every stage -- so a patch that changes the objective config alone reports success and
trains the stock recipe. That is the failure these tests exist for.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path

import pytest

UPSTREAM = Path(__file__).parent / "data/upstream"


def load(name: str):
    path = Path(__file__).parents[1] / f"brev/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


body = load("patch_body_joint_position")


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


def objective_yaml(tree: Path) -> str:
    return (tree / "configs/objective/recon_body.yaml").read_text()


def curriculum(tree: Path) -> str:
    return (tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text()


def test_the_weight_lands_in_all_three_files(tree: Path):
    assert body.main([str(tree), "--weight", "1.0"]) == 0
    assert (
        "body_joint_position_weight: float = 1\n" in (tree / "objectives/recon_body.py").read_text()
    )
    assert "body_joint_position_weight: 1\n" in objective_yaml(tree)
    assert "      body_joint_position: 1\n" in curriculum(tree)
    assert "      body_joint_position: 5.0\n" not in curriculum(tree)


def test_the_patched_objective_is_still_valid_python(tree: Path):
    body.main([str(tree), "--weight", "1.0"])
    ast.parse((tree / "objectives/recon_body.py").read_text())


def test_the_marker_records_what_the_tree_carries(tree: Path):
    body.main([str(tree), "--weight", "1.0"])
    assert "# v2d body-joint-position weight=1 std=none" in objective_yaml(tree)
    assert body.carried(tree / "configs/objective/recon_body.yaml") == (1.0, None)


def test_a_second_run_at_the_same_values_is_a_no_op_that_reports_it(tree: Path, capsys):
    body.main([str(tree), "--weight", "1.0"])
    before = objective_yaml(tree)
    capsys.readouterr()

    assert body.main([str(tree), "--weight", "1.0"]) == 0

    printed = capsys.readouterr().out
    assert "already patched" in printed
    assert objective_yaml(tree) == before


def test_the_no_op_report_satisfies_the_runners_arm_flag_guard(tree: Path, capsys):
    """The colon-guard bug: the runner killed every second clip on a box over this exact line."""
    import subprocess

    body.main([str(tree), "--weight", "1.0"])
    capsys.readouterr()
    body.main([str(tree), "--weight", "1.0"])
    printed = capsys.readouterr().out.strip()

    done = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "brev/arm_flags.sh"), "--report-ok", printed],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, f"the runner would kill the clip on: {printed!r}"


def test_a_second_run_at_other_values_refuses_rather_than_reporting_success(tree: Path):
    body.main([str(tree), "--weight", "1.0"])
    with pytest.raises(SystemExit, match="already carries weight=1"):
        body.main([str(tree), "--weight", "2.0"])


def test_check_writes_nothing(tree: Path):
    before = objective_yaml(tree), curriculum(tree)
    assert body.main([str(tree), "--weight", "1.0", "--check"]) == 0
    assert (objective_yaml(tree), curriculum(tree)) == before


def test_a_curriculum_that_does_not_name_the_weight_is_a_hard_failure(tree: Path):
    path = tree / "configs/curriculum/recon_body_flash_sac.yaml"
    path.write_text(path.read_text().replace("      body_joint_position: 5.0\n", ""))
    with pytest.raises(SystemExit, match="would not reach training"):
        body.main([str(tree), "--weight", "1.0"])


def test_a_non_positive_weight_is_refused(tree: Path):
    with pytest.raises(SystemExit, match="finite and positive"):
        body.main([str(tree), "--weight", "0"])


def test_the_std_is_optional_and_left_alone_by_default(tree: Path):
    body.main([str(tree), "--weight", "1.0"])
    assert "body_joint_position_std: 1.0\n" in objective_yaml(tree)
    assert "std=none" in objective_yaml(tree)


def test_the_std_lands_when_asked_for(tree: Path):
    body.main([str(tree), "--weight", "1.0", "--std", "0.5"])
    assert "body_joint_position_std: 0.5\n" in objective_yaml(tree)
    assert (
        "body_joint_position_std: float = 0.5\n" in (tree / "objectives/recon_body.py").read_text()
    )
