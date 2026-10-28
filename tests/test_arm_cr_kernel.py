"""Compile and launch arm CR's patched warp kernels, here, on the CPU.

This is the smoke test the brief asked for, as far as this machine can take it: the real patched
kernel source, compiled by the real warp, launched on worlds whose answers are computable by hand.
flash_chord itself is not installed here and the training image is x86-64 CUDA, so what is NOT
covered is hydra composition, the reference loader, the SAC loop and throughput -- docs/ARM_CR.md
says so, and names them as the first commands of the first cell.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

from v2d.arms.kernel_harness import KERNELS, check, extract, term_names

REPO = Path(__file__).parents[1]
UPSTREAM = Path(__file__).parent / "data/upstream"

pytest.importorskip("warp", reason="the kernel smoke test needs warp-lang")


def patch(name: str, root: Path, argv: list[str] | None = None) -> None:
    spec = importlib.util.spec_from_file_location(name, REPO / f"brev/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main([str(root), *(argv or [])]) == 0


@pytest.fixture
def stock(tmp_path: Path) -> Path:
    root = tmp_path / "stock/flash_chord"
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


@pytest.fixture
def cr_only(stock: Path, tmp_path: Path) -> Path:
    root = tmp_path / "cr/flash_chord"
    shutil.copytree(stock, root)
    patch("patch_contact_role", root)
    return root


@pytest.fixture
def full_stack(stock: Path, tmp_path: Path) -> Path:
    """The recipe a CR cell actually trains: the shipping patches, then CR, in canonical order."""
    root = tmp_path / "stack/flash_chord"
    shutil.copytree(stock, root)
    patch("patch_start_curriculum", root)
    patch("patch_object_orientation", root)
    patch("patch_contact_role", root)
    return root


def test_the_patched_kernels_compile_and_the_new_terms_are_right(
    cr_only: Path, stock: Path, tmp_path: Path
) -> None:
    findings = check(cr_only, stock, tmp_path / "scratch")
    assert findings["pass"], findings
    assert findings["term_count"] == 14
    assert findings["all_terms_finite"]
    assert findings["new_terms_nonzero"]
    for world in findings["worlds"]:
        assert world["matches_hand_computed"], world


def test_it_compiles_on_top_of_the_whole_shipping_stack(
    full_stack: Path, stock: Path, tmp_path: Path
) -> None:
    findings = check(full_stack, stock, tmp_path / "scratch")
    assert findings["pass"], findings
    assert findings["term_count"] == 15
    assert "object_orientation" in findings["term_names"]


def test_the_twelve_stock_terms_are_untouched(cr_only: Path, stock: Path, tmp_path: Path) -> None:
    """Everything but body_joint_position must come out bit-identical to the stock kernel."""
    findings = check(cr_only, stock, tmp_path / "scratch")
    assert findings["force_closure_unchanged"]
    for world in findings["worlds"]:
        assert world["stock_terms_unchanged"], world


def test_the_body_weight_falls_only_on_contact_frames(
    cr_only: Path, stock: Path, tmp_path: Path
) -> None:
    findings = check(cr_only, stock, tmp_path / "scratch")
    scaled = [w for w in findings["worlds"] if w["in_contact"]]
    free = [w for w in findings["worlds"] if not w["in_contact"]]
    assert scaled and free
    for world in scaled:
        assert world["body_joint_position"] == pytest.approx(
            world["body_joint_position_stock"] * 0.2, abs=1e-6
        )
    for world in free:
        assert world["body_joint_position"] == pytest.approx(
            world["body_joint_position_stock"], abs=1e-9
        )


def test_a_free_hand_earns_nothing(cr_only: Path, stock: Path, tmp_path: Path) -> None:
    """World 2 has both hands in the free role with their palms on the object."""
    findings = check(cr_only, stock, tmp_path / "scratch")
    free = next(w for w in findings["worlds"] if not any(r > 0.5 for r in w["roles"]))
    assert free["contact_role_palm"] == 0.0
    assert free["contact_consistency"] == 0.0
    assert min(free["palm_gap_cm"]) < 8.0  # the palms ARE on it; only the role withholds reward


def test_consistency_disagrees_with_proximity_when_the_sim_does(
    cr_only: Path, stock: Path, tmp_path: Path
) -> None:
    """World 0 has both palms 2 cm out but only one hand loading: the terms must differ."""
    findings = check(cr_only, stock, tmp_path / "scratch")
    both = findings["worlds"][0]
    assert both["contact_role_palm"] > 0.8
    assert both["contact_consistency"] == pytest.approx(0.5)


def test_the_harness_refuses_a_kernel_argument_it_has_no_value_for(
    cr_only: Path, tmp_path: Path
) -> None:
    """A silently-zeroed argument would make every later assertion meaningless."""
    source = (cr_only / "objectives/recon_body.py").read_text()
    source = source.replace(
        "    contact_role_palm_std: float,\n",
        "    contact_role_palm_std: float,\n    something_new: float,\n",
        1,
    )
    (cr_only / "objectives/recon_body.py").write_text(source)
    with pytest.raises(ValueError, match="something_new"):
        check(cr_only, None, tmp_path / "scratch")


def test_the_extractor_needs_both_kernels(stock: Path) -> None:
    source = (stock / "objectives/recon_body.py").read_text()
    module = extract(source)
    assert all(f"def {name}(" in module for name in KERNELS)
    with pytest.raises(ValueError, match="missing"):
        extract(source.replace("def compose_recon_body_objective(", "def gone("))


def test_term_names_come_from_the_tree(cr_only: Path) -> None:
    names = term_names((cr_only / "objectives/recon_body.py").read_text())
    assert names[-2:] == ("contact_role_palm", "contact_consistency")
