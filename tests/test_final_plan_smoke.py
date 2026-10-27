"""The smoke tests docs/FINAL_PLAN.md stands on: everything that can be checked without a box.

Each test here stands in for one line of that doc's SMOKE TEST section, so the evidence is a test
name rather than a pasted log, and it re-runs on every change.

What is NOT here, and is named in the doc instead: a real Hydra compose (`--cfg job`). It needs
the training image, which exists only on a box. The nearest free check is below -- every patched
config still parses, and carries the values the arm claims -- and the compose itself is the first
command of the first cell.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).parents[1]
UPSTREAM = Path(__file__).parent / "data/upstream"
SCRATCH = Path(
    os.environ.get(
        "V2D_SCRATCH",
        "/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/"
        "111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad",
    )
)
#: Every arm the final plan can launch, and the flags each one carries.
PLANNED_ARMS = {
    "T1": "start_curriculum,orientation_reward,object_position",
    "T3-VALID": "start_curriculum,orientation_reward,object_position",
    "T3-FULL": "start_curriculum,orientation_reward,object_position",
    "W3A": "start_curriculum,orientation_reward,object_position",
    "W3B": "start_curriculum,orientation_reward,object_position",
    "W3R": "start_curriculum,orientation_reward,object_position,body_joint_position",
}


def load(name: str):
    path = REPO / f"brev/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def second_wave(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(REPO / "brev/second_wave.sh"), *args],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "V2D_SCRATCH": str(SCRATCH)},
        cwd=str(REPO),
    )


# ------------------------------------------------------------------ the patch stack, twice over
def test_the_full_patch_stack_applies_and_the_second_pass_is_an_accepted_no_op(tree: Path, capsys):
    """A box patches once per clip; from the second clip on, every script is a no-op.

    The no-op report has to satisfy the runner's guard, which is what killed 2 clips and would
    have killed 15 more on 2026-10-27.
    """
    start, orientation = load("patch_start_curriculum"), load("patch_object_orientation")
    position, body = load("patch_object_position"), load("patch_body_joint_position")
    runs = [
        (start, [str(tree)]),
        (orientation, [str(tree), "--std", "0.3"]),
        (position, [str(tree), "--std", "0.08"]),
        (body, [str(tree), "--weight", "1.0"]),
    ]
    for module, argv in runs:
        assert module.main(argv) == 0
    capsys.readouterr()

    for module, argv in runs:
        assert module.main(argv) == 0
        report = capsys.readouterr().out.strip()
        assert "already patched" in report, report
        done = subprocess.run(
            ["bash", str(REPO / "brev/arm_flags.sh"), "--report-ok", report],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, f"the runner would kill the clip on: {report!r}"


def test_every_patched_config_still_parses_and_carries_its_arm(tree: Path):
    """The free stand-in for `--cfg job`: the files Hydra would compose are still valid YAML."""
    load("patch_object_position").main([str(tree), "--std", "0.08", "--weight", "5.0"])
    load("patch_body_joint_position").main([str(tree), "--weight", "1.0"])

    objective = yaml.safe_load((tree / "configs/objective/recon_body.yaml").read_text())
    assert objective["object_position_std"] == 0.08
    assert objective["object_position_weight"] == 5
    assert objective["body_joint_position_weight"] == 1
    assert objective["_target_"].endswith("ReconBodyObjectiveConfig")

    curriculum = yaml.safe_load((tree / "configs/curriculum/recon_body_flash_sac.yaml").read_text())
    assert len(curriculum["thresholds"]) == 9
    for stage in curriculum["stages"]:
        weights = stage["objective_weights"]
        assert weights["object_position"] == 5
        assert weights["body_joint_position"] == 1


# ------------------------------------------------------------------ the planner
def test_every_planned_arm_is_in_the_group_table_with_the_flags_the_plan_claims():
    table = (REPO / "brev/second_wave.sh").read_text()
    for group, flags in PLANNED_ARMS.items():
        row = next((line for line in table.splitlines() if line.startswith(f"{group}|")), None)
        assert row is not None, f"{group} is not in the group table"
        assert row.split("|")[2] == flags, row


def test_an_arm_whose_tree_is_not_staged_says_so_rather_than_planning_around_it():
    done = second_wave("--groups", "W3A", "--cap", "500")
    assert done.returncode == 0
    assert "NOT STAGED YET" in done.stderr
    assert "w3_bridge_v2" in done.stderr


def test_the_tier_three_validation_arm_plans_at_the_insurance_budget():
    done = second_wave("--groups", "T3-VALID", "--cap", "500")
    assert done.returncode == 0
    assert "steps 409600 (checkpoint every 409600)" in done.stdout
    assert "tier tier3" in done.stdout
    assert "30 clips" in done.stdout


def test_a_longer_budget_is_priced_as_the_gpu_hours_it_is():
    """W3R's 700M steps is two clips' worth of GPU in one cell; pricing it as one halves it."""
    done = second_wave("--groups", "W3R", "--cap", "500")
    assert done.returncode == 0
    assert "steps 699998208" in done.stdout
    hours = [
        float(part)
        for line in done.stdout.splitlines()
        if line.startswith("W3R")
        for part in [line.split("h/box")[0].split()[-1]]
    ]
    assert hours and hours[0] == pytest.approx(0.6 + 2 * 8.8, abs=0.2)


def test_the_planner_refuses_to_go_over_the_cap_it_was_given():
    done = second_wave("--groups", "T3-FULL", "--cap", "5")
    assert done.returncode == 3
    assert "REFUSED" in done.stderr


def test_a_clip_list_that_lands_later_overrides_the_placeholder(tmp_path: Path):
    """T3-FULL's clips come from interp's reconstruction-quality order, which is not in yet."""
    root = tmp_path / "second_wave"
    root.mkdir()
    (root / "clips_T3-FULL.txt").write_text("3 24\n")
    done = subprocess.run(
        [
            "bash",
            str(REPO / "brev/second_wave.sh"),
            "--groups",
            "T3-FULL",
            "--cap",
            "500",
            "--root",
            str(root),
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "V2D_SCRATCH": str(SCRATCH)},
        cwd=str(REPO),
    )
    assert done.returncode == 0
    assert "takes its clips from" in done.stderr
    assert "T3-FULL  2 clips" in done.stdout


# ------------------------------------------------------------------ the staged trees
@pytest.mark.skipif(not (SCRATCH / "t1_precision.md5").is_file(), reason="T1 is not staged here")
def test_the_tier_one_tree_matches_its_manifest():
    done = subprocess.run(
        ["md5sum", "-c", "--quiet", str(SCRATCH / "t1_precision.md5")],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(SCRATCH / "t1_precision"),
    )
    assert done.returncode == 0, done.stdout[-400:]
    assert len((SCRATCH / "t1_precision.md5").read_text().splitlines()) == 182


@pytest.mark.skipif(not (SCRATCH / "t1_precision").is_dir(), reason="T1 is not staged here")
def test_the_launcher_dry_run_names_every_box_the_map_and_the_arm():
    done = subprocess.run(
        [
            "bash",
            str(REPO / "brev/launch_t2_combined.sh"),
            "--map",
            str(SCRATCH / "tier1_episode_map.txt"),
            "--stage",
            str(SCRATCH / "t1_precision"),
            "--out",
            str(SCRATCH / "wave3/run_T1"),
            "--tier",
            "tier1",
            "--name-prefix",
            "v2d-t1p-b",
            "--arm-flags",
            "start_curriculum,orientation_reward,object_position",
            "--arm-env",
            "V2D_OBJECT_POSITION_STD=0.08",
            "--boxes",
            "8",
            "--clips-per-box",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, "V2D_SCRATCH": str(SCRATCH)},
        cwd=str(REPO),
    )
    assert done.returncode == 0
    assert "Nothing created" in done.stdout
    for index in range(1, 9):
        assert f"v2d-t1p-b{index:02d}" in done.stdout
    assert "tier1" in done.stdout
    assert "tier1_episode_map.txt" in done.stdout
    assert "start_curriculum,orientation_reward,object_position" in done.stdout


# ------------------------------------------------------------------ the adoption rule, 3 seeds
def test_the_adoption_rule_chains_over_a_three_seed_tree(tmp_path: Path):
    """W3A and W3B are the same recipe at two seeds, and T3-FULL may add a third.

    Three candidates for one clip must chain -- each judged against the incumbent of that moment --
    or a clip "adopts" twice and which rollout ships is undefined. The rule reads the input
    reference only, so the seed that sits closest to the noisy input wins even when another seed
    would score better against truth; that is the registered rule, and it is pinned here.
    """
    import json

    from test_adopt import write_cell
    from v2d.select.adopt import main as adopt_main

    old = tmp_path / "first_wave"
    write_cell(old / "e03", achieved_x=[0.10] * 40, reference_x=[0.0] * 40, auc=0.30)
    seeds = {"W3A": 0.08, "W3B": 0.02, "W3C": 0.05}
    groups = []
    for arm, offset in seeds.items():
        results = tmp_path / f"run_{arm}" / "results"
        write_cell(results / "e03", achieved_x=[offset] * 40, reference_x=[0.0] * 40, auc=0.10)
        groups += ["--group", f"{arm}={results}"]

    out = tmp_path / "adoption.json"
    adopt_main(["--old", str(old), *groups, "-o", str(out)])
    record = json.loads(out.read_text())

    clip = record["clips"]["e03"]
    assert clip["ships"]["arm"] == "W3B", "the seed nearest the INPUT reference must ship"
    assert [c["arm"] for c in clip["candidates"]] == ["W3A", "W3B", "W3C"]
    assert [c["became_incumbent"] for c in clip["candidates"]] == [True, True, False]
    assert all(c["decision"]["verdict_reads_ground_truth"] is False for c in clip["candidates"])
    assert [c["decision"]["judged_against"] for c in clip["candidates"]] == [
        "first wave",
        "W3A",
        "W3B",
    ]
