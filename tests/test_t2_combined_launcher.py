"""Guards on the 30-clip run's plan: the queue, and that scoring cannot leak into selection.

Two things here would be expensive and silent if they broke. A clip dropped from the queue is a
clip missing from the submission, found out a day and ~$100 later. And the kit scorer reads the
Tier 1 clean truth, so if the checkpoint choice could see a score, the choice would be made on the
answer -- which is the one thing the selection rule exists to avoid.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from test_pick_checkpoint import held, run, write_checkpoint

REPO = Path(__file__).parents[1]
LAUNCHER = REPO / "brev/launch_t2_combined.sh"


def fake_stage(root: Path, episodes: list[int]) -> tuple[Path, Path]:
    """The least a stage needs for the launcher to agree to plan against it."""
    stage = root / "t2_combined"
    (stage / "whole_body/soma").mkdir(parents=True)
    (stage / "provenance.json").write_text(json.dumps({"clips": {}}))
    (stage / "sources.txt").write_text(
        "".join(f"{episode} seq-{episode} raw /dev/null\n" for episode in episodes)
    )
    Path(f"{stage}.md5").write_text("")
    mapping = root / "map.txt"
    mapping.write_text("".join(f"{episode} seq-{episode}\n" for episode in episodes))
    return stage, mapping


def plan(root: Path, episodes: list[int], *extra: str) -> subprocess.CompletedProcess:
    stage, mapping = fake_stage(root, episodes)
    return subprocess.run(
        ["bash", str(LAUNCHER), "--stage", str(stage), "--map", str(mapping), *extra],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": str(root), "V2D_SCRATCH": str(root)},
    )


def test_the_plan_creates_nothing_and_says_so(tmp_path: Path):
    done = plan(tmp_path, list(range(30)))
    assert done.returncode == 0, done.stderr
    assert "plan only. Nothing created." in done.stdout
    assert "--go" in done.stdout


def test_the_queue_holds_every_clip_once_with_the_recorded_four_first(tmp_path: Path):
    episodes = list(range(30))
    assert plan(tmp_path, episodes).returncode == 0
    queue = (tmp_path / "t2_combined_run/queue.txt").read_text().splitlines()
    ordered = [int(line.split()[0]) for line in queue if line.strip()]
    # The four clips with a stock record lead, so the first read ~5 h in is comparable.
    assert ordered[:4] == [3, 9, 18, 24]
    # And nothing is lost or repeated: a clip missing here is a clip missing from the submission.
    assert sorted(ordered) == episodes


def test_a_queue_that_lost_a_clip_stops_the_launch(tmp_path: Path):
    # --first naming a clip the map does not have must not silently shrink the run.
    done = plan(tmp_path, [1, 2, 3], "--first", "3 99")
    assert done.returncode == 2
    assert "not in" in done.stderr


def test_an_unknown_arm_flag_stops_the_launch_before_any_box(tmp_path: Path):
    done = plan(tmp_path, [1], "--arm-flags", "orientation_rewrad")
    assert done.returncode == 2
    assert "unknown arm flag" in done.stderr


def test_a_missing_stage_stops_the_launch(tmp_path: Path):
    done = subprocess.run(
        ["bash", str(LAUNCHER), "--stage", str(tmp_path / "nope"), "--map", str(tmp_path / "m")],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "V2D_SCRATCH": str(tmp_path)},
    )
    assert done.returncode == 2
    assert "no episode map" in done.stderr or "no staged references" in done.stderr


def test_a_kit_score_sitting_next_to_a_checkpoint_cannot_change_the_pick(tmp_path: Path):
    """score_pulled.sh writes score.json into these very directories; selection must not see it."""
    cell = tmp_path / "e03"
    write_checkpoint(cell, 100, *held(0.05))
    write_checkpoint(cell, 200, *held(0.02))  # closest to the input reference: the pick
    before = run(tmp_path, f"e03={cell}")["episodes"]["e03"]["selected_environment_steps"]
    assert before == 200
    # Now plant scores that would reverse the choice if they were read at all.
    for steps, auc in ((100, 0.99), (200, 0.01)):
        (cell / f"policy_{steps}/score.json").write_text(
            json.dumps(
                {
                    "overall": {
                        "add_auc": auc,
                        "mppe_cm": 1.0,
                        "spider_sr": 1.0,
                        "maniptrans_sr": 1.0,
                    }
                }
            )
        )
    after = run(tmp_path, f"e03={cell}")["episodes"]["e03"]["selected_environment_steps"]
    assert after == 200


def test_the_runner_the_launcher_starts_is_the_one_we_test(tmp_path: Path):
    # The launcher ships brev/run_tier2_full.sh and every file it sources; if the runner stopped
    # sourcing arm_flags.sh, the recipe would silently become stock.
    runner = (REPO / "brev/run_tier2_full.sh").read_text()
    launcher = LAUNCHER.read_text()
    assert 'arm_flags.sh"' in runner
    assert "arm_flags_apply" in runner
    assert "$ARM_OVERRIDES" in runner
    assert "run_tier2_full.sh" in launcher
    assert "arm_flags_files" in launcher
    # And the clip's own asset root and results dir are passed, not assumed.
    assert "ASSET_ROOT=" in launcher and "RESULTS_DIR=" in launcher


# ---------------------------------------------------------------- box names and replacement
# A failed box is now replaced, which needs a name no other slot will take. Three of eight creates
# failed provider-side on 2026-10-26 and nothing replaced them: 30 clips, 28 slots.
HELPER = REPO / "brev/box_names.sh"


def claim(tmp_path: Path, max_index: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", f'. "{HELPER}"; box_name_claim "{tmp_path}/names" {max_index}'],
        capture_output=True, text=True, check=False,
    )


def test_each_name_is_handed_out_once(tmp_path: Path):
    assert [claim(tmp_path, 3).stdout.strip() for _ in range(3)] == ["1", "2", "3"]


def test_the_cap_is_the_end_of_it(tmp_path: Path):
    for _ in range(2):
        claim(tmp_path, 2)
    refused = claim(tmp_path, 2)
    assert refused.returncode == 1
    assert refused.stdout.strip() == ""


def test_concurrent_slots_never_take_the_same_name(tmp_path: Path):
    """mkdir is the lock; without it two slots race and two boxes get one name."""
    racers = [
        subprocess.Popen(
            ["bash", "-c", f'. "{HELPER}"; box_name_claim "{tmp_path}/names" 8'],
            stdout=subprocess.PIPE, text=True,
        )
        for _ in range(8)
    ]
    got = sorted(int(r.communicate()[0].strip()) for r in racers)
    assert got == list(range(1, 9))


def test_a_specific_name_can_be_reserved_once(tmp_path: Path):
    reserve = f'. "{HELPER}"; box_name_reserve "{tmp_path}/names" 9'
    assert subprocess.run(["bash", "-c", reserve], check=False).returncode == 0
    assert subprocess.run(["bash", "-c", reserve], check=False).returncode == 1


def test_planning_hands_out_no_box_name(tmp_path: Path):
    """The plan path must stay free of side effects now that it can allocate names."""
    done = plan(tmp_path, list(range(30)))
    assert done.returncode == 0, done.stderr
    assert "plan only. Nothing created." in done.stdout
    assert not (tmp_path / "t2_combined_run/boxnames").exists()


def test_the_replacement_budget_is_a_flag_and_bounds_the_total(tmp_path: Path):
    """--max-replacements is what stops a provider outage being paid for one box at a time."""
    done = plan(tmp_path, list(range(30)), "--boxes", "8", "--max-replacements", "2")
    assert done.returncode == 0, done.stderr
    # The cap is BOXES + MAX_REPLACEMENTS, and the plan must still refuse to create anything.
    assert "plan only. Nothing created." in done.stdout
    assert "unknown argument" not in done.stderr
