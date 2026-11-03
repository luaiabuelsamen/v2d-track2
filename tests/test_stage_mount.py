"""--mount and the stage directory have to agree, and nothing downstream can tell when they don't.

The incident: `brev/stage_t2_combined.sh --out $S/t1_precision --source ...` was run without
`--mount t1_precision`, so MOUNT kept its `t2_combined` default. The in-container
`object_mesh_paths`/`object_urdf_paths` written into all 30 parquets pointed at
/v2d/assets/t2_combined while the launcher mounts the tree at ~/v2d_assets/t1_precision.

Every check in the chain passed: the stage reported "30 clips, 182 files, 0 problems", the md5 gate
matched on the first attempt, the box was healthy with the image built. Then train_flash_sac died
in seconds against a mesh that was not there, produced no checkpoint, and the runner's
no-checkpoint path touches DONE -- which the launcher's poller reads as a clean finish. Seven
boxes reported "finished after 0 min, 0 trajectories" and deleted themselves.

So the mismatch is refused at argument time, where it is still free, and that refusal is pinned
here together with the escape hatch for a consumer that really does mount under another name.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / "brev" / "stage_t2_combined.sh"


def run_stage(tmp_path: Path, *args: str, env_extra: dict[str, str] | None = None):
    motion = tmp_path / "motion"
    (motion / "whole_body" / "soma").mkdir(parents=True)
    mapfile = tmp_path / "map.txt"
    mapfile.write_text("2 some_sequence_id\n")
    env = dict(os.environ, V2D_SCRATCH=str(tmp_path))
    env.update(env_extra or {})
    return subprocess.run(
        ["bash", str(STAGE), "--source", f"t1={motion}", "--map", str(mapfile), *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def test_a_mount_that_disagrees_with_the_stage_directory_is_refused(tmp_path: Path) -> None:
    done = run_stage(tmp_path, "--out", str(tmp_path / "t1_precision"))
    assert done.returncode == 2, f"expected a refusal, got rc={done.returncode}\n{done.stderr}"
    assert "--mount is 't2_combined'" in done.stderr
    assert "t1_precision" in done.stderr
    # The message has to name the fix, not just the fault.
    assert "Pass --mount t1_precision" in done.stderr


def test_the_matching_pair_is_not_refused(tmp_path: Path) -> None:
    """--mount t1_precision with the same basename gets past the check (and fails later, on data)."""
    done = run_stage(tmp_path, "--out", str(tmp_path / "t1_precision"), "--mount", "t1_precision")
    assert "--mount is" not in done.stderr, done.stderr


def test_the_default_out_always_agrees(tmp_path: Path) -> None:
    """With no --out, OUT defaults to $S/$MOUNT, so the pair cannot disagree."""
    done = run_stage(tmp_path, "--mount", "anything_at_all")
    assert "--mount is" not in done.stderr, done.stderr


def test_the_escape_hatch_is_honoured(tmp_path: Path) -> None:
    done = run_stage(
        tmp_path,
        "--out",
        str(tmp_path / "t1_precision"),
        env_extra={"V2D_ALLOW_MOUNT_MISMATCH": "1"},
    )
    assert "--mount is" not in done.stderr, done.stderr
