"""`brev/patch_warm_start.py`: initialise a run's actor from a shipped policy checkpoint.

There is no jax, flax or optax on this Jetson, so none of this imports the patched tree. What it
can do without them is check the two things that actually go wrong with a patch like this: that the
edits land in the right scope in the real upstream file (not merely somewhere), and that the
arithmetic of the learning-rate guard is what the docstring claims. The guard is re-implemented
here in numpy against hand-computed values, the same way the arm-CR kernel harness stands in for
warp.
"""

from __future__ import annotations

import ast
import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest

UPSTREAM = Path("/home/jetson3/projects/v2d-upstream/robotic_grounding/flash_chord/src/flash_chord")
PATCH = Path(__file__).resolve().parent.parent / "brev" / "patch_warm_start.py"

pytestmark = pytest.mark.skipif(
    not UPSTREAM.is_dir(), reason="the upstream flash_chord source is not checked out here"
)


def load_patch():
    spec = importlib.util.spec_from_file_location("patch_warm_start", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    root = tmp_path / "flash_chord"
    shutil.copytree(UPSTREAM, root)
    return root


def apply(root: Path, *, check: bool = False) -> int:
    module = load_patch()
    argv = [str(root)]
    if check:
        argv.append("--check")
    return module.main(argv)


def test_it_patches_the_three_files_and_is_idempotent(tree: Path, capsys) -> None:
    assert apply(tree) == 0
    first = capsys.readouterr().out
    assert first.count("patched:") == 3
    assert apply(tree) == 0
    assert capsys.readouterr().out.count("left alone:") == 3


def test_check_changes_nothing(tree: Path) -> None:
    before = {p: p.read_bytes() for p in tree.rglob("*.py")}
    assert apply(tree, check=True) == 0
    assert all(path.read_bytes() == blob for path, blob in before.items())


def test_the_new_config_keys_exist_with_the_documented_defaults(tree: Path) -> None:
    apply(tree)
    source = (tree / "training/flash_sac/config.py").read_text()
    module = ast.parse(source)
    defaults: dict[str, object] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            try:
                defaults[node.target.id] = ast.literal_eval(node.value)
            except ValueError:
                defaults[node.target.id] = "<expr>"
    assert defaults["warm_start_actor_path"] is None
    # the sixth checkpoint boundary: one interval of 49,999,872 short of a full 349,999,104 run
    assert defaults["warm_start_environment_steps"] == 299_999_232
    assert defaults["warm_start_environment_steps"] + 49_999_872 == 349_999_104
    assert defaults["warm_start_actor_rate_multiplier"] == 0.1
    assert defaults["warm_start_guard_updates"] == 2_000


def test_the_runner_branch_lands_after_create_learner_and_before_resume(tree: Path) -> None:
    """Scope, not mere presence: `learner` must exist, and resume must still win if both are set."""
    apply(tree)
    lines = (tree / "training/flash_sac/runner.py").read_text().splitlines()
    created = next(i for i, line in enumerate(lines) if "learner, executables = create_learner(" in line)
    warm = next(i for i, line in enumerate(lines) if "if config.warm_start_actor_path" in line)
    resume = next(i for i, line in enumerate(lines) if line.strip() == "if config.resume:")
    assert created < warm < resume
    assert "and not config.resume" in lines[warm]
    # stage_at has to be in scope for the curriculum placement
    defined = next(i for i, line in enumerate(lines) if "def stage_at(" in line)
    assert defined < warm


def test_the_runner_uses_the_inference_loader_so_a_training_state_is_refused(tree: Path) -> None:
    apply(tree)
    source = (tree / "training/flash_sac/runner.py").read_text()
    assert "load_actor_checkpoint(" in source
    assert "ActorInferenceState(" in source
    # and it must be imported, or the branch is a NameError waiting for a config key
    assert "    load_actor_checkpoint,\n" in source
    assert "    ActorInferenceState,\n" in source
    assert "restore_learner_checkpoint" in source  # resume path left intact


def test_the_critic_is_left_cold(tree: Path) -> None:
    """The patch must not pretend to restore a critic it does not have."""
    apply(tree)
    source = (tree / "training/flash_sac/runner.py").read_text()
    start = source.index("if config.warm_start_actor_path")
    branch = source[start : source.index("if config.resume:", start)]
    assert "actor=learner.actor.replace(" in branch
    for forbidden in ("critic=learner.critic.replace(", "target_critic=", "temperature="):
        assert forbidden not in branch


def test_the_rate_guard_is_traced_not_branched(tree: Path) -> None:
    apply(tree)
    source = (tree / "training/flash_sac/learner.py").read_text()
    assert "actor_rate = jnp.where(" in source
    assert "state.actor.schedule_step < config.optimization.warm_start_guard_updates" in source
    # it must come after actor_rate is first bound, or it shadows nothing
    assert source.index("actor_rate = (") < source.index("actor_rate = jnp.where(")
    assert ast.parse(source)


def test_a_moved_upstream_is_refused_and_changes_nothing(tree: Path) -> None:
    config = tree / "training/flash_sac/config.py"
    config.write_text(config.read_text().replace("    resume: bool = False\n", ""))
    before = {p: p.read_bytes() for p in tree.rglob("*.py")}
    with pytest.raises(SystemExit, match="not the one the patch was written against"):
        apply(tree)
    assert all(path.read_bytes() == blob for path, blob in before.items())


def test_a_duplicated_anchor_is_refused(tree: Path) -> None:
    learner = tree / "training/flash_sac/learner.py"
    anchor = "        * config.optimization.actor_learning_rate_multiplier\n    )\n"
    learner.write_text(learner.read_text().replace(anchor, anchor * 2, 1))
    with pytest.raises(SystemExit, match="occurrences of the anchor"):
        apply(tree)


# --------------------------------------------------------------------------- the guard arithmetic
def guarded_rate(rate: float, step: int, *, updates: int, multiplier: float) -> float:
    """numpy stand-in for the patched `jnp.where`, so the claim can be checked without jax."""
    return float(np.where(step < updates, rate * multiplier, rate))


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 3.0e-5), (1_999, 3.0e-5), (2_000, 3.0e-4), (2_001, 3.0e-4), (10_000, 3.0e-4)],
)
def test_the_guard_holds_the_rate_down_for_exactly_the_first_n_updates(step, expected) -> None:
    rate = 3.0e-4
    got = guarded_rate(rate, step, updates=2_000, multiplier=0.1)
    assert got == pytest.approx(expected, rel=1e-12)


def test_the_guard_is_inert_at_multiplier_one() -> None:
    """A stock run must be bit-identical, so the default path has to be a no-op."""
    for step in (0, 1, 2_000, 1_000_000):
        assert guarded_rate(7.5e-4, step, updates=2_000, multiplier=1.0) == 7.5e-4
