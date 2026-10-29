#!/usr/bin/env python3
"""Warm-start a flash_sac run from a shipped *actor* checkpoint, on a different reference.

Upstream can already resume a run: `training.resume` + `training.checkpoint_path`
(flash_sac/config.py), restored by `restore_learner_checkpoint` at runner.py:465-480. Neither half
of that helps us, for two reasons that are worth stating because they are the whole justification
for this patch:

1. `restore_learner_checkpoint` calls `_validate_metadata(path, "training_state")`. Every file in
   `ship_ckpts_wave2/*/policy_*/policy.safetensors` is `checkpoint_kind: policy` -- the deployable
   actor, parameters and BatchNorm statistics only. The `state_*.safetensors` training states were
   never pulled off the boxes and those boxes are gone, so there is nothing to resume FROM.
2. Resume also calls `validate_resume_reference_config`, which raises when the saved reference
   contract differs from the current one, and that contract includes `parquet`. Pointing a resumed
   run at the clean Tier 1 reference, or at a Tier 3 reconstruction, is exactly what it refuses.
   Resume continues the same run on the same data; it is not a transfer mechanism.

So this adds a narrower thing: `training.warm_start_actor_path`, which initialises the learner's
ACTOR from a `policy` checkpoint and leaves everything else cold, then starts the step counter
late so the curriculum sits where the actor was trained.

    python patch_warm_start.py /workspace/src/flash_chord [--check]

Three edits, each anchored on a short unique string and applied once:

  config.py   `warm_start_actor_path: str | None = None` and `warm_start_environment_steps: int`
              on TrainingConfig, plus `warm_start_guard_updates` and
              `warm_start_actor_rate_multiplier` on OptimizationConfig.

  runner.py   after `create_learner`, a branch parallel to `if config.resume:` that loads the
              actor with `load_actor_checkpoint` (the same call evaluation uses, so the same
              metadata validation applies and a training-state file is rejected), replaces
              `learner.actor.params`/`batch_stats`, sets `environment_steps`, and puts the
              curriculum at the stage that step count implies via the existing `stage_at`.
              The critic, its target, the temperature, the replay and the optimizer moments all
              stay freshly initialised -- a cold critic is the price of having only the actor.

  learner.py  the cold critic is also the risk: its early gradients are noise, and at the full
              late-schedule learning rate they would walk the actor straight off the policy we
              paid for. So `actor_rate` is scaled by `warm_start_actor_rate_multiplier` while
              `state.actor.schedule_step < warm_start_guard_updates`. It is `jnp.where`, not a
              Python branch, because this runs inside jit. It reuses the same seam that
              `residual_mean_learning_rate_multiplier` already occupies.

Defaults: `warm_start_environment_steps` is 299,999,232, the sixth checkpoint boundary, chosen so
the curriculum resolves to its final stage and a 349,999,104-step budget leaves 49,999,872 steps of
fine-tuning -- one seventh of a full run, and exactly one checkpoint interval, so the usual
save cadence produces precisely one new checkpoint.

Idempotent: a tree that already carries the keys is reported and left alone. Exits non-zero having
changed nothing if any anchor is missing or appears more than once, which is what makes an upstream
that has moved a refusal rather than a silent half-patch.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

#: TrainingConfig gains the path and the start step. Anchored on `resume: bool = False`, which is
#: the neighbouring switch and appears once.
CONFIG_EDITS: tuple[tuple[str, str], ...] = (
    (
        "    resume: bool = False\n    checkpoint_path: str | None = None\n",
        "    resume: bool = False\n"
        "    checkpoint_path: str | None = None\n"
        "    #: Initialise the ACTOR from a deployable `policy` checkpoint and leave the critic,\n"
        "    #: temperature, replay and optimizer moments cold. Unlike `resume` this does not\n"
        "    #: require a training_state checkpoint and does not validate the reference contract,\n"
        "    #: which is the point: it exists to fine-tune onto a DIFFERENT reference parquet.\n"
        "    warm_start_actor_path: str | None = None\n"
        "    #: The step the warm-started run reports itself at, so the curriculum resolves to the\n"
        "    #: stage the actor was trained in. 299,999,232 is the sixth checkpoint boundary.\n"
        "    warm_start_environment_steps: int = 299_999_232\n",
    ),
    (
        "    residual_mean_learning_rate_multiplier: float = 1.0\n",
        "    residual_mean_learning_rate_multiplier: float = 1.0\n"
        "    #: While warm-starting, the actor's learning rate is scaled by this for the first\n"
        "    #: `warm_start_guard_updates` updates, so a cold critic's early gradients cannot walk\n"
        "    #: a known-good actor off the policy. Inert unless warm_start_actor_path is set.\n"
        "    warm_start_actor_rate_multiplier: float = 0.1\n"
        "    warm_start_guard_updates: int = 2_000\n",
    ),
)

#: The learning-rate guard, inserted immediately after `actor_rate` is computed.
LEARNER_EDITS: tuple[tuple[str, str], ...] = (
    (
        "        * config.optimization.actor_learning_rate_multiplier\n    )\n",
        "        * config.optimization.actor_learning_rate_multiplier\n    )\n"
        "    # A warm-started actor faces a cold critic, whose first gradients are noise. Hold the\n"
        "    # actor's rate down until the critic has seen some data. `jnp.where` rather than an\n"
        "    # `if`, because this is traced. Inert when the multiplier is 1.0.\n"
        "    actor_rate = jnp.where(\n"
        "        state.actor.schedule_step < config.optimization.warm_start_guard_updates,\n"
        "        actor_rate * config.optimization.warm_start_actor_rate_multiplier,\n"
        "        actor_rate,\n"
        "    )\n",
    ),
)

#: The runner branch. Anchored on the resume branch's own first line, and placed BEFORE it so a
#: config that sets both still resumes -- resume carries a critic and is strictly better.
RUNNER_EDITS: tuple[tuple[str, str], ...] = (
    (
        "    if config.resume:\n",
        "    if config.warm_start_actor_path and not config.resume:\n"
        "        # Actor only. `load_actor_checkpoint` is the call evaluation uses, so a file that\n"
        "        # is not a deployable policy is refused here rather than loaded as garbage.\n"
        "        warm_actor, warm_metadata = load_actor_checkpoint(\n"
        "            config.warm_start_actor_path,\n"
        "            ActorInferenceState(\n"
        "                params=learner.actor.params, batch_stats=learner.actor.batch_stats\n"
        "            ),\n"
        "        )\n"
        "        warm_steps = int(config.warm_start_environment_steps)\n"
        "        _, warm_stage = stage_at(warm_steps)\n"
        "        env.set_curriculum(warm_stage)\n"
        "        learner = learner.replace(\n"
        "            actor=learner.actor.replace(\n"
        "                params=warm_actor.params, batch_stats=warm_actor.batch_stats\n"
        "            ),\n"
        "            environment_steps=jnp.asarray(warm_steps, dtype=learner.environment_steps.dtype),\n"
        "            observation=learner.observation.at[:].set(jax.device_put(env.reset(), jax_device)),\n"
        "            critic_context=learner.critic_context.at[:].set(\n"
        "                jax.device_put(env.initial_critic_context, jax_device)\n"
        "            ),\n"
        "        )\n"
        "        print(\n"
        "            f'warm start: actor from {config.warm_start_actor_path} '\n"
        "            f'({warm_metadata.get(\"user.checkpoint_kind\")}), steps set to {warm_steps:,}, '\n"
        "            f'curriculum stage {warm_stage}, critic cold, actor rate x'\n"
        "            f'{config.optimization.warm_start_actor_rate_multiplier} for the first '\n"
        "            f'{config.optimization.warm_start_guard_updates:,} updates',\n"
        "            flush=True,\n"
        "        )\n"
        "    if config.resume:\n",
    ),
    (
        "    restore_learner_checkpoint,\n",
        "    restore_learner_checkpoint,\n    load_actor_checkpoint,\n    ActorInferenceState,\n",
    ),
)

#: The dataclass is not what Hydra validates an override against. `configs/training/
#: reference_flash_sac.yaml` is the `training` node, Hydra composes it as a plain dict and
#: instantiates TrainingConfig from it via `_target_`, so a key that exists on the dataclass but
#: not in this YAML is refused at the command line with "Key 'warm_start_actor_path' is not in
#: struct, object_type=dict". That is exactly how the first pilot died: three clips, nine seconds
#: each, after the image build was paid for. The runner's preflight imported the dataclass and was
#: satisfied, which proved the wrong half.
YAML_EDITS: tuple[tuple[str, str], ...] = (
    (
        "resume: false\ncheckpoint_path: null\n",
        "resume: false\n"
        "checkpoint_path: null\n"
        "# Warm start: initialise the ACTOR from a deployable policy checkpoint. Present here and\n"
        "# not only on the dataclass, because Hydra validates overrides against this dict.\n"
        "warm_start_actor_path: null\n"
        "warm_start_environment_steps: 299999232\n",
    ),
    (
        "  residual_mean_learning_rate_multiplier: 1.0\n",
        "  residual_mean_learning_rate_multiplier: 1.0\n"
        "  warm_start_actor_rate_multiplier: 0.1\n"
        "  warm_start_guard_updates: 2000\n",
    ),
)

TARGETS: dict[str, tuple[tuple[str, str], ...]] = {
    "training/flash_sac/config.py": CONFIG_EDITS,
    "training/flash_sac/learner.py": LEARNER_EDITS,
    "training/flash_sac/runner.py": RUNNER_EDITS,
    "configs/training/reference_flash_sac.yaml": YAML_EDITS,
}

#: Present in a tree that already carries the patch.
MARKER = "warm_start_actor_path"


def plan_file(path: Path, edits: tuple[tuple[str, str], ...]) -> tuple[str, str | None]:
    """(report, new text or None if already patched). Raises if an anchor is missing/ambiguous."""
    text = path.read_text()
    if MARKER in text or "warm_start_actor_rate_multiplier" in text:
        return f"left alone: {path} -- already carries the warm-start keys", None
    for old, _new in edits:
        found = text.count(old)
        if found != 1:
            raise SystemExit(
                f"FATAL: {path} has {found} occurrences of the anchor\n"
                f"-----\n{old}-----\n"
                "which means this upstream is not the one the patch was written against. "
                "Nothing has been changed."
            )
    for old, new in edits:
        text = text.replace(old, new, 1)
    if path.suffix == ".py":
        try:
            ast.parse(text)
        except SyntaxError as error:  # pragma: no cover - a guard, not a path
            raise SystemExit(f"FATAL: patching {path} produced invalid Python: {error}") from error
    return f"patched: {path}", text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="the installed flash_chord package root")
    parser.add_argument("--check", action="store_true", help="report, change nothing")
    args = parser.parse_args(argv)

    plans: list[tuple[Path, str]] = []
    report: list[str] = []
    for relative, edits in TARGETS.items():
        path = args.root / relative
        if not path.is_file():
            raise SystemExit(f"FATAL: {path} is not there; is this a flash_chord source root?")
        line, text = plan_file(path, edits)
        report.append(line)
        if text is not None:
            plans.append((path, text))

    if args.check:
        for line in report:
            print(line.replace("patched:", "would patch:"))
        return 0

    # Write only after every file has planned successfully, so a failure part-way through cannot
    # leave a tree that imports but trains something nobody chose.
    for path, text in plans:
        path.write_text(text)
    for line in report:
        print(f"warm start | {line}")
    print(
        "warm start | actor-only init, cold critic, step counter and curriculum moved to the "
        "configured start"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
