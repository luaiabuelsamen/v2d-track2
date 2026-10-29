# Warm-starting a shipped actor onto a different reference

Fine-tuning the 30 wave-2 actors for ~50M steps instead of training 350M from scratch is worth
about **7× the compute**, which at $0.68/box-hour is the difference between $12.65 and $88. This is
what it takes, what it cannot do, and what would have to be true for it to be worth launching.

## Why upstream's `resume` does not do this

`training.resume` + `training.checkpoint_path` exist (`flash_sac/config.py`), restored at
`runner.py:465-480`. Both halves of that path refuse our case, and the refusals are the reason the
patch exists rather than a flag:

| | |
|---|---|
| `restore_learner_checkpoint` → `_validate_metadata(path, "training_state")` | every file we hold is `checkpoint_kind: policy` — the deployable actor, params and BatchNorm only. The `state_*.safetensors` training states were never pulled and those boxes are gone. |
| `validate_resume_reference_config` → `_validate_resolved_reference_config` | raises when the saved reference contract differs, and that contract **includes `parquet`**. Fine-tuning onto the clean Tier 1 reference, or onto a Tier 3 reconstruction, is precisely what it rejects. |

Resume continues the same run on the same data. It is not a transfer mechanism, and no combination
of its flags makes it one.

## What the patch adds

`brev/patch_warm_start.py`, three anchored edits, idempotent, refusing a moved upstream:

- **`training.warm_start_actor_path`** — loads the actor with `load_actor_checkpoint`, the same
  call evaluation uses, so a file that is not a deployable policy is refused rather than loaded as
  noise. Params and BatchNorm statistics only.
- **`training.warm_start_environment_steps`**, default **299,999,232** — the sixth checkpoint
  boundary. The step counter starts there, so `stage_at` puts the curriculum in its final stage
  (the actor's own regime) and a 349,999,104-step budget leaves **49,999,872** steps, which is one
  checkpoint interval, so the normal save cadence writes exactly one new checkpoint.
- **`optimization.warm_start_actor_rate_multiplier`** (0.1) and **`warm_start_guard_updates`**
  (2,000) — the critic, its target, the temperature, the replay and the optimizer moments are all
  **cold**, because an actor checkpoint carries none of them. A cold critic's first gradients are
  noise, and at the late-schedule learning rate they would walk a known-good actor off the policy
  we paid for. So the actor's rate is scaled while `schedule_step < guard_updates`, via `jnp.where`
  because it is traced, reusing the seam `residual_mean_learning_rate_multiplier` already occupies.

Everything is inert unless `warm_start_actor_path` is set, and `resume` still wins if both are
given — resume carries a critic and is strictly better.

## Smoke test, on the Jetson, free

There is no jax, flax or optax on this machine, so nothing imports the patched tree. 15 tests in
`tests/test_patch_warm_start.py` check the two things that actually go wrong:

- **scope, not presence**: the branch lands after `learner, executables = create_learner(` and
  before `if config.resume:`, with `stage_at` already defined above it; both new names are added
  to the import list; the critic is demonstrably *not* restored (the branch is asserted to contain
  no `critic=`, `target_critic=` or `temperature=`); `jnp.where` appears after `actor_rate` is
  bound. A moved or duplicated anchor is refused with every file byte-identical afterwards.
- **the guard's arithmetic**, re-implemented in numpy against hand-computed values: 3.0e-4 × 0.1
  at steps 0 and 1,999, full rate from step 2,000 on, and bit-identical at multiplier 1.0 so a
  stock run is unaffected.

Applied to a copy of the real upstream tree it patches 3 files, is idempotent, and `--check`
leaves every byte alone — all verified.

## Staging

`brev/stage_warm_start.sh --target tier1|tier3`, reusing the eval stager so there is one copy of
the path logic: each actor's recorded `/v2d/assets/<stage>/.../data.parquet` gets the **new**
reference, the support USDA goes where the scene builder looks, mesh paths are rebased, manifest
last. It also writes a plan carrying each clip's in-container actor path.

- **tier1 is ready**: 30/30 clean references retargeted, 0.0 mm rigid-fit residual. A 3-clip pilot
  stages in seconds (verified: 103 MB, manifest 21 = 21 files).
- **tier3 is not, and the script refuses rather than substituting**: FoundationPose gives an object
  *trajectory*; the trainer needs a whole-body solve against it, which is
  `v2d retarget --object-npz <recon>` — free, CPU, ~4 min/clip, **one worker** (two were OOM-killed
  on this 15 GB machine). The refusal prints that command.

## Cost

| | box-h | $ |
|---|---|---|
| 50M steps per clip (50/350 × 3.9 h measured) | 0.56 | 0.38 |
| 3-clip pilot, 1 box | 2.3 | **1.56** |
| 30 clips, 3 boxes × 10 clips sequential, + 0.6 h bring-up each | 18.6 | **12.65** |
| the same 30 clips trained from scratch, for comparison | ~130 | ~88 |

Credit is **$30.38** (Brev's email, 22:12 UTC 2026-10-28), less tonight's Tier 3 and Tier 1 eval
boxes. So the pilot is affordable twice over and the full 30 fits once, with nothing left for a
second seed.

## Pre-registered bar, written before launch

**The pilot is 3 clips, and the comparison is against the SAME input reference**, because
fine-tuning and changing the input are two variables and this run only tests one. Clips **e03,
e09, e18** — one table clip that scores well (0.7806), one that ships from a tail arm (e09), one
floor clip (e18) — fine-tuned on the clean Tier 1 reference and then evaluated on the clean Tier 1
reference, against the shipped actor evaluated on that same reference, which is what the Tier 1
eval box is producing tonight.

Adopt only if **mean kit AUC over the three clips rises** and **no clip drops below its own shipped
number minus 0.02**. A cold critic damaging the actor shows up as exactly that drop, which is the
failure this bar is built to catch. If the pilot passes, the same recipe goes to all 30 for $12.65;
if it fails, the patch is recorded as tried and the credit stays for Tier 3.

Nothing here is launched. The last command is the stage, and the create is Luai's.
