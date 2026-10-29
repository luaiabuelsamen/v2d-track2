# Tier 1 fine-tune: pre-registered before launch (2026-10-29)

## Why this and not more Tier 2 work

Our Tier 1 and Tier 2 boards carry the *same file* at AUC 0.46523, because all three tiers are
graded against the same truth and only the permitted input differs. Against Tier 2's published
CHORD baseline (0.374) that is 24% ahead. Against Tier 1's (**0.51038**) it is **9% behind**. Tier 1
is the only tier where we sit below a published baseline, and it is roughly a third of the track
rank, so it is the weakest position we hold.

Tier 1 permits a **clean** object trajectory as input. We have already measured the cheap way to
use it and it fails: driving the 30 shipped actors with the clean Tier 1 track scored **0.28937**,
down from 0.46523, with only 2 of 30 clips improving (`results/t1_clean_input_eval.json`). Each
actor is tuned to the trajectory it trained on, so a cleaner *different* reference is a distribution
shift. The reference has to be adapted **to**, not swapped under the policy.

That is what `brev/patch_warm_start.py` does, and it is the one asset we built and never ran.

## What runs

Per clip: the shipped wave-2 actor is loaded into a fresh learner via
`training.warm_start_actor_path`, the critic stays cold, the step counter starts at 299,999,232 so
the curriculum sits in the stage the actor was trained in, and a `jnp.where` guard scales the
actor's learning rate by 0.1 for the first 2,000 updates so a cold critic's early gradients cannot
walk a known-good actor off its policy. Budget 349,999,104 total, which leaves **49,999,872 steps**
of fine-tuning — one seventh of a full run, and exactly one checkpoint interval, so the usual save
cadence writes precisely one new checkpoint. Reference: the clean Tier 1 retarget
(`$S/t1_clean/motion/tier1`, 30/30 at 0.0 mm rigid-fit residual). Seed 42. Support per clip from
the data.

Recipe: `start_curriculum,orientation_reward,object_position,warm_start` with
`V2D_OBJECT_POSITION_STD=0.08` and orientation std 0.3 — the code patches arm A carried. The other
two parts of the shipping recipe are **reference** properties rather than patches and so travel with
the new reference automatically: CLEAN was the bridge-smoothed Tier 2 tree, and FING was interp's
re-solved finger parquets (there is no `patch_finger_targets.py`; the flag name maps to a file that
does not exist). The clean Tier 1 retarget carries its own solved `hand_finger_joints`, so the
fingers are present, from our own retarget rather than interp's.

## The pilot, and the bar

Three clips chosen now for spread across the shipped range, not after seeing results:

| clip | object | arm | shipped (on the board) | clean-input eval | fine-tuning must beat |
|---|---|---|---|---|---|
| e24 | watering_can | A | 0.89422 | 0.76713 | 0.89422 |
| e00 | tall_bar_stool | A | 0.51775 | 0.30631 | 0.51775 |
| e11 | light_blue_book | A | 0.46757 | &mdash; | 0.46757 |
| **mean** | | | **0.62651** | | **0.62651** |

**Why all three come from arm A, which changed the clip set.** The first draft of this page used
e24, e00 and e09 for spread. e09's shipped rollout comes from arm B4x, which was trained with
orientation std **0.8**; e24 and e00 come from arm A at std **0.3**. Fine-tuning them under one
recipe would have changed the reward for e09 as well as the reference, and the result could not then
be attributed. All three clips now ship from arm A, so the recipe below matches what every pilot
actor was trained on and the reference is the only variable. The cost is that the pilot contains no
clip from the weak tail; the tail's failures are Tier 2 input failures and are a separate question.

**Bar, registered:** the full 30-clip fine-tune is launched only if the three-clip mean kit AUC
**beats 0.62651** and **no clip falls by more than 0.02** below its own shipped number. Ties go to
the shipped entry, which is already on the board. Per-clip adoption afterwards uses the same
input-only z-drop rule as the wave-2 entry, against the **clean** reference, so no truth is read to
decide what ships.

**What a failure means, stated now:** if the three-clip mean lands between the cold-swap level
(0.42407 on the original clip set) and 0.62651 (shipped), fine-tuning works but 50M steps is not
enough, and the honest read is that
Tier 1 cannot be bought with the credit that remains. If it lands at or below 0.42407, the
warm start is not transferring the actor at all and the run is a bug, not a result — the runner
checks for the patch's `warm start: actor from` line and treats its absence as a failed fine-tune.

## Cost

| | boxes | clips/box | GPU-h | at $0.68 |
|---|---|---|---|---|
| pilot | 1 | 3 sequential | ~2.3 | **$1.60** |
| full run, if the bar clears | 3 | 10 sequential | ~21 | **$14.30** |

Credit remaining: **$30.38**. Pilot plus full run is $15.90, leaving $14.48 for the Tier 2 tail.

## This patch has never run on a GPU

There is no jax on the Jetson, so `tests/test_patch_warm_start.py` checks the edits land in the
right scope in the real upstream tree and re-implements the rate guard in numpy against
hand-computed values. It cannot check that `load_actor_checkpoint`, `env.set_curriculum(stage)` and
the `learner.replace(...)` call actually work together. The pilot is therefore also the patch's
first real test, which is the main reason it is one box and three clips rather than thirty.
