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
(`$S/t1_clean/motion/tier1`, 30/30 at 0.0 mm rigid-fit residual). Everything else stock. Seed 42.
Support per clip from the data.

## The pilot, and the bar

Three clips chosen now for spread across the shipped range, not after seeing results:

| clip | object | shipped (on the board) | clean-input eval | what fine-tuning has to beat |
|---|---|---|---|---|
| e24 | watering_can | 0.89422 | 0.76713 | 0.89422 |
| e00 | tall_bar_stool | 0.51775 | 0.30631 | 0.51775 |
| e09 | big_red_bowl | 0.30029 | 0.19878 | 0.30029 |
| **mean** | | **0.57075** | **0.42407** | **0.57075** |

**Bar, registered:** the full 30-clip fine-tune is launched only if the three-clip mean kit AUC
**beats 0.57075** and **no clip falls by more than 0.02** below its own shipped number. Ties go to
the shipped entry, which is already on the board. Per-clip adoption afterwards uses the same
input-only z-drop rule as the wave-2 entry, against the **clean** reference, so no truth is read to
decide what ships.

**What a failure means, stated now:** if the three-clip mean lands between 0.42407 (the cold swap)
and 0.57075 (shipped), fine-tuning works but 50M steps is not enough, and the honest read is that
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
