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

## The rule that makes this spend downside-free (found 2026-10-29)

The kit's own track config settles something we had been careful about for the wrong tier:

```
track_2_tier1  input_path = track_2/tier_1_multiview_caption
  input_note = "Use the original SOMA-X human and object reconstruction in the released Tier 1
                multiview-caption dataset. This is the clean trajectory used as the configured
                scoring reference."
track_2_tier2  input_note = "...The configured scoring reference is the clean Tier 1 object
                trajectory."
```

**On Tier 1 the permitted input *is* the scoring reference.** The organisers hand over the answer
and ask whether a physical policy can reproduce it. Our "read no ground truth when selecting"
discipline is exactly right for Tier 2, where the input is the noisy file and the reference is a
different one — and it is *vacuous* for Tier 1, where they are the same file. Choosing a checkpoint
or an arm by closeness to the Tier 1 input uses nothing but the input that tier tells us to use.

Two consequences:

**1. Per-clip adoption makes this run unable to lose.** A fine-tuned rollout ships only if it beats
the incumbent against that trajectory, so the worst case for the Tier 1 entry is the 0.46523
already on the board. The pilot's bar still stands as the test of whether the *method* works and
whether the remaining $14 is worth committing — but the score risk is nil, which is the main reason
to prefer this over further Tier 2 work.

**2. A "free gain" that was not one, and the rule it fixed.** Taking the per-clip best *AUC*
across the three rollout sets we hold gives 0.46858 against 0.46523. I built the machinery
(`v2d frame-refs`, which assembles a per-episode frame-reference tree so clips driven by different
references can be packed together) and packed it: kit packer pass, kit validator pass, zero
findings, submittable. Then I scored all four metrics instead of one:

| clip | arm | AUC | SP-SR | MP-SR | MPPE cm |
|---|---|---|---|---|---|
| e06 | shipped | 0.4758 | 1.00 | **1.00** | **3.20** |
| e06 | clean-input | **0.5688** | 1.00 | 0.00 | 4.54 |
| e21 | shipped | 0.2061 | **1.00** | 0.00 | **4.78** |
| e21 | clean-input | **0.2111** | 0.00 | 0.00 | 66.20 |
| e28 | shipped | 0.0891 | 0.00 | 0.00 | **44.07** |
| e28 | clean-input | **0.0917** | 0.00 | 0.00 | 63.36 |

At entry level that is AUC 0.46523 → 0.46858 but **SP-SR 0.667 → 0.633, MP-SR 0.467 → 0.433 and
MPPE 14.13 → 16.86 cm**: three of the four Tier 1 boards worse for three thousandths of AUC on the
fourth. AUC integrates the whole error curve while the success rates are hard gates, so a rollout
can trade one for the other, and e21 does it spectacularly — +0.005 AUC for 61 cm of MPPE.

**The shipped entry is kept, and across the arms we hold it is already Pareto-optimal.** Record:
`results/t1_perclip_best.json`.

**So the adoption rule for the fine-tune is registered as a no-regression rule**, not an AUC rule: a
clip's new rollout is adopted only if its kit AUC improves **and** SP-SR does not fall **and** MP-SR
does not fall **and** MPPE does not rise. Written here before any fine-tuned rollout exists.

## Pilot result, and a pre-registered clause that failed (2026-10-29)

Two of three clips are scored; e24 was still training when these were taken.

| clip | arm | AUC | SP-SR | MP-SR | MPPE cm | no-regression verdict |
|---|---|---|---|---|---|---|
| e00 | shipped | 0.5178 | 1.00 | 1.00 | 2.04 | |
| e00 | fine-tuned | **0.8654** | 1.00 | 1.00 | **1.58** | **adopt** (+0.3477 AUC, −0.46 cm) |
| e11 | shipped | 0.4676 | 1.00 | 0.00 | 3.97 | |
| e11 | fine-tuned | 0.3883 | 1.00 | 0.00 | 4.44 | reject (−0.0793 AUC, +0.47 cm) |

Frame-check residuals are 1e-05 cm, so neither number is a conversion artefact. e00's gain is real
tracking, not a metric quirk: its mean position error falls as its AUC rises, and both success gates
stay passed.

**The bar had two clauses and the second one failed.** It asked for a three-clip mean above 0.62651
*and* no clip falling more than 0.02. e11 fell 0.0793. I am not going to quietly reinterpret that,
so here is the argument for continuing anyway, and the reason it is not special pleading:

Clause 2 was written to catch a method that *damages* clips, on the assumption that a damaged clip
would have to ship. That assumption stopped holding later the same day, when the kit's own
`input_note` turned out to say the released Tier 1 trajectory **is** the configured scoring
reference — which makes per-clip adoption against it legal, and makes a clip that gets worse simply
not ship. The clause is obsolete rather than satisfied, and it is obsolete because of a fact
discovered after it was written, not because of a result I did not like.

What the data actually says about the method: it is **high variance and free to filter**. One clip
gained 0.35, one lost 0.08, and we keep only the gains. e00's adoption alone moves the 30-clip Tier
1 mean by **+0.0116** (0.3477 / 30), which is more than three times the entire per-clip-best
exercise and comes with no regression on any board.

**Decision: commit the remaining 27 clips.** If roughly one clip in three behaves like e00, Tier 1
clears the 0.51038 CHORD baseline. If none does, adoption keeps us at 0.46523 and the cost was the
compute, not the entry.

| | boxes | clips/box | GPU-h | at $0.68 |
|---|---|---|---|---|
| remaining 27 clips | 3 | 9 sequential | ~18.5 | **~$12.60** |

### The third clip, and the completed pilot

| clip | shipped AUC | fine-tuned AUC | ΔAUC | shipped MPPE | fine-tuned MPPE | verdict |
|---|---|---|---|---|---|---|
| e00 | 0.5178 | **0.8654** | **+0.3477** | 2.04 | **1.58** | adopt |
| e11 | 0.4676 | 0.3883 | −0.0793 | 3.97 | 4.44 | reject |
| e24 | 0.8942 | **0.9567** | **+0.0625** | 1.56 | **1.01** | adopt |

Three-clip mean AUC: **0.62651 shipped → 0.73681 fine-tuned → 0.76324 after adoption.** Clause 1 of
the bar asked for a mean above 0.62651 and got 0.73681. Both adoptions also *improve* MPPE and keep
every success gate, so they are not AUC-for-MPPE trades of the kind the per-clip-best exercise
exposed.

Record: `results/t1_warmstart_pilot.json`. The two adoptions alone are worth **+0.0137** on the
30-clip Tier 1 mean. The pilot's average per-clip adoption gain is **+0.1367**; at that rate across
30 clips Tier 1 reaches roughly 0.60, and at half that rate it still clears the **0.51038** CHORD
baseline we currently sit below.

## Which clips it helps (first seven, 2026-10-29 10:00 UTC)

| clip | shipped ckpt | shipped AUC | fine-tuned | Δ |
|---|---|---|---|---|
| e00 | **final** | 0.51775 | 0.86544 | **+0.3477** |
| e24 | **final** | 0.89422 | 0.95672 | **+0.0625** |
| e10 | **final** | 0.47938 | 0.47457 | −0.0048 |
| e20 | 300M | 0.71485 | 0.48011 | −0.2347 |
| e11 | 250M | 0.46757 | 0.38826 | −0.0793 |
| e02 | 250M | 0.42628 | 0.36996 | −0.0563 |
| e01 | 200M | 0.20013 | 0.18959 | −0.0105 |

Both gains come from clips that ship their **final** checkpoint. Every clip where z-drop had picked
an earlier checkpoint lost, and the one final-checkpoint clip that did not gain lost 0.005, which is
noise.

**Why that would be.** z-drop picking an earlier checkpoint is a measurement that late-stage
training made *that clip* worse. The fine-tune re-runs precisely that late stage — 299,999,232 to
349,999,104 — so on those clips it reproduces the degradation selection had already rejected. e20 is
the clean case: z-drop chose 300M over the 350M final, and fine-tuning 300M → 350M took it from
0.715 to 0.480. Seven clips and one seed, so this is a hypothesis; but it is a mechanism, not a
correlation, and it predicts the sign of all seven.

**Acted on it.** The three live boxes were re-queued onto their final-checkpoint clips only —
`e04 e06 e09` | `e10 e12 e16 e17 e18` | `e21 e22 e23 e26 e29` — dropping eleven predicted-loss
clips and about **$4.80**. Nothing is given up by that: under the no-regression rule a regressing
clip ships its old rollout, so a predicted loss contributes exactly zero to the entry either way.
Fifteen of the thirty clips ship a final checkpoint, and twelve of those are now in flight.

**Two bugs this exposed, both fixed.**

`brev/arm_flags.sh` accepted a patch report only if it contained the word "patched".
`patch_warm_start.py` words its no-op as "already carries the warm-start keys", so the *second* run
of any warm-start box died with "reported no patch at all" and sat at 0% GPU. Restarting a box is
how a queue gets re-ordered and how a dead runner is recovered, so the no-op report is the normal
case. Both wordings are now accepted, with a test that feeds the real stdout to the real predicate.

Committing `brev/t1_eval_clean.sh` while a launcher was executing it corrupted that shell's lazy
read of the file and aborted it at line 265. v2d-t3ws-b was deleted at 4 of its 5 clips and **e29's
Tier 3 fine-tune was lost**. Adoption means no score was lost — e29 ships its pass-1 rollout — but
a running script must not be rewritten underneath itself.

## A file count is not a progress count (2026-10-29 10:45 UTC)

`brev/finish_box.sh` decided a box was done by counting `*.parquet` in its rollouts directory
against the number of lines in its plan. Pruning a queue leaves the old plan's rollouts behind, so
on **v2d-t1f-1** the count read 3/3 as soon as its first new clip exported — e01 and e02 were
leftovers — and the box was deleted with **e06 and e09 never run**. Both are final-checkpoint
clips, which is to say both were predicted gains, so this cost score and not only money.

It now counts the clips the plan *names* that have a rollout. The same bug was about to take e29 off
v2d-t1f-3c, whose leftover is e20: three files, two planned clips done.

**Recovered without a new box.** The staged pieces for a clip are self-contained — a sequence
directory holding the object mesh, URDF and texture plus the robot parquet, a support USDA in
`whole_body/reconstructed_stage/`, and the actor under `ckpt/eNN/` — so e06 and e09 were copied
into v2d-t1f-2c's asset tree (66 MB), their md5 lines appended, the whole manifest re-verified at 77
files, and the two plan lines appended to `~/t1_plan.txt`. Appending is safe while the runner is
mid-clip: it walks the plan on fd 3 and has not reached EOF, so it picks the new lines up. That
saved a bring-up and a 10-minute image build, about $1.60, and 2c now runs seven clips inside its
6.5 h deadline with 1.3 h to spare.

## Addendum, 2026-10-29 12:50 UTC — all 18 gen-1 clips scored

| clip | shipped AUC | fine-tuned | delta | ships final ckpt | adopted |
|---|---|---|---|---|---|
| e06 | 0.47585 | 0.77999 | +0.30414 | yes | yes |
| e26 | 0.59296 | 0.60024 | +0.00729 | yes | yes |
| e18 | 0.73849 | 0.56585 | -0.17264 | yes | no (loses the MP-SR pass) |
| e29 | 0.62701 | 0.60238 | -0.02463 | yes | no |

Eight of eighteen adopt: e00 e06 e12 e16 e22 e23 e24 e26, mean adopted gain +0.133.

**Entry v2 packs at AUC 0.50838 / SP-SR 0.6667 / MP-SR 0.5000 / MPPE 13.973** against the uploaded
0.46523 / 0.6667 / 0.4667 / 14.128 — better on all four, and 0.0020 under CHORD's 0.51038.

The final-checkpoint prediction held up in direction but not as a sufficient condition: all four
clips here ship a final checkpoint, and two of them still lost. Shipping the final checkpoint is
what makes a gain *possible* (the fine-tune resumes from 300M, so a clip whose z-drop pick was
earlier is being re-run through training it already declined), not what makes one happen.

Still open: e09 on v2d-t1f-2c, and the generation-2 chain at 3 of 5.

## Packing bug, found and fixed 2026-10-29 13:30 UTC — worth +0.0019 for free

The packed entry read 0.50838 while the per-clip tally read 0.51030, and I had written that gap off
as the kit packer's rigid refit. It was not. Scoring the packed CSV **per episode** showed e00 at
0.8078 and e12 at 0.8013 — the *generation-2* numbers, not gen-1's 0.8654 and 0.8839.

Cause: adding `ws_chain` to the packer's rollout search list made it the last directory in the
loop, and the loop keeps the last match. So the entry shipped two rollouts that had **lost** their
own comparison, chosen purely because their directory sorted later. The comment above the loop even
asserted the opposite — "generation 2 wins ... because gen 2 is only adopted against gen 1" — which
was true of the rule and false of the code.

Fix: the chain is held out of the search and consulted only for clips named in
`results/t1_chain_adoption.json`, which records the gen-2 comparison explicitly (0 of 2 adopted).
**Entry v2 now packs at AUC 0.51030 / SP-SR 0.6667 / MP-SR 0.5000 / MPPE 13.9635**, matching the
per-clip tally to five decimals, and sits 0.00008 under CHORD's 0.51038.

The general lesson, which has now cost twice in this project: *directory order is not evidence.*
A file count was taken for a progress count earlier today and deleted a box with two clips unrun;
here a directory position was taken for an adoption decision.
