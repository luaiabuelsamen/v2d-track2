# AB4 — the four-clip A/B, pre-registered

Written **2026-10-26, before any arm is launched**. Clips, recipe, bar and decision rule are fixed
here and are not editable afterwards. Luai approves or rejects the arms; the stock record below is
already measured and is the thing to beat.

## Fixed for every cell

| | |
|---|---|
| clips | **e03** (table, mug), **e09** (two-handed, bowl), **e18** (floor, trash can), **e24** (floor) |
| recipe | `experiment=g1_recon_body_flash_sac`, NVIDIA stock |
| steps | **349,999,104**, checkpoint every 49,999,872 (7 checkpoints) |
| seed | `reset.seed=42` |
| eval | `evaluation.world_count=1` — the submission world count |
| scene.support | per clip, read from that clip's support USDA (`brev/support_flag.sh`) |
| score | the **official kit scorer**, `v2d rollout-to-npz` → `v2d score --tier tier2` |

Each arm changes **exactly one thing**. Anything else differing is a confound and voids the cell.

## The stock record to beat (kit scorer, final checkpoint)

From `results/ab350_raw_baseline.json`; e03/e09/e18 are RAW on the released Tier 2 retarget, e24 is
the full350 run on the gated-cleaned reference.

| clip | kit AUC | kit MPPE | drops? |
|---|---|---|---|
| e03 | 0.0602 | 89.78 cm | **yes** |
| e09 | 0.2409 | 5.31 cm | no |
| e18 | 0.3826 | 4.17 cm | no |
| e24 | 0.4500 | 4.16 cm | no |
| **mean** | **0.28342** | **25.86 cm** | **1 of 4** |

## Arms

**Arm A — palm-fix retarget** (ws/hand, `v2d retarget --palm-standoff DIR`, which implies
`--palm-centre`). The reference parquet changes; recipe untouched. ws/hand measured the palm offset
falling from 3.3–4.7 cm to 0.13–0.37 cm with the object-frame check at 0.0 mm on 18 motions, so the
grasp geometry the policy is asked to reproduce is physically reachable. Stand-off NPZs exist for
e03, e09, e24; **e18's must be generated first** (`v2d hand-standoff --episodes 18`, CPU, free).

**Arm B — an object orientation reward term.** NVIDIA's `configs/objective/recon_body.yaml` has
**no object-orientation term at all**: the 12 terms are termination, action-rate-l2, action-l2,
soft-joint-limit, pelvis-position, pelvis-orientation, body-joint-position, **object-position**,
progress, palm-position, palm-orientation, force-closure. The object is shaped by position only,
`exp(-|p-p*|²/σ²)` with `object_position_std = 0.2`
(`flash_chord/objectives/recon_body.py:276-278`). So "raise the object orientation weight" cannot be
done as a config override — the arm is a 13th term in our fork, mirroring the existing
`palm_orientation` term on the tracked object quaternion against the reference.

Value: **weight 1.0, std 0.3 rad** (`brev/patch_object_orientation.py --std`, default 0.3). The
weight is copied from `palm_orientation_weight: 1.0` — `pelvis_orientation` is 1.0 as well, so 1.0
is the house calibration for an orientation term here, and copying it keeps the arm a single
addition rather than a retuning of the recipe. The std is **not** copied: `palm_orientation_std` is
0.4, but the object's angle is scored and the palm's is not. Both values are flat across the
63–96° our checkpoints actually sit at, so the choice only changes the slope inside the 0–20° band
where ADD-AUC is won, and 0.3 (half reward at 14.3°) doubles it. The evaluator's own orientation
gate is 1.5 rad, far too loose to read a value off, so the std is a parameter rather than a
derivation.

Why this term and not a bigger position weight: on the three clips that already hold the object,
MPPE is 4.2–5.3 cm while the object **orientation** error is 25.0°, 38.5° and 44.8°. Position is
nearly solved and orientation is not, and orientation is scored (ADD-AUC runs over mesh points) but
never rewarded. Arm B closes exactly that gap.

## Decision rule, fixed now

An arm is **adopted** only if, over the four clips:

1. its **mean kit AUC beats the stock record on both** comparisons — final checkpoint (> 0.28342)
   and no-drop-selected checkpoint (> 0.36635, see below); **and**
2. **no additional clip drops** — at most 1 of 4, which is stock.

**Ties go to stock.** Both comparisons must be won, so an arm cannot be adopted on a single
favourable checkpoint. An adopted arm runs on all 30 clips as track 4, inside the cap.

## Free item: checkpoint selection with no ground truth

Each clip already has 7 paid-for checkpoints, and the final one is not the best. Selection rule,
using only the **noisy input reference** the policy trained against — Tier 1 clean is never read:

> Reject a checkpoint if the achieved object sits **≥ 5 cm below** its reference **and** stays
> under z = 0.15 m for **≥ 10 consecutive frames**. Among the rest take the smallest mean distance
> to that same input reference. If none survive, take the smallest mean distance overall.

The 5 cm margin is half the evaluator's own 0.10 m object position threshold; 10 frames is 0.2 s at
the 50 Hz control rate. Both guard against the bare `z < 0.15 m` form, which fires on a 2-frame,
7 mm *lift lag* and so rejects 6 of e24's 7 checkpoints for nothing.

Applied to our existing 28 checkpoints (`scratchpad/nodrop_select_margin.json`):

| clip | drop-free checkpoints | picks | kit AUC | was (final) |
|---|---|---|---|---|
| e03 | none | **99M** | 0.3556 | 0.0602 |
| e09 | 99, 149, 249, 349M | 349M | 0.2409 | 0.2409 |
| e18 | all 7 | 349M | 0.3826 | 0.3826 |
| e24 | 199, 299, 349M | **299M** | 0.4863 | 0.4500 |
| **mean** | | | **0.36635** | 0.28342 |

**Four-clip mean kit AUC 0.28342 → 0.36635 (+0.083), mean MPPE 25.86 → 13.04 cm, for no compute and
no ground truth.** It applies unchanged to all 30 clips at submission time. The bare-line variant
gives 0.35728; the margin variant is registered because it is justified by the evaluator's own
threshold and the control rate, and because the e24 artifact is visible in the *z* columns alone —
the AUCs of both variants are quoted here so the choice is on the record.

## Cost and budget position

4.4 h per clip at 349,999,104 steps (measured: 13,461 s and 13,769 s of training plus seven
evaluations), $0.68/h on `massedcompute_A6000_plus`.

| | cells | boxes | GPU-h | cost |
|---|---|---|---|---|
| track 1 — 30-clip stock Tier 2 | 30 | 8, 3–4 clips each, sequential | 135 | **$91.9** |
| track 2 — arm B, 4 clips | 4 | 2, 2 clips each | 18.4 | **$12.5** |
| track 3 — arm A, 4 clips | 4 | 1, 4 clips | 18.0 | **$12.2** |
| **tonight** | | **11** | **171** | **$116.6** |
| track 4 — an adopted arm on 30 clips | 30 | 8 | 135 | $91.9 |
| **total if track 4 runs** | | | | **$208.5** |

Budget: **$21.13 already spent**, **$230 newly authorized** (hard cap), so the ceiling is $251.13
cumulative. $208.5 of the $230 leaves **$21.5** of headroom. Every box carries a 24 h deletion
authority, a dead-runner check that fails loudly, and a 10-minute pull. The live ledger is
`scratchpad/spend_closed.txt` plus `scratchpad/spend.sh`, and the hard stop is a refusal to create a
box once committed spend plus projected spend reaches $230 of new money.
