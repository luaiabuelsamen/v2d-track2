# T1P: Tier 1 precision, from scratch on the clean references

**Pre-registered 2026-11-02, before launch.** One variable against the shipped Tier 1 entry: the
policies are trained from scratch on the *clean tier_1* reference rather than on the noisy Tier 2
retarget that 16 of the 30 shipped clips chase. Recipe otherwise the shipping one — START +
orientation std 0.3 + object_position std 0.08 — seed 42, 349,999,104 steps, `world_count=1`,
`scene.support` per clip from the data.

## Why this clip set, and why this is the only lever left

Three of the four cheapest rungs on all twelve boards are Tier 1, priced against the next
*distinct* score (ties share a rank, confirmed in writing by the organisers on 2026-11-02):

| board | our score | +1 rank needs |
|---|---|---|
| tier1-mp-sr | 0.50000 | **1 clip** crossing the 3 cm / 30 deg gate |
| tier1-mppe | 12.06020 | **+1.9 cm summed** over 30 clips (DAP at 11.99700) |
| tier1-auc | 0.54808 | +1.03 summed AUC |

Seven clips sit just outside the MP-SR gate: **e29 3.85, e11 3.97, e02 4.57, e21 4.79, e09 4.89,
e13 4.98, e17 5.15 cm**. The same seven serve all three rungs — a centimetre each is ~7 cm summed
against the 1.9 cm MPPE needs, and their AUC runs 0.206–0.627, so there is room there too.

**The warm start cannot collect them, and that is measured, not assumed.** Six of these seven were
fine-tuned on the clean Tier 1 reference and every one came back worse on MPPE
(`results/t1_finetune_adoption.json`, `results/t1_rank_sensitivity.json`):

| clip | shipped MPPE | fine-tuned MPPE | shipped AUC | fine-tuned AUC |
|---|---|---|---|---|
| e29 | 3.85 | 8.72 | 0.627 | 0.602 |
| e11 | 3.97 | 4.44 | 0.468 | 0.388 |
| e02 | 4.57 | 35.05 | 0.426 | 0.370 |
| e21 | 4.79 | 10.34 | 0.206 | 0.223 |
| e09 | 4.89 | 11.66 | 0.300 | 0.224 |
| e17 | 5.15 | 5.29 | 0.447 | 0.328 |
| e13 | — | never tried | 0.159 | — |

That arm trades position accuracy for reference tracking, 12 of 12 rejected clips over.

## The mechanism this arm tests, stated as a falsifiable claim

The warm start failed because **re-fitting a converged actor to a different trajectory is
distribution shift** — our own measurement, the cold swap of the clean reference under the shipped
actors took the entry from 0.465 to 0.289 AUC with only 2 of 30 clips improving. That is a fact
about *re-fitting*, not about the reference.

Training from scratch on the clean reference is a different thing and has never been run on
Tier 1. The claim: these policies currently chase a reference carrying **1.94 cm of error against
truth**, so a policy that tracks its reference perfectly still scores ~1.94 cm of MPPE it did not
earn. Removing that floor is worth roughly the gap these seven clips need.

**If the arm fails**, the honest reading is that the deficit is not reference noise but the policy's
own precision, and no reference change collects it.

## The bar, registered now

- **Per clip:** adopt only under the standing Tier 1 no-regression rule — AUC improves and no other
  board regresses. Legal and truth-equivalent, because Tier 1's permitted input *is* the configured
  scoring reference (organisers' answer 2, `results/organiser_answers.json`), so a better rollout is
  simply kept and a worse one is simply not shipped.
- **Arm-level success:** at least one of the three rungs above crosses — one clip under the MP-SR
  gate, or 1.9 cm of summed MPPE, or +1.03 of summed AUC.
- **Downside is the bill, not the entry.** No clip ships unless it beats what is already uploaded.

## Cost

7 clips, 7 boxes, 1 clip each, 4.4 GPU-h + 0.6 h bring-up at $0.68/h = **$23.80**. Checkpoints are
kit-scored on pull and selected per clip; z-drop selection is unnecessary here because the
permitted input is the scoring reference, but it is still recorded for the provenance trail.

## Not in this arm, and why

**Tier 3 is held.** The organisers' answer 3 forbids the ground-truth mesh for training, and our
Tier 3 chain uses it both as FoundationPose's CAD model and as the simulator's object geometry
(`src/v2d/retarget/sequences.py:270`). A follow-up naming both uses is in the same email thread.
Until it is answered, Tier 3 compute buys an entry that may be withdrawn — including the staged
13-clip warm-start batch, which stays staged.
