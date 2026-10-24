# Pre-registration — does cleaning the Tier 2 object orientation restore the grasp signal?

Written 2026-10-24, **before any of the arms below were run.** Predictions are committed in this
file first so the result cannot be read back onto them afterwards.

## Why this experiment

Tier 2's injected noise costs 36 % of the hand-object contact coverage in the retargeted
references: paired across all 30 episodes, mean coverage falls **0.5744 → 0.3701**, 26 of 30
clips degrade and 8 lose more than half (`results/retarget_qa.json`). Contact is what the G1
recipe rewards, so the policy is currently being trained to track an object its hands
geometrically miss.

Separately, the scoring ladder says the Tier 2 orientation error (21.4°) splits into 15.8° of
support-recoverable **tilt** and 11.4° of unrecoverable **yaw**, and that fixing only the tilt
would lift the input ceiling from AUC 0.425 to 0.654 (`results/tier2_tilt_fixed.json`).

Those are two separate claims resting on two separate measurements. This experiment tests
whether they are the same claim: **if tilt is the recoverable part, cleaning it should restore
the contact the noise destroyed, and cleaning yaw should not.**

## What Tier 2 actually perturbs — measured first, and it constrains the answer

Comparing the tiers column by column across all 30 episodes:

| Column | max abs difference | nature |
|---|---|---|
| `observation.object.pose` | 1.999 | position + orientation noise |
| `observation.human.pose` | root joint only: mean **5.21°**, max 8.00° | rigid root rotation |
| `observation.human.translation` | mean **1.09 cm**, max 1.60 cm | rigid root translation |
| `observation.human.{identity_coeffs,scale_params,bone_length_flexibles}` | 0.0 | untouched |
| all 76 non-root joints | 0.0 | **bit-identical** |

So Tier 2 noise on the human is *exactly* a rigid perturbation of the whole body — root
rotation plus root translation, with the articulation untouched. The hands therefore sit in the
same place relative to the torso but are rigidly displaced in world space, on the order of
**~6.4 cm from the root rotation at a 0.7 m lever, plus ~1.1 cm of translation**, against the
retargeter's **2 cm** contact threshold.

**This was not known when the prediction was first stated, and it weakens it.** An object-only
cleaner cannot undo a displacement of the hands. So "tilt restores most of the lost contact"
cannot be tested against Tier 1; it has to be tested against what *any* object-pose cleaner
could achieve. That requires a ceiling arm, added below.

## Arms

All arms retarget on CPU through the same `soma_to_g1.py` at commit
`8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef`, 30 episodes each, and all four non-baseline arms
keep **Tier 2's noisy human parameters**, because that is what a real pipeline receives. Only
the object trajectory differs.

| Arm | Human | Object trajectory | Role |
|---|---|---|---|
| `tier1` | clean | clean | upper reference (already run) |
| `tier2` | noisy | noisy, as released | baseline (already run) |
| `object_clean` | noisy | **Tier 1's clean object pose** | **ceiling**: the most any object-pose cleaner can buy |
| `rot_clean` | noisy | Tier 2 position, clean orientation | **ceiling for an orientation-only cleaner** |
| `tilt_fixed` | noisy | Tier 2 with tilt replaced by clean tilt | the cleanup under test |
| `yaw_fixed` | noisy | Tier 2 with yaw replaced by clean yaw | the control that should *not* work |

`tilt_fixed` and `yaw_fixed` are **oracles**: they are built using the clean reference, so they
are not deployable. They bound what a real support-contact cleaner could reach. A submitted
pipeline may not use them, and nothing here implies otherwise.

`object_clean` is the arm that makes the result interpretable. Without it, a weak tilt result is
ambiguous between "tilt is not the recoverable part" and "the human root noise dominates".

**Amendment, made before any arm was run.** `rot_clean` was added after the plumbing check
showed the flaw it fixes. `tilt_fixed` and `yaw_fixed` keep Tier 2's **positions** by
construction, but `object_clean` also removes the 1.94 cm position noise. Recovery measured
against `object_clean` is therefore depressed for an orientation-only cleaner no matter how well
tilt works, which could have made a real effect read as "partial". `rot_clean` is Tier 2's
position with the clean orientation outright, so it is the honest denominator for tilt and yaw,
and `tilt_fixed`/`yaw_fixed` decompose the path to it. Both recovery figures are reported; the
decision rule below now reads against `rot_clean`, and the `object_clean` figure is reported
alongside so the position noise's share is visible rather than hidden in a denominator.

## Metric

Per clip, hand-object contact coverage = the larger of the two
`{left,right}_hand_contact_active: n/T` fractions the retargeter logs at its 2 cm threshold,
exactly as `scripts/retarget_qa.py` already parses it. Aggregate = mean over the 30 episodes,
paired by episode index. Episode 29 (`rolling_luggage_left_foot_push`) manipulates the object
with a foot and has ~0 hand contact in every arm by construction; it is kept in the mean but
reported separately so it cannot carry a conclusion.

Recovery fraction for an arm, the quantity the predictions are stated in:

    recovery_rot    = (C_arm - C_tier2) / (C_rot_clean - C_tier2)      # orientation-only ceiling
    recovery_object = (C_arm - C_tier2) / (C_object_clean - C_tier2)   # whole-object ceiling

0 means the arm restored nothing; 1 means it reached that ceiling. `recovery_rot` is the figure
the decision rule uses, because tilt and yaw only ever touch orientation.

## Predictions

Stated before running, in the order they will be checked.

**P1 — the ceiling is below Tier 1.** `C_object_clean < C_tier1 = 0.5744`, because the human
root noise alone costs contact. Point prediction **0.42–0.52**, i.e. the human perturbation
accounts for 25–60 % of the total 0.2043 loss.

**P2 — tilt beats yaw.** `C_tilt_fixed > C_yaw_fixed`, mean paired difference **at least
+0.03**, with the sign consistent in **at least 20 of 30** episodes.

**P3 — tilt recovers most of what is recoverable.** `recovery_rot(tilt_fixed) >= 0.5`.

**P4 — yaw recovers little.** `recovery_rot(yaw_fixed) <= 0.25`.

**P6 — the two arms bracket the orientation ceiling.** `C_tilt_fixed` and `C_yaw_fixed` both lie
between `C_tier2` and `C_rot_clean`, since each fixes one component of the error `rot_clean`
removes entirely. A tilt or yaw arm *above* `rot_clean` would mean the contact metric is not
monotone in orientation error and the whole framing is wrong.

**P5 — the 8 worst clips move most.** Of the 8 clips that lost over half their contact
(e00, e01, e14, e15, e18, e19, e22, e28), at least 5 gain more contact under `tilt_fixed` than
the 30-clip mean gain.

## Decision rule, fixed in advance

- **Holds** — P2 and P3 both pass. The tilt cleanup goes into the Tier 2 pipeline and the Brev
  gate trains on cleaned references.
- **Fails** — `recovery_rot(tilt_fixed) < 0.25`. The orientation cleanup does not help the grasp
  signal. It stays in for its scored-metric gain only (AUC 0.425 → 0.654 is already measured
  and is not affected by this experiment either way), and I say so plainly rather than
  implying the two results support each other.
- **Partial** — recovery between 0.25 and 0.5, or P2 passes while P3 fails. Report the number,
  build the cleanup in for the scoring gain, and do not claim a training-signal benefit.

Whatever happens, P1's answer is reported, because a ceiling well below Tier 1 means the
remaining contact loss is the human root noise and a *second* cleaner — aligning the perturbed
root — would be the thing worth building next.

## Validation

Every retargeted clip in every arm is checked by `scripts/validate_retarget.py` against **that
arm's own** object trajectory, not against the released tier. An arm's result is only read once
all 30 of its clips pass the 1.0 mm rigid-fit and 1.0° orientation bars, so a contact change
can never be an artefact of a retarget that silently diverged.

## Cost

CPU only, $0, no GPU (`~/projects/_ops/GPU_RELEASED` untouched), no Brev. 120 clips at ~3.5 min
each with 2 workers ≈ 4 h wall clock. 2 workers is the ceiling on this 15.6 GB box; 3 lost 28
clips to OOM kills.
