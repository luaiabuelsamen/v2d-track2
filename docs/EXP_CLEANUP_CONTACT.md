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

---

# Results — 2026-10-24

All four arms completed 30/30 clips at **0.0 mm** worst rigid-fit residual and ≤0.043° worst
orientation residual, so the pre-registered validation bar is met and the numbers are readable.
Three clips (tilt_fixed e14, rot_clean e00/e05) died silently during mesh loading on the first
pass — the same OOM signature as before — and were re-run at one worker before anything was
scored. Everything below is `results/cleanup_contact.json`.

## Contact coverage

| Arm | Contact | vs Tier 2 | recovery vs `rot_clean` | recovery vs `object_clean` |
|---|---|---|---|---|
| `tier1` (clean human + clean object) | 0.5744 | +0.2043 | — | — |
| `object_clean` (noisy human, clean object) | 0.4967 | +0.1266 | 1.256 | 1.000 |
| `rot_clean` (noisy human, clean orientation) | 0.4709 | +0.1008 | 1.000 | 0.796 |
| **`tilt_fixed`** | **0.4352** | **+0.0651** | **0.646** | 0.514 |
| `yaw_fixed` | 0.3825 | +0.0124 | 0.123 | 0.098 |
| `tier2` (as released) | 0.3701 | — | 0.000 | 0.000 |

## Prediction scorecard — 5 of 6 pass

| | Prediction | Result | |
|---|---|---|---|
| P1 | ceiling below Tier 1, in 0.42–0.52 | `object_clean` = **0.4967** | **pass** |
| P2 | tilt beats yaw: mean ≥ +0.03 **and** sign in ≥20/30 | mean **+0.0527** ✓, sign **19/30** ✗ | **fail** |
| P3 | `recovery_rot(tilt_fixed)` ≥ 0.5 | **0.646** | **pass** |
| P4 | `recovery_rot(yaw_fixed)` ≤ 0.25 | **0.123** | **pass** |
| P5 | ≥5 of the 8 worst-hit clips beat the mean gain | **8 of 8** | **pass** |
| P6 | tilt and yaw bracket the orientation ceiling | 0.370 < 0.383 < 0.435 < 0.471 | **pass** |

**Verdict: PARTIAL**, by the rule fixed in advance. P2 failed by **one episode** on sign
consistency while passing on magnitude, and the rule required both. The rule's three cases did
not cleanly cover this outcome — P2 failing while P3 passes well above its bar — and the scorer
took the conservative branch. I am not re-reading the rule to upgrade this to HOLDS; that is the
move pre-registration exists to prevent.

So: **the tilt cleanup is adopted for its scored-metric gain, and no training-signal benefit is
claimed from this experiment.**

## What the numbers say beyond the scorecard

**Tilt is genuinely the recoverable component, not merely the larger one.** Per degree of
orientation error removed, tilt restores **0.00412** contact and yaw **0.00109** — tilt is
**3.8× more efficient per degree**, against an angle only 1.39× larger (15.80° vs 11.40°). That
asymmetry is the real content of the result, and it is what the support-plane argument predicted.

**A post-hoc explanation for the 11 sign flips was tested and rejected.** The obvious story —
"yaw won on the clips where yaw is the bigger error" — holds at the group level (tilt/yaw error
ratio 1.06 on yaw-won clips vs 1.64 on tilt-won clips, and 7 of 11 have yaw larger) but fails
per clip: contact gain correlates with degrees removed at only **r = +0.107** for tilt and
**+0.178** for the tilt-minus-yaw difference. What actually predicts a clip's gain is how much
contact the full orientation cleanup can restore there at all (**r = +0.848** against
`rot_clean - tier2`). Per-clip recovery is therefore governed by clip-specific geometry, not by
angular magnitude — which is both why the sign flips on a third of the clips and why no per-clip
reliability should be claimed. The honest unit of this result is the mean.

**P1 is the finding with the longest reach.** `object_clean` sits at 0.4967 against Tier 1's
0.5744, so **38 % of the contact loss (0.0777 of 0.2043) survives a perfect object-pose
cleaner** and is attributable to Tier 2's rigid root perturbation of the human. A second cleaner
that re-aligns the perturbed root is the larger remaining win, and it is a different piece of
work from this one.

## The caveat that governs what gets built

`tilt_fixed` is an **oracle**: it is constructed from the clean reference. So is the AUC
0.425 → 0.654 figure from the scoring ladder. **Neither is achievable by a submitted pipeline**,
and both are upper bounds on a deployable support-contact cleaner, not forecasts of it. Adopting
"the cleanup" therefore means building a cleaner that reaches for that bound using only Tier 2
inputs plus the released ground plane — the penetration-aware cleanup — and then measuring how
much of the 0.646 it actually delivers. Any number quoted for the pipeline must come from that
deployable cleaner, never from these arms.

## Follow-up: what the *deployable* cleaner actually restores

The caveat above demanded that any number quoted for the pipeline come from the deployable
cleaner (`scripts/clean_support_contact.py`, which uses no clean reference), so its 30 clips were
retargeted on CPU under the same conditions as the arms — 30/30 passing at 0.0 mm rigid-fit
residual and 0.043° orientation residual (`results/arm_support_cleaned.json`). Contact coverage
is `results/arm_support_cleaned_contact.json`.

| reference | contact | vs Tier 2 | recovery vs `rot_clean` | share of the tilt oracle's gain |
|---|---|---|---|---|
| `tier1` | 0.5744 | +0.2043 | — | — |
| `rot_clean` (oracle) | 0.4709 | +0.1008 | 1.000 | — |
| `tilt_fixed` (oracle) | 0.4352 | +0.0651 | 0.646 | 1.000 |
| **support-contact cleaned (deployable)** | **0.3841** | **+0.0141** | **0.140** | **0.217** |
| `yaw_fixed` (oracle) | 0.3825 | +0.0124 | 0.123 | 0.190 |
| `tier2` (as released) | 0.3701 | — | 0.000 | 0.000 |

**The deployable cleaner does not restore the grasp signal.** +0.0141 mean contact is 14 % of the
orientation-cleanup ceiling and barely above `yaw_fixed`, the arm pre-registered as the control
that should *not* work. Per clip it is not even reliably positive: **15 up, 10 down, 5 untouched**
(sign test p = 0.42), mean +0.0418 where it helps against −0.0206 where it hurts, and the amount
of tilt it removes does not predict the gain (r = −0.03 over the 25 cleaned clips). On the 8
worst-hit clips it recovers +0.0227 of a possible +0.14.

Two checks say this is a real limit of the cleaner rather than a plumbing fault:

- **The 5 clips it declined to touch** (e03, e06, e07, e20, e25 — no floor-level rest segment)
  have contact change of *exactly* 0.0000. Nothing leaked between arms, and the measured effect
  belongs entirely to the 25 clips it did modify.
- **It captures the same fraction on both axes it moves**: 21.7 % of the tilt oracle's contact
  gain and 18.0 % of its AUC gain (0.4247 → 0.4660 of 0.6541). One cleaner doing about a fifth of
  the oracle's work, consistently, is a coherent picture — not a contact-specific failure.

The mechanism is the obvious one: the cleaner only corrects frames where the object rests at
floor level, 38 % of frames on average, and contact is scored during the *manipulated* frames it
propagates into by Slerp with no support evidence of its own.

**This confirms the pre-registered PARTIAL action rather than softening it.** The adopted
justification for cleaning Tier 2 references is the scored-metric gain — AUC 0.4247 → 0.4660,
SP-SR 0.9667 → 1.0000, MPPE 2.3605 → 2.3053 cm, no metric regressing — and explicitly **not** a
training-signal benefit. The honest statement for the Brev gate is that it trains on references
that score better, not on references with a measurably better grasp signal.

The ceiling this leaves on the table is the actionable part: `tilt_fixed` reaches +0.0651 and
`rot_clean` +0.1008, so a cleaner with a support hypothesis for non-resting frames — the 14
table episodes have released `support_surface/` primitives that this cleaner ignores — has 4–7×
the contact headroom of what is built. That, and the root-alignment cleaner P1 points to, are
where the remaining contact loss lives.
