# Pre-registration: the two cleanups the last experiment pointed at

`docs/EXP_CLEANUP_CONTACT.md` closed by naming two places where the contact loss still lives,
and this is the test of both. **Written and committed before either cleaner is implemented**;
the predictions, their thresholds and the decision rule below are fixed and will be applied
mechanically, not re-read after the numbers come in.

The two headroom claims being acted on, both from that experiment:

1. The shipped cleaner corrects only *floor-level* rest, 38 % of frames, because the floor is
   the only support plane whose position is known. The oracle that fixes tilt everywhere reaches
   contact 0.4352 and AUC 0.6541 against the cleaner's 0.3841 / 0.4660 — **4-7x the headroom**.
   The release ships `support_surface/` primitives for the 14 table episodes which the cleaner
   ignores.
2. A *perfect* object-pose cleaner (`object_clean`) still leaves 38 % of the contact loss
   (0.0777 of 0.2043) on the table, attributable to Tier 2's rigid perturbation of the human
   root. Nothing built so far touches it.

## Measured before predicting

Four facts were established first, so the predictions below rest on data rather than hope. All
four are in `results/probe_support_root.json`.

**(a) The primitives are usable without the missing transform.** They live in a Z-up
`g1_retarget_world`; the object poses are in a Y-up frame; no transform between them ships. But a
*height above the floor* is a scalar, invariant to any transform that preserves up and slides
within the floor plane. So a primitive's top-face height is directly comparable to the
lowest-vertex height the cleaner already measures. It checks out: of the **96 non-floor rest
segments** the cleaner currently rejects, **73 sit within 2 cm of a horizontal primitive top**,
median gap **0.65 cm**. Sub-centimetre agreement over 73 segments in 13 episodes is also the
empirical evidence that the two frames share an up axis and that the Z-up frame's floor is at
z = 0 — if either were false the heights would disagree systematically.

**(b) The primitive gate answers the question that forced the floor-only restriction.** The
cleaner refuses non-floor rest because *motionless is not supported* — `watering_can` is held
steady in mid-air while pouring and produced seven bogus "rest" segments at 13-14 cm. It ships
**no primitive at all**, so the gate refuses it for a reason rather than by blanket rule.
`squeegee` rests at 3.9-8.3 cm with primitives at 69.8/71.6 cm — a 62-66 cm gap, refused.
`vacuum` ships only a `chair_back` Cube that is thin in *x*, a panel you lean against rather
than a surface you rest on, and is refused by the thin-axis test.

**(c) The table extension unlocks 5 episodes outright.** Candidate support heights are the floor
plus every horizontal primitive top. Across 30 clips: 350 rest segments, **227 accepted today,
92 newly accepted** (1.41x), 31 still refused. Thirteen episodes change. Critically, the **five
episodes with zero floor-level rest — e03, e06, e07, e20, e25 — are exactly the five the cleaner
leaves untouched today**, and each gains 6-17 segments. That is why a better-than-linear gain is
plausible: those five currently contribute exactly nothing.

**(d) The root perturbation is mostly *not* a constant offset.** Per clip the root rotation error
averages **5.17°** (max 8.00°) and the translation error **1.09 cm**; removing the best constant
rotation leaves **3.16°**, so only **38.9 %** of the rotation error is a fixed per-clip offset
(43.5 % for translation). All 76 non-root joints are bit-identical in every clip, so the body's
shape and articulation are untouched and the noise is purely a rigid, time-varying displacement.
A constant-offset estimator therefore caps out at ~39 % of it, and a per-frame estimator is
needed to do better.

**(e) What the existing pipeline already corrects, so this does not duplicate it.** The
retargeter's `ground_alignment.py` applies a per-frame **Z-translation** that drags the lowest
contact onto a horizontal plane (capped at 1 cm, and the QA shows the cap is never reached), and
propagates the robot's Z-delta to the object during interaction. It applies **no rotation**, and
its plane is horizontal-only. A root *rotation* cleaner is new work.

## The interventions

**Arm `table` — the object-pose cleaner, extended to real support surfaces.** Same pipeline as
`scripts/clean_support_contact.py`, with the floor-only gate replaced by: build candidate support
heights from the floor plus every *horizontal* primitive top (thin axis = z); accept a rest
segment when its measured height is no more than `--support-tolerance-cm` (3, as today) **above**
the nearest candidate and no more than `--support-penetration-cm` (10) below it; use that
candidate's plane for the correction. Penetration is allowed because a tilt error is exactly what
digs a corner through a surface — `traffic_cone` sits 2.3 cm under its table top and
`woven_basket` 4-6 cm under its desk, which is the error being corrected, not a reason to refuse.
The 10 cm bound is safe because the **lowest table top in any episode sits 46.9 cm above the
floor**, so no floor segment can be mistaken for a table segment or vice versa.

> **Correction, made while committing the probe and before any arm ran.** This paragraph first
> claimed candidate heights are "at least 46.9 cm apart in every episode", which is false: e25
> ships two tops 0.36 cm apart. The statistic that carries the safety argument is the
> floor-to-lowest-top gap, which is the 46.9 cm quoted above; two tops that close are the same
> plane and confusing them is harmless. No prediction or threshold changes.

The support normal for a table top is taken to be the **floor normal**. Both frames are
gravity-aligned descriptions of one room and the primitive's own axis is the Z-up frame's up, so
up maps to up; fact (a) is the empirical check. This is an assumption and it is the most likely
way this arm fails.

**Arm `root` — a foot-planting cleaner for the human root.** The same argument as the object
cleaner, applied to the body: a planted foot does not move, so during a stance segment the stance
foot's world pose should be constant, and whatever rigid root correction holds it still is an
estimate of the perturbation. Unlike a constant-offset estimator this is per-frame, so it can
reach the 61 % of the error that fact (d) shows is time-varying. It uses only the noisy SOMA
parameters and the body model — no clean reference. The correction is applied as a rigid
transform of the whole body about the stance foot, written back into the root rotation vector and
the translation, which is exactly the subspace Tier 2 perturbs.

**Arm `table_root` — both.** This is the candidate reference set.

> **Amendment, made on calibration data before any outcome was measured.** The root cleaner's
> stance detector was specified above with an absolute speed threshold, as the object cleaner
> uses. Calibration on paired clips (`results/probe_root_stance.json`) shows that cannot work:
> the median slower-foot speed is 0.0053 m/s in Tier 1 and **0.0330 in Tier 2, a 6.2x noise
> floor**. At 0.02 m/s Tier 1 calls 85 % of frames planted and Tier 2 only 22 %; at 0.05 m/s
> Tier 2 reaches 77 % but Tier 1 is at 97 %; by 0.10 m/s both exceed 99 % and the test says
> nothing. No fixed speed recovers Tier 1's stance set from Tier 2's data. The detector now uses
> **each clip's own 0.4 quantile of that foot's speed**, which asks only which frames are
> quietest — an ordering the perturbation barely disturbs — and needs no absolute scale. 0.4 is
> conservative by more than a factor of two against Tier 1's 85 %.
>
> A cheaper fix was tried first and **rejected by the same measurement**. Since the perturbation
> is rigid it ought to move both feet alike, so the left-right speed *difference* should reject
> it as common mode. The measured correlation is **-0.098, +0.019, +0.035** — nothing is shared,
> because a rotation about the root pushes two feet on opposite sides of it in opposite
> directions.
>
> No prediction or threshold changes. R1-R4 stand exactly as written, and no contact or score
> number for any root arm had been computed when this was made.

> **Second amendment, same conditions: R3 as written is not measurable.** R3 asks for the
> stance-foot **vertical** std. There is no vertical to measure it along. The released ground
> plane is expressed in the object frame while the SOMA body comes out in its own frame — the
> feet sit ~270 cm from that plane — and no transform between them ships, which is the same gap
> that made "stationary" the right constraint in the first place. I wrote R3 before establishing
> that, and it cannot be evaluated as phrased.
>
> It is replaced by **R3′: the stance-foot centroid's 3D scatter within its own stance segment
> falls ≥ 30 % in ≥ 24/30 clips.** Same bars, same direction, and strictly harder — 3D scatter
> contains the vertical component and two more. R3 is recorded as unevaluable rather than
> quietly reinterpreted, and no outcome number existed when this was decided either.

## Predictions

Numbers to beat, all already committed: raw Tier 2 scores AUC **0.4247** / SP-SR **0.9667** /
MP-SR **1.0000** / MPPE **2.3605** cm with contact **0.3701**; the shipped floor-only cleaner
scores **0.4660 / 1.0000 / 1.0000 / 2.3053** with contact **0.3841**; the tilt oracle reaches
**0.6541** and contact **0.4352**; `object_clean` reaches contact **0.4967** and Tier 1
**0.5744**.

| | Prediction | Threshold |
|---|---|---|
| T1 | the table extension lifts AUC meaningfully above floor-only | AUC ≥ **0.4860** (point prediction 0.50, range 0.48-0.53) |
| T2 | and costs nothing elsewhere, because rank-sum scoring weights every metric equally | SP-SR ≥ 1.0000, MP-SR ≥ 1.0000, MPPE ≤ 2.3053 cm |
| T3 | every episode now has a usable support hypothesis | 30/30 cleaned, 0 left unchanged |
| T4 | the five newly unlocked clips are where the gain comes from | per-episode AUC improves by ≥ 0.01 on **each** of e03, e06, e07, e20, e25 |
| T5 | the gate does not become permissive | e24 `watering_can` and e28 `squeegee` gain **no** newly corrected segments |
| T6 | and the grasp signal improves too | contact ≥ **0.3950** (point prediction 0.41) |
| R1 | the root cleaner cannot move the scorer, since all four metrics are functions of the object pose alone | scores of `root` bit-identical to raw Tier 2, and `table_root` to `table` |
| R2 | foot planting recovers part of the root-attributable contact loss | contact(`table_root`) − contact(`table`) ≥ **+0.010** paired (ceiling +0.0777; point prediction +0.020) |
| R3 | and it does what it claims mechanically, whether or not contact follows | stance-foot vertical std falls ≥ 30 % in ≥ 24/30 clips |
| R4 | neither cleaner breaks the pipeline | 30/30 retargets pass at ≤ 1.0 mm rigid fit and ≤ 1.0° orientation, in every arm |

R1 is a plumbing check, not a discovery: the four metric files compute MPPE keypoints as
object-frame offsets rotated by the object rotation, so nothing about the human enters any
metric. If R1 fails, something is wired wrong and the other numbers are void.

## Decision rule, fixed in advance

**Which object-pose cleaner becomes the Tier-2 reference set:** the variant with the highest AUC
among {raw Tier 2, floor-only, table-extended} **subject to no metric being worse than raw Tier
2's**. If two are within 0.005 AUC, the simpler one wins. This is decided on the local scorer
alone, because the scored ceiling is what a submission is graded on.

**Whether the root alignment ships in that reference set:** include it if and only if R1, R2 and
R4 all hold. If R2 fails, the root alignment is excluded, the code and the measurement are kept,
and no benefit is claimed — the same discipline the last experiment ended on. If R3 fails while
R2 passes, it is included on the strength of the outcome measure but recorded as a mechanism we
did not confirm.

**If T1 fails but T2 holds and contact improves (T6):** the table extension is kept for the
training signal only and the floor-only cleaner stays the scored reference set. The two decisions
are independent because the scored artifact (the object npz) and the training artifact (the
retargeted motion) are different files.

## Caveats

- The tilt oracle (0.6541) and `tilt_fixed` (0.4352) are built from the clean reference and are
  **upper bounds, not forecasts**. Every number quoted for the pipeline comes from an arm that
  sees only Tier 2 inputs, the object mesh, the released ground plane and the released support
  primitives.
- The support primitives are released data available to any entrant for all tiers, so using them
  is not an oracle. The *clean trajectory* is what may not be used.
- Contact coverage is measured on 30 clips with a 2 cm threshold and its per-clip sign was shown
  last time to be unreliable (15 up / 10 down at p = 0.42 for the floor-only cleaner). The mean
  is the unit; a paired mean gain is the only contact claim these arms can support.
- Episode 29 manipulates its object with a foot and has ~0 hand contact in every arm by
  construction. It stays in the mean and carries no conclusion.

---

# Results

## The support-surface extension, scored

`scripts/clean_support_contact.py --support-primitives`, scored with the kit's own metric code
(`results/tier2_support_table.json`). The floor-only path was first confirmed **bit-identical**
to the committed baseline — all 30 npz files compare equal and the per-episode record matches —
so this is strictly an addition.

| reference | AUC | SP-SR | MP-SR | MPPE cm |
|---|---|---|---|---|
| Tier 2 released | 0.4247 | 0.9667 | 1.0000 | 2.3605 |
| floor-only cleaner | 0.4660 | 1.0000 | 1.0000 | 2.3053 |
| **support-surface cleaner** | **0.4755** | **1.0000** | **1.0000** | **2.2777** |
| tilt oracle (unreachable) | 0.6541 | 1.0000 | 1.0000 | 2.0672 |
| clean reference | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

| | Prediction | Result | |
|---|---|---|---|
| T1 | AUC ≥ 0.4860 | **0.4755** (+0.0095 over floor-only) | **fail** |
| T2 | no metric regresses | SP-SR 1.0000, MP-SR 1.0000, MPPE **2.2777** | **pass** |
| T3 | 30/30 cleaned | **30/30**, 0 unchanged | **pass** |
| T4 | each of e03/e06/e07/e20/e25 gains ≥ 0.01 AUC | +0.1459, +0.0245, +0.0571, **+0.0021**, +0.0328 | **fail (4/5)** |
| T5 | e24 and e28 gain no corrected segments | 254→254 and 76→76 frames, floor only | **pass** |

**T1 failed on magnitude and the decision rule still selects this variant.** Those are separate
questions and were written as separate things in advance. 41 % more accepted rest segments bought
23 % of the gain the floor segments did, so my point prediction of 0.50 was wrong and the
mechanism is not linear in segment count. But the rule — highest AUC subject to no metric worse
than raw Tier 2, simpler wins inside 0.005 — picks the support-surface variant on 0.4755 vs
0.4660, a 0.0095 gap. **It becomes the Tier 2 reference set.**

## Where the gain and the loss actually are

Per episode: **11 improve, 18 are untouched, 1 regresses.** Mean ΔAUC is **+0.0525** on the five
newly unlocked clips and **+0.0009** on the other 25, so essentially the whole effect lives in
the episodes that previously received no correction at all. That also explains T1: 18 clips could
not move, because they have no primitive or no non-floor rest.

**e17 `traffic_cone` regresses −0.0815**, the largest per-clip move in either direction. I
guessed the cause was the propagation interpolating between a floor-derived and a table-derived
correction, and that guess is **wrong**. Splitting the clip in thirds against the clean reference
(analysis only — the cleaner never sees it) puts the damage squarely in the table phase, and it
grows monotonically into it:

| | first third | middle | last third |
|---|---|---|---|
| floor-only orientation error | 22.50° | 19.20° | 19.41° |
| support-surface | 22.57° | 23.03° | **28.57°** |
| delta | +0.07° | +3.83° | **+9.16°** |

The first third is untouched, so the interpolation is not the problem — the table corrections
themselves are. The likely mechanism is the penetration allowance doing exactly what it was
designed to do, in a case where it should not: the cone's "rest" segments sit 2.3 cm *below* the
77.1 cm table top, which the 10 cm window accepts as tilt-driven penetration, but a cone held
briefly in the air beside the table while being set down looks identical by that test. This is
the `watering_can` failure re-admitted through the window that tables required. It is **not**
being tuned away here: a gate change chosen by looking at e17's score is a gate fitted to e17.
It is the obvious subject of the next pre-registration.

For contrast, where the extension works it works decisively. **e03 `contigo_coffee_mug`** goes
from 25.65° to 5.85° mean orientation error, **771 of 781 frames better and none worse**,
ΔAUC +0.1459. **e20 `rice_cooker`** — the clip that failed T4 — is simply near-neutral: −0.65°
overall with 211 frames worse and 279 better, a correction that neither helps nor hurts rather
than one that failed.
