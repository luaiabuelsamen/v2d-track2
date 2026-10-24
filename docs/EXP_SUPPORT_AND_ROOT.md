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

## The root cleaner fails at its own task

Before any contact arm ran, the obvious direct question: does the foot-planting cleaner actually
remove Tier 2's root perturbation? It is cheap to ask — compare the cleaned root against the
Tier 1 root, parquets and npz only, no body model — and the answer is no
(`results/root_perturbation_removed.json`; analysis against the clean reference, which the
cleaner never sees).

| | before | after | change | clips improved |
|---|---|---|---|---|
| root rotation error | 5.174° | **5.584°** | **−7.9 %** | **3 / 30** |
| root translation error | 1.087 cm | **2.102 cm** | **−93.3 %** | **0 / 30** |

It makes the thing it was built to fix **worse**, and the translation nearly twice as bad on
every single clip. The cleaner runs on all 30 (mean correction 1.30°, mean body shift 1.67 cm,
34 % of frames in stance), so this is not a coverage failure. It is the estimator being wrong.

**Why, and it is the interesting part.** The constraint "a planted foot does not move" is a
statement about *relative* motion, and its reference pose is the **median of the noisy foot
positions over the segment**. That median is itself displaced by the mean perturbation across
that segment, so pinning the foot to it removes within-segment wobble while preserving — and
redistributing into the root — the segment-mean error. Worse, where the detector calls stance on
frames the foot genuinely moves through, pinning injects an error that was not there. The
translation damage exceeding the rotation damage fits exactly: the Kabsch fit absorbs genuine
foot travel into the root translation.

The object cleaner does not have this problem, and the contrast is the lesson. **Its reference is
external to the noisy data** — the floor's normal and a support surface's height are geometric
facts about the room, independent of the trajectory being cleaned. The root cleaner has no
external reference at all, only an internal consistency condition, and an internal condition
cannot recover an absolute placement error. The one external reference the body *has* is the
ground plane, and it is expressed in a frame with no published transform to the body's own — the
gap that drove this design in the first place.

So: **a cleaner needs an external reference, not merely a self-consistency condition.** That is
the transferable result, and it says a root cleaner is a worse-posed problem than it looked, not
that this implementation needs tuning.

**The pre-registered contact arm was run anyway.** R2 asks for the paired contact difference
between `table_root` and `table`, and declining to run a committed test because I expect it to
fail is precisely the bias pre-registration exists to prevent. It is also not a foregone
conclusion: contact depends on hand-to-object geometry, and a body further from the clean
reference overall could still happen to place the hands better.

## T6: the scored metrics and the grasp signal come apart

The table arm retargeted 30/30 at 0.0 mm rigid-fit residual and 0.033° orientation
(`results/arm_table.json`), so its contact number is readable.

| reference | contact | vs Tier 2 | AUC |
|---|---|---|---|
| tier1 | 0.5744 | +0.2043 | 1.0000 |
| floor-only cleaner | 0.3841 | +0.0140 | 0.4660 |
| **support-surface cleaner** | **0.3817** | **+0.0116** | **0.4755** |
| tier2 released | 0.3701 | — | 0.4247 |

**T6 fails: 0.3817 against a 0.3950 bar.** It fails because contact did not move, not because it
fell. Paired against the floor-only cleaner the difference is **−0.0024 with 6 clips up, 5 down
and 19 tied, two-sided sign test p = 1.000** — indistinguishable from zero on any reading. So
the support-surface extension **buys a clear scored gain and leaves the grasp signal
untouched**: AUC +0.0095 and MPPE −0.028 cm, contact unchanged.

The dissociation is the finding. Over the 12 episodes the extension actually changed, per-clip
AUC gain and contact change do correlate (**r = +0.481**), but the mean AUC gain is **+0.0237**
while the mean contact change is **−0.0060**. Improving the object's absolute pose is not the
same as closing the hand-to-object gap, and the reason is sitting in the previous section: the
retargeted hands follow the *human*, whose root is still displaced by a perturbation that moves
the hands ~7 cm against a 2 cm contact threshold, and which the root cleaner failed to remove. A
correction that moves the object closer to its true pose can move it *further* from where the
perturbed hands actually are. `e09 big_red_bowl` is the clearest case: AUC +0.0218, contact
−0.037.

This is also the quantitative answer to why both object cleaners capture so little of the tilt
oracle's contact gain. The oracle fixes the object's orientation while the hands stay wrong;
most of the remaining contact loss is on the human side, and `object_clean` already measured that
at **38 % of the total** (0.0777 of 0.2043). Nothing that touches only the object can reach it.

---

# Correction: the root cleaner was built on a false premise

`interp` found a frame bug (their commit `c46d27a`, `results/body_floor_frame.json`) and it
invalidates the reasoning behind the root cleaner. Verified independently here before accepting
it: with the retargeter's own `diag(1,-1,-1)` applied, the lowest foot vertex sits **−1.70,
−1.39 and −0.26 cm** from the released ground plane on episodes 0, 7 and 24. Without it, 260–271
cm. `det = +1`, so it is a proper rotation — 180° about x — not a reflection.

**So the body, the floor plane and the object do share a frame, and I said they did not.** Worse,
I had the transform written down in my own converter's docstring — "`poses.npy` is consumed as
`einsum("ij,tjk->tik", diag(1,-1,-1,1), poses)`" — and read the ~270 cm figure as evidence of an
unreachable frame instead of as evidence of an unapplied flip I had already documented.

Three things follow, and the first two are retractions:

1. **The root cleaner's design rationale was wrong.** It was built on "stationary" because that
   constraint is frame-independent and the ground plane was thought unreachable. The plane was
   reachable the whole time.
2. **The R3 amendment was wrong.** I declared R3 (stance-foot *vertical* std) unevaluable because
   "there is no vertical to measure along". There is: the released plane normal, after the flip.
   R3′ (3D scatter) was a reasonable substitute but it was not necessary.
3. **The lesson survives, but its corollary does not.** "A cleaner needs an external reference,
   not merely a self-consistency condition" still explains why the foot-planting cleaner failed.
   What was wrong is the claim that *no* external reference was available for the body. One was,
   which makes the failure **fixable rather than intrinsic**.

## Re-running "with the flip" is literally a no-op, so this does something else

The existing cleaner Kabsch-fits stance-foot vertices onto their own segment median. That is
invariant under any rigid transform of the input, and `diag(1,-1,-1)` is a proper rotation, so
applying it changes the output by exactly nothing. Checked rather than asserted, below.

The substantive version is the one the failure analysis asks for: **anchor the feet to the floor
plane** instead of to a statistic of the noisy data. A floor pins three of six degrees of
freedom — height, and the two tilt angles — and leaves yaw and horizontal position free, which
is an honest accounting of what a floor can tell you.

## Pre-registration: the floor-anchored root cleaner

Measured against the clean Tier 1 root, which is used for scoring only. Raw Tier 2 is **5.174°**
of root rotation error and **1.087 cm** of translation error; the foot-planting cleaner made both
worse, at 5.584° and 2.102 cm.

| | Prediction | Threshold |
|---|---|---|
| F1 | anchoring to the floor removes the tilt part of the root error, leaving yaw | root rotation error ≤ **4.50°** (point prediction 3.0–4.0) |
| F2 | and the vertical part of the translation error | root translation error ≤ **1.00 cm** (point prediction 0.7–0.9) |
| F3 | and it helps on most clips, not just on average | rotation error improves on ≥ **20/30** |
| F4 | do-no-harm: run on already-clean Tier 1 input it barely moves the root | induced rotation ≤ **2.0°** mean |

**Decision.** If F1 and F2 both hold, the floor-anchored cleaner replaces foot-planting and goes
to a contact arm. If either fails, the root is left alone and the 38 % of contact loss attributed
to it stays unaddressed — reported, not papered over. F3 and F4 failing while F1 and F2 pass
would mean a cleaner that helps on average by a mechanism I cannot localise, which is the same
situation the object cleaner's gate just caught, and it would not ship.

Yaw is explicitly **not** a target. A floor cannot constrain rotation about its own normal, so a
residual of roughly the yaw component is the expected floor of this method, not a shortfall.

## R2, finally measured: the root cleaner significantly harms contact

The `table_root` arm completed 30/30 at 0.0 mm rigid-fit residual (`results/arm_table_root.json`),
so the paired contact comparison R2 asked for is now available
(`results/contact_table_root.json`).

| arm | contact | paired vs `table` |
|---|---|---|
| `tier1` | 0.5744 | — |
| `support_cleaned` (floor-only) | 0.3841 | +0.0024, 5 up / 6 down / 19 tied, p = 1.000 |
| `table` | 0.3817 | — |
| `tier2` released | 0.3701 | — |
| **`table_root`** | **0.3664** | **−0.0153, 9 up / 21 down / 0 tied, p = 0.0428** |

| | Prediction | Result | |
|---|---|---|---|
| R1 | scores cannot move, the object pose is untouched | `object_source` byte-identical between arms | **holds by construction** |
| R2 | paired contact gain ≥ +0.010 | **−0.0153, p = 0.0428** | **fail** |
| R3′ | 3D stance scatter falls ≥30 % on ≥24/30 | −0.256 mean, 0/30 clips | **fail** |
| R4 | 30/30 retargets pass at ≤1.0 mm / ≤1.0° | **30/30 at 0.0 mm / 0.033°** | **pass** |

This is the first contact result in the whole project that is **statistically significant in the
harmful direction**: 21 of 30 clips lose contact, none tie, p = 0.0428. Every other contact
finding has been a wash. It lines up exactly with the upstream measurement — the cleaner raises
root rotation error 5.174° → 5.584° and nearly doubles translation error — so the chain is
coherent end to end: a worse root means worse hands, and worse hands means less contact.

**Decision, as pre-registered: the root alignment is excluded.** R2 was the gate and it failed in
the wrong direction. The 38 % of contact loss attributable to the root stays unaddressed, and is
reported rather than papered over.
