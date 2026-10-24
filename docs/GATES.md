# Cleaner gates

`~/projects/_ops/briefs/v2d_allhands.md` requires every cleaner to pass three gates in order,
with the bar pre-registered before running:

1. **known-answer** — recover noise we injected ourselves
2. **do-no-harm** — clean input stays put
3. **local score on real Tier 2**

Gate 3 was already in place (`v2d score`, every row of the README's ceiling table). Gates 1 and 2
are below. **Written and committed before `v2d gate-cleaner` was run once.**

## What the gates inject and measure

A single **tilt** per clip: a 12° rotation about an axis lying in the support plane, sign chosen
per clip from a seed so a cleaner cannot pass by always tilting one way. Tilt specifically,
because that is the component the support-contact argument claims to recover; a rotation about
the support normal is yaw, which no support argument can recover and which the cleaner is built
not to touch. Recovery is measured on **rest frames only**, and only the tilt component of the
residual is counted.

The injected noise is ours and comes from a seed. The clean Tier 1 trajectory is used to build
the test and to measure recovery — never as an input or a target for anything that runs on
Tier 2, which is the standing rule.

## Pre-registered bars

| gate | measure | bar | prediction |
|---|---|---|---|
| G1 known-answer | mean recovered fraction of the injected tilt, over clips with rest frames | **≥ 0.70** | 0.85–0.98 |
| G1b | worst single clip's recovered fraction | **≥ 0.20** | some clips will be poor; a facet mix-up on a round object is the likely cause |
| G2 do-no-harm, orientation | mean tilt change when run on already-clean input | **≤ 3.0°** | 1–2°, because real clean data is not perfectly flush against its support |
| G2b do-no-harm, position | worst position drift on clean input | **≤ 1e-6 cm** | exactly 0: with `--seat-vertical` off the cleaner writes orientation only, so anything else is a bug |

G1's bar is high on purpose. The injected tilt is constant across the clip and is exactly the
error the cleaner is built for, so if it cannot take most of that back out on rest frames the
mechanism does not work and the AUC gain would need another explanation.

G2b is the gate that matters most for trust. A cleaner that improves the mean by damaging
already-good clips is worse than no cleaner, and this is the cheapest possible check against the
failure mode the root cleaner actually had.

## Decision rule, fixed in advance

If G1 or G2 fails, the support-surface cleaner does **not** stay in the pipeline on the strength
of its scored gain, and the ceiling table's 0.4755 is reported as unexplained pending a
diagnosis. A scored improvement from a cleaner that cannot recover known noise, or that damages
clean input, is not a result worth shipping.

---

# Results: both cleaners fail the known-answer gate

| | G1 mean recovery (≥0.70) | G1b worst clip (≥0.20) | G2 do-no-harm tilt (≤3.0°) | G2b drift (≤1e-6 cm) | verdict |
|---|---|---|---|---|---|
| floor-only | **0.2137** | **−0.2758** | 1.4264° | **0.0000** | **FAIL** |
| support-surface (shipped) | **0.2683** | **−0.2758** | 1.7397° | **0.0000** | **FAIL** |

My prediction was 0.85–0.98. The answer is **0.21–0.27**, and nine of thirty clips come out
*worse* than the tilt we injected. G2 passes on both, exactly as predicted: 1.4–1.7° of movement
on clean input, and position drift of precisely zero because with `--seat-vertical` off the
cleaner writes orientation only.

So the pre-registered consequence applies: **the support-surface cleaner does not stay in the
pipeline on the strength of its scored gain, and AUC 0.4755 is reported as part-explained.** Both
variants fail, and the table extension is slightly *better* than floor-only (0.2683 vs 0.2137),
so the defect is in the mechanism they share, not in the extension.

## The diagnosis

Recovery is not noise — it is a clean spectrum that tracks object shape.

| recovers well (≥0.7) | recovers negatively (<0) |
|---|---|
| blue_easter_basket 0.905, watering_can 0.870, cyan_water_bottle 0.806, spray_bottle 0.738, big_red_bowl 0.723 | squeegee −0.276, rolling_luggage −0.165, toy_stroller −0.141, round_table −0.114, skinny_wood_chair −0.112, iron −0.100, step_ladder −0.069, baseball_bat −0.049, woven_basket −0.021 |

Everything in the left column has **one unambiguous flat base**. Everything in the right column
does not: a squeegee, a cane, a stroller, a bat, a chair. The cleaner picks its support facet as
*the convex-hull facet whose outward normal best agrees with the object's downward direction*,
summed over the segment. When no facet dominates, that argmax is close to arbitrary — and after
a 12° tilt a **different** facet wins, so the cleaner confidently flattens the wrong face. The
`--max-tilt-deg 45` cap does not catch it, because a wrong facet on a blocky object is often
only a few degrees away.

This is also the explanation for the e17 `traffic_cone` regression recorded in
`docs/EXP_SUPPORT_AND_ROOT.md`, which I had attributed to the penetration allowance. A cone's
hull is a fan of triangles with no flat base at all, so it is the single worst case for facet
selection. The penetration-allowance story was a guess; this is measured.

## What the AUC gain actually is

Splitting the 30 clips by their gate result and measuring AUC against **raw Tier 2**:

| | clips | mean AUC gain |
|---|---|---|
| gate-passing (recovery ≥ 0.5) | 7 | **+0.0755** |
| gate-failing (recovery < 0) | 9 | **+0.0261** |

Correlation between per-clip recovery and per-clip AUC gain is **+0.219** against raw Tier 2 and
**+0.495** against absolute AUC. So the mechanism is real where the facet is unambiguous and
earns about three times as much there — but the gate-failing clips still gain, and the
support-contact argument does not explain why they do. Plausibly the correction is a partial
regression toward a more typical orientation; that is a hypothesis, not a measurement, and it is
flagged as such.

## What this changes

- The README's ceiling table stands as a **measurement** of what the cleaned reference scores. It
  no longer carries the claim that the support-contact mechanism is what produces it.
- Per the brief's rule to choose on synthetic validation rather than on a Tier-1-scored number,
  the next step is a cleaner gated **per clip** on synthetic recovery: apply the correction only
  where injected noise is demonstrably recovered, and leave the other clips untouched. That
  decision uses no Tier 1 information at all. Pre-registered below before running.
- The facet-selection rule itself is the thing to replace, and the obvious candidate is to
  require a *dominant* facet — one whose agreement with "down" beats the runner-up by a margin,
  or whose area is a large fraction of the hull — rather than taking an argmax that may be a
  coin flip.

## Pre-registration: the per-clip gated cleaner

**Rule.** Run the known-answer gate per clip. Apply the support-surface correction only to clips
whose recovered fraction is ≥ **0.50**; pass every other clip through unchanged. The threshold is
chosen from the gate's own distribution, which is synthetic, and no Tier 2 score is consulted in
setting it.

| | Prediction | Threshold |
|---|---|---|
| H1 | the gated cleaner's gate score is high by construction, since failing clips are excluded | mean recovery over *corrected* clips ≥ 0.70 |
| H2 | it does not regress the scored metrics against raw Tier 2 | AUC ≥ 0.4247, SP-SR ≥ 0.9667, MP-SR ≥ 1.0000, MPPE ≤ 2.3605 |
| H3 | it scores below the ungated cleaner, because it gives up the unexplained gain on failing clips | AUC between **0.44 and 0.4755** |
| H4 | and the gain it keeps is the mechanism's | AUC ≥ raw + 7 × 0.0755 / 30 = 0.4247 + 0.0176 = **0.4423** |

**Decision.** Ship the gated cleaner if H1 and H2 hold, even if its AUC is below 0.4755, because
a smaller gain that the mechanism explains is worth more than a larger one it does not. If H2
fails, ship no cleaner and report the raw Tier 2 ceiling.
