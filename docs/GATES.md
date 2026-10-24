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
