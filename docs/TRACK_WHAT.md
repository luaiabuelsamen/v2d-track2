# Can the clean object pose be recovered from the clean hands? No — 0 of 29 clips

`v2d noise-channels --parts b`, written long ago and never run until 2026-10-29 18:55 UTC.
Record: `results/noise_channels_part_b.json`.

## Why the question mattered

Tier 2's noise is not spread evenly. The object's orientation is corrupted by **21.4°** (15.8° tilt
+ 11.4° yaw) and its position by 1.94 cm, while the human pose is corrupted on the **root joint
only** (5.17°) with 1.09 cm of translation — so the hands arrive essentially clean
(`results/probe_support_root.json`).

That sets up an obvious idea. If a grasped object were rigid in the palm frame, its orientation
during held frames could be *re-derived* from the clean hand pose instead of trusted from the noisy
channel. It would be legal — the cleaner may read Tier 2 inputs, the object mesh, the released
ground plane and the released support primitives — and it would be free, because it is CPU
preprocessing of references rather than training.

It would also be the only idea on the table that could beat our own **reference ceiling**, which is
what actually binds Tier 2:

| reference scored against truth | AUC |
|---|---|
| Tier 2 as released | 0.4247 |
| our support-surface cleaner | **0.4755** |
| our submitted entry | 0.46522 |
| **1st place on the board** | **0.56768** |
| tilt oracle *(uses truth, not submittable)* | 0.6541 |

Our policy is already tracking the cleaned reference about as well as that reference permits. The
leader sits *above* our cleaner's ceiling, so they are recovering orientation we are not.

## The measurement

For every clip, on each hand's own contact frames, the object pose expressed in that palm's frame:
position spread is the root of the summed per-axis variance, orientation spread is the mean
geodesic angle from the mean rotation. Rigid means **< 2 cm and < 10°**.

| group | clips | rigid | median position spread | median orientation spread |
|---|---|---|---|---|
| one-handed | 11 | **0** | 7.01 cm | 21.6° |
| two-handed | 18 | **0** | 8.13 cm | 56.5° |
| all measured | 29 | **0** | — | — |
| *restricted to a single contact segment* | 29 | **0** | — | — |

One clip had too few held frames to measure.

## What it means

**The hand is not an anchor for the object's orientation, on any clip.** The median one-handed
orientation spread of 21.6° is the same size as the Tier 2 orientation noise itself (21.4°), so
re-deriving orientation from the palm would substitute one 21° error for another. Restricting to a
single contact segment does not rescue it, so this is not an artefact of pooling across regrasps —
the object genuinely moves in the hand.

This confirms and generalises the earlier spot measurement (a grasping hand wanders 4.9 cm and 44°)
from a sample to all 30 clips, with the within-segment control attached.

It also explains the shape of our cleaner's ceiling. Resting frames can be corrected, because the
support surface pins the object's tilt and that is released data. **Held frames cannot be corrected
from anything we are allowed to read** — not from the hand, and the object's own orientation channel
is the corrupted one. That is why the cleaner stops at 0.4755 against a 0.6541 tilt oracle.

So whatever the leader at 0.56768 is doing, it is not this. The remaining candidates are a learned
prior over object trajectories (which needs training data and GPU) or physics-based inference during
contact — and neither is reachable on the ~$2.70 of credit left.
