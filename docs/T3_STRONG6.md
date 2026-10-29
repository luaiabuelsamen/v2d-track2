# Tier 3, the six strongest untouched clips — pre-registered, staged, not launched

## Why these six

The Tier 3 fine-tune ran ten clips and lifted their mean AUC 0.33953 -> 0.39432, adopting five
(e00 e21 e22 e23 e26). **Twenty-one clips were never fine-tuned at all**, and Tier 3 is our weakest
board by a wide margin: the entry is 0.25059 against Tier 1's 0.50838.

Ranked by pass-1 AUC, the strongest untouched clips are **e06 0.3122, e12 0.3038, e17 0.2858,
e27 0.2725, e16 0.2499, e15 0.2428**. That is the same band as the clips that gained on Tier 3
(e18 0.3248, e23 0.3027), and the Tier 1 arm independently measured that the fine-tune sharpens a
clip that already tracks rather than rescuing one that does not.

## The run

Six clips, two boxes, three each, same recipe as the Tier 3 fine-tune that worked: warm start from
the pass-1 actor at 299,999,232 -> 349,999,104, START + ORI 0.3 + object_position std 0.08, seed 42,
support per clip, reference = our own retargeted video reconstruction. About 3.3 h per box, so
**6.6 box-hours, about $4.50.**

## The bar, registered now

Adoption is the **input-only** rule (`tools/t3/adopt.py`), unchanged: a clip ships its new rollout
only if the z-drop pick is closer to its own `reference_*` columns than the old one. No ground truth
is read, because on Tier 3 the permitted input is a reconstruction we made and is *not* the scoring
reference — the opposite of Tier 1. That rule got the sign right on 7 of the 9 clips it has judged
and has never once adopted a loss, including an e08 regression of -0.307; both misses were gains it
declined, so it is conservative in the direction that cannot hurt us.

**Registered expectation.** At the measured Tier 3 rate (5 of 10 adopt, about +0.11 AUC each), six
clips project to roughly +0.33 summed, i.e. **+0.011 on the 30-clip entry mean: 0.25059 -> about
0.262.** The arm counts as having worked if at least two clips adopt.

**Not launched.** The remaining Brev credit is about $6 and the Tier 1 strong-four arm is in flight
against it. This launches only once that lands and the credit is confirmed to cover it, because
running out of credit mid-run would lose the clips already paid for.

## Staged and verified, 2026-10-29 13:20 UTC

Both trees staged, 3 of 3 clips each, every clip with a support USDA, manifests 21 files each.

**Reference verified by content, not by name.** The staged parquets are byte-*different* from the
reconstruction tree because the stager rewrites the file, so md5 proves nothing here. Compared on
`robot_root_position`, each staged reference is **0.0000 cm** from its reconstruction solve and
**0.97-1.94 cm** from the clean Tier 1 retarget — it is the reconstruction, and it is not a clean
reference wearing a Tier 3 label.

**Revised on the spend projection (results/brev_spend_warmstart.json): ONE box, not two.** Projected
credit remaining is $6.73, and two boxes at ~$4.50 would leave almost nothing for a recovery if a
box failed bring-up at the measured ~1/3 create-failure rate. Box a (e06 e12 e17, the three
strongest untouched clips, ~$2.25) launches first; box b launches only if box a adopts and the
credit still covers it.

Launch command:

    bash $S/launch_t3s6.sh a      # e06 e12 e17
    bash $S/launch_t3s6.sh b      # e15 e16 e27
