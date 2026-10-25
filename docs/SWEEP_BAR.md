# Judging bar for the C/D/E/F seed sweep

Written and committed before the sweep launches. vamp built the arms and left the bar to v2d
(`docs/WS_residual.md`: "The bar is v2d's to write down before launch"), with one constraint:
judge each arm against its own spread, because the old 0.014 margin from B2's seed-to-seed
spread does not carry over.

## What B3 actually left us

| run | AUC | | run | AUC |
| --- | --- | --- | --- | --- |
| A s42 | 0.2013 | | B s42 | 0.2035 |
| A s43 | 0.2016 | | B s43 | 0.2000 |
| C s42 | **0.3605** | | C s43 | 0.2023 |

Five of the six runs form a cluster spanning 0.2000-0.2035, SD 0.0013. The sixth, C s42, sits
**123 cluster-SDs** above it. Arm C's two-seed mean beats A by 0.0799 AUC.

## Why the mean test is the wrong test here

Arm C's sample SD is 0.1119 -- it is large precisely *because* the arm is bimodal, so the SD is
measuring the gap between two modes, not noise around a centre. Feeding it into a t-test:

| seeds/arm | SE(C - A) | mean test detects | verdict |
| --- | --- | --- | --- |
| 4 | 0.0791 | >= 0.155 AUC | effect is 0.0799: **underpowered about 2x** |
| 16 | 0.0395 | >= 0.0775 AUC | first n where the real effect clears |

A four-seed mean comparison cannot resolve this effect. Sixteen seeds per arm is 64 runs, about
9.6 h of A6000 time; that is not what the effect is worth at this stage. So the mean is reported
but is **not** the deciding statistic.

## The deciding statistic: escape rate

A run "escapes" if its AUC > **0.25**. The threshold is fixed now, before the runs, and nothing
about the conclusion depends on where in the empty region it sits: 0.25 is 37 cluster-SDs above
the cluster mean and far below C s42, and *any* threshold in [0.21, 0.35] partitions B3's six
runs identically. It is a statement about which mode a run landed in, not a score cutoff.

Baseline: the four A and B runs from B3, 0 of 4 escaped.

Two-sided Fisher exact against that 0/4 baseline:

| escapes | p | reading |
| --- | --- | --- |
| 4/4 | **0.029** | the only outcome at n=4 that clears 0.05 |
| 3/4 | 0.143 | suggestive, not established |
| 2/4 | 0.429 | nothing |
| 2/2 | 0.067 | the best an n=2 arm can do -- cannot clear 0.05 |

## Seed allocation, and what it does and does not buy

Luai set the allocation: **4 seeds for C (44-47), 2 each for D, E, F (44-45)** -- 10 runs, about
1.5 h. This differs from vamp's proposed 4 each (16 runs) and the consequence is explicit:

- **C** can reach significance, but only on a clean sweep (4/4, p = 0.029).
- **D, E, F** cannot reach p < 0.05 at n=2 even if both seeds escape (p = 0.067). They are run
  here to *locate the mechanism*, not to establish an effect, and will be reported as
  descriptive. Promoting any of them to a claim needs 2 more seeds each.

I am recording that rather than quietly reporting p-values that the design cannot support.

## Predictions, registered now

From vamp's Amendments 3 and 4: the gain comes from training occasionally seeing the object not
held by the VOC target, within the correction bound.

| arm | correction | prediction |
| --- | --- | --- |
| C | 2 cm, 20 deg, body + VOC targets | bimodal, escape rate above 0 |
| D | 1 cm, 10 deg | escape rate between A and C |
| E | C annealed to 0 at the budget | near A, no escapes, small spread |
| F | C on reward and termination only, VOC target untouched | near B (~0.20), no escapes |

E near A and F near B support the VOC explanation. **F escaping refutes it** -- the gain would
then not be about the VOC target at all.

## What ships

Arm C replaces stock arm A as the Tier 2 training recipe only if **both** hold:

1. escape rate >= 3/4, and
2. at least one seed beats the best A or B run by more than 0.05 AUC.

Otherwise the 30-clip Tier 2 run uses stock arm A. One escape in four (p = 1.0) is a curiosity,
not a recipe, and at $76-86 per 30-clip pass a coin-flip recipe is worse than a steady one.
