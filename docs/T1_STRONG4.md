# Tier 1, the four strong clips the prune skipped — pre-registered before launch

## What the prune got wrong, and what it got right

When the Tier 1 fine-tune queues were pruned, the rule was "run only clips that ship their final
checkpoint", on the reasoning that a clip whose z-drop pick was an *earlier* checkpoint had already
measured that late-stage training hurt it, and the fine-tune re-runs exactly 299,999,232 ->
349,999,104. Eleven clips were dropped on that rule, saving ~$4.80.

With all 18 scored, the real split is not final-checkpoint, it is **how good the clip already is**:

| shipped AUC | n | adopted | mean delta |
|---|---|---|---|
| < 0.35 | 3 | **0** | +0.0019 |
| >= 0.35 | 15 | **8** | +0.0441 |

Every one of the eight adoptions came from a clip already at 0.4758 or better. The fine-tune does
not rescue a weak clip; it sharpens a clip that already tracks. (e18 and e29 show the converse:
shipping a final checkpoint is what makes a gain *possible*, not what makes one happen — both ship
final checkpoints and both lost.)

Four of the eleven pruned clips are strong: **e03 0.7806, e08 0.6654, e15 0.6653, e27 0.7795.**
Those were dropped for the wrong reason, and they are the cheapest AUC left on Tier 1.

## The run

Four clips, two boxes, two clips each, gen-1 recipe unchanged (START + ORI 0.3 + object_position
std 0.08, seed 42, warm start from the shipped actor at 299,999,232 -> 349,999,104, scene.support
per clip). Clean Tier 1 references, same staging as gen 1. ~1.1 h per clip, so ~2.4 box-hours:
**about $1.70 of compute, ~$3 with bring-up.**

## The bar, registered now

Adoption is the same no-regression rule as the rest of the Tier 1 fine-tune
(`tools/t1/adopt.py`): a clip replaces its shipped rollout only if kit AUC rises **and** spider_sr
does not fall **and** maniptrans_sr does not fall **and** mppe_cm does not rise. Reading the kit
score is legal on Tier 1 and only on Tier 1, because `config/tracks.json` names the released Tier 1
trajectory — the permitted input — as the configured scoring reference.

There is therefore **no downside case**: a clip that regresses contributes nothing and ships its
old rollout. The question is only whether the $3 buys anything.

**Registered expectation.** At the measured strong-clip rate (8 of 15 adopt, mean adopted gain
+0.169), four clips project to about +0.36 of summed AUC, i.e. **+0.012 on the 30-clip entry mean**:
0.50838 -> about **0.520**, against CHORD's 0.51038. The arm counts as having worked if at least
one clip adopts; it counts as having been worth launching if the entry mean crosses 0.51038.

**The seven weak clips are deliberately not run**: e05 e07 e13 e14 e19 e25 e28, all shipping under
0.27, where the measured adoption rate is 0 of 3 and the mean delta is +0.002. Paying ~$5 for them
would be buying a number the data says is zero.
