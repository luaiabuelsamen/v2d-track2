# Tier 3 fine-tune: pre-registered (2026-10-29)

## I had this wrong, and the mean is why

After pass 1 I wrote in `results/t3_policy_eval.json` that "the policy already scores at the
reconstruction's own level (0.2323 vs 0.2280), so GPU spent on better Tier 3 policies buys little".
That compared **means**, and the means hide two opposite effects that happen to cancel:

| clips | reconstruction AUC | policy AUC | effect |
|---|---|---|---|
| the 10 that track well | 0.4368 | 0.3395 | the policy **loses 0.0973** |
| the other 20 | 0.1236 | 0.1787 | the policy **gains 0.0551** |
| all 30 | 0.2280 | 0.2323 | +0.0043, which looks like nothing happening |

The worst per-clip losses are e26 −0.324, e18 −0.289, e21 −0.266, e22 −0.221, e29 −0.203. On those
clips the reconstruction is good — under 5° mean orientation error and under 16 cm — so the
reference is not what is hurting them. What is hurting them is the same distribution shift the Tier
1 experiment measured: an actor driven by a reference it never trained on tracks it badly.

This also explains the one number that looked worst about pass 1. **SP-SR is 0.000 for the policy
and 0.333 for the reconstruction**, because nine clips' reconstructions sit inside the 10 cm gate
and the policy's own tracking error pushes them out. e18 is the clearest: the reconstruction is
3.95 cm from truth, the policy's rollout is 31.2 cm.

## What runs

The same warm start as Tier 1 (`brev/patch_warm_start.py`), with the **Tier 3 reconstruction's
retargeted motion** as the reference instead of the clean Tier 1 track. 50M steps of fine-tuning
from each clip's shipped actor, cold critic, learning-rate guard for 2,000 updates, seed 42, support
per clip, otherwise the arm-A recipe those actors were trained with.

Clips — the ten the reconstruction tracks well, which is where the loss is:

`e00 e08 e18 e20 e21 e22 e23 e24 e26 e29`

Nine ship from arm A. **e21 ships from the first wave**, which did not carry Part A's tightened
object-position std, so fine-tuning it under arm-A flags changes its reward as well as its
reference. It is included anyway, and this is safe for a shipping run because the per-clip
**adoption rule** decides what ships: a fine-tuned rollout replaces the pass-1 rollout only if its
z-drop-selected checkpoint sits **closer to the reconstruction input** than the pass-1 rollout does.
No truth is read. On these clips that rule points the right way, because the reconstruction is
closer to truth than the policy is — tracking the input better *is* the goal here.

## The bar, registered

Adopt per clip under the input-only rule above. The arm as a whole counts as a success only if the
ten-clip mean kit AUC **beats 0.3395** (what pass 1 scored on these same ten) and no clip falls more
than 0.02 below its pass-1 number. The ceiling is 0.4368, the reconstruction's own score on these
clips, and a policy cannot be expected to reach its input exactly.

**Stated now, so it cannot be rationalised later:** if the ten-clip mean lands below 0.3395, the
warm start does not transfer across a reference this different, and that also retires the Tier 1
fine-tune as a method rather than just as a run.

## Cost

| | boxes | clips/box | GPU-h | at $0.68 |
|---|---|---|---|---|
| this arm | 2 | 5 sequential | ~7.1 | **$4.83** |

Budget: $30.38 credit. Tier 1 pilot $1.60 (running), this $4.83, Tier 1 full run $14.30 if its bar
clears — $20.73, leaving ~$9.65.

## What was tried first and did not work

`src/v2d/stage/t3_clean.py` rejects there-and-back relocalisation jumps in the reconstruction and
lightly smooths position, using no ground truth. It found 61 outlier frames across 5 of 30 clips and
improved mean position error by **0.13 cm**, from 66.44 to 66.31. The diagnosis
(`results/t3_recon_diag.json`) had split the clips into 11 jump / 11 drift / 8 clean and e01 looked
like a pure filtering problem — 10 jumps over 20 cm against a 6.7 cm median. It was not: the large
errors are sustained mis-registration, not outliers. The cleaner is kept because it is cheap and
correct, but it is not a lever.
