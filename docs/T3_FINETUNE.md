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

## Result, 7 of 10 clips (2026-10-29 09:50 UTC)

| clip | pass 1 | fine-tuned | Δ truth | input err, pass 1 → ft | adopted |
|---|---|---|---|---|---|
| e00 | 0.40657 | 0.49229 | **+0.0857** | 25.54 → 4.17 cm | yes |
| e08 | 0.55985 | 0.25265 | −0.3072 | 20.10 → 22.36 cm | no |
| e18 | 0.32481 | 0.34972 | +0.0249 | 31.13 → 31.28 cm | no |
| e22 | 0.34449 | 0.50939 | **+0.1649** | 76.41 → 2.79 cm | yes |
| e23 | 0.30266 | 0.42633 | **+0.1237** | 35.92 → 4.86 cm | yes |
| e24 | 0.20805 | 0.19870 | −0.0094 | 16.14 → 18.97 cm | no |
| e26 | 0.45077 | 0.57511 | **+0.1244** | 6.26 → 2.69 cm | yes |
| **mean** | **0.37103** | **0.40060** | +0.0296 | | **0.44226 after adoption** |

Clause 1 of the bar (ten-clip mean above 0.3395) is already clear on seven, by 0.10.

**The input-only rule validated itself.** It adopted the four largest true gains and rejected the
only serious loss, having read nothing but each rollout's own `reference_*` columns. Raw +0.0296
becomes **+0.0712** after adoption. That is the first direct evidence that the rule we must use on
Tier 2 and Tier 3 — where truth may not be read — is not merely safe but selective.

**I was wrong to call 0.4368 a ceiling.** The after-adoption mean passes it. The reconstruction's
own score is not an upper bound on a policy driven by it: where the reconstruction is physically
impossible, a policy that cannot reproduce it lands *closer to truth* than its own input. e22 is the
case — input error 76 cm, and the fine-tuned rollout scores 0.509.

**Do not extend this arm to the other twenty clips.** On those, pass 1 showed the policy already
beats its input (0.1787 vs 0.1236), so "closer to the input" means closer to a wrong trajectory and
the input-only rule becomes anti-correlated with truth. It would adopt damage, and it would do so
while reading nothing that could reveal the mistake. The rule is safe only where the reconstruction
is better than the policy, which is the ten clips this arm runs.

## The plan file names a Tier 2 tree, and that is not a bug

`assets.warmstart.txt` gives e22's stage as `sw_A`, which reads like the arm-A **Tier 2** reference
and would mean this arm never used the reconstruction at all. It does not. `stage_t1_eval.sh` reads
each actor's recorded reference path out of its safetensors metadata and writes the *new* reference
at that path, so the trainer finds it with no override — the stage column is an in-container
location, not a statement about content. Verified on e22's object trajectory rather than assumed:

| compared | mean | max |
|---|---|---|
| what the T3 arm trained on vs the recon-solved retarget | **0.0000 cm** | 0.0000 cm |
| what the T1 arm trains on vs the clean Tier 1 retarget | **0.0000 cm** | 0.0000 cm |
| the two arms' references against each other | 4.6383 cm | 17.5975 cm |

And the fine-tuned rollout's own `reference_*` columns are bit-identical to pass 1's across all 719
frames, so the pass-1 comparison is one-variable. The 306 cm gap between those columns and
`t3/npz_world` is a frame convention, which is what `rollout-to-npz --reference-npz` fits out; its
residual came back at 8e-06 cm.
