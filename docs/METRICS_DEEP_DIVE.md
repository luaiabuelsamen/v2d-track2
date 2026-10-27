# Metrics deep dive: the 30-clip combined run, read at the metric level

Board entry AUC **0.39076**, SP-SR 0.56666, MP-SR 0.16666, MPPE 17.35839
(`results/tier2_submission.json`). This reads the same 30 rollouts against the clean truth, one
question per section. Record: `results/metrics_deep_dive.json`. Figures:
`results/figures/deep_dive/` with `captions.json`. Reproduce with `v2d deep-dive` and
`v2d deep-dive-figures`.

**Read this first.** Both success rates threshold the **time-averaged** error, not per-frame
success, so every clip passes or fails once. Our computation reproduces the kit exactly: 17/30
SP-SR = 0.5667 and 5/30 MP-SR = 0.1667, the published scores. Errors below are against Tier 1
truth; the first-loss frames in `docs/SECOND_WAVE.md` were against the noisy input, so they differ.

## The table

`pos`/`ori` says which channel the AUC is lost to (whichever substitution gains more by >0.03);
`MP±` is the margin to MP-SR's 3 cm position threshold in cm; `sel±` is what the registered z-drop
pick cost against that clip's best checkpoint; `panto` is section E.

| clip | object | bucket | AUC | pos-only | ori-only | lost to | pos cm | ori° | lost@ | <5/10/20cm | SP | MP | MP± | shape | sel± | panto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| e24 | watering_can | strong | 0.757 | 0.760 | 0.973 | pos | 2.79 | 7.2 | — | 0.91/1.00/1.00 | Y | **Y** | +0.21 | peaked | +0.008 | 0.00 |
| e18 | blue_trash_can | strong | 0.736 | 0.785 | 0.881 | pos | 2.44 | 5.0 | — | 0.77/1.00/1.00 | Y | **Y** | +0.56 | peaked | +0.000 | 0.00 |
| e12 | black_platform | strong | 0.710 | 0.775 | 0.883 | pos | 2.59 | 4.4 | — | 0.89/1.00/1.00 | Y | **Y** | +0.41 | peaked | +0.009 | 0.00 |
| e27 | woven_basket | strong | 0.692 | 0.731 | 0.926 | pos | 3.02 | 8.8 | — | 0.78/1.00/1.00 | Y | n | **−0.02** | peaked | +0.015 | 0.00 |
| e03 | contigo_coffee_mug | strong | 0.659 | 0.679 | 0.947 | pos | 3.52 | 13.4 | — | 0.75/1.00/1.00 | Y | n | −0.52 | collapsing | +0.000 | 0.00 |
| e29 | rolling_luggage | strong | 0.593 | 0.622 | 0.936 | pos | 4.04 | 4.3 | — | 0.64/1.00/1.00 | Y | n | −1.04 | collapsing | +0.000 | 0.00 |
| e26 | skateboard | strong | 0.577 | 0.889 | 0.618 | ori | 1.68 | 23.0 | — | 0.98/1.00/1.00 | Y | **Y** | +1.32 | peaked | +0.001 | 0.00 |
| e22 | step_ladder | strong | 0.528 | 0.719 | 0.658 | ori | 3.04 | 13.0 | — | 0.81/1.00/1.00 | Y | n | **−0.04** | monotone | +0.074 | 0.00 |
| e23 | toy_stroller | strong | 0.499 | 0.683 | 0.686 | — | 3.48 | 17.8 | — | 0.80/1.00/1.00 | Y | n | −0.48 | monotone | +0.008 | 0.00 |
| e00 | tall_bar_stool | strong | 0.498 | 0.743 | 0.587 | ori | 2.87 | 10.8 | — | 0.90/1.00/1.00 | Y | **Y** | +0.13 | monotone | +0.000 | 0.00 |
| e06 | monitor | strong | 0.473 | 0.699 | 0.633 | ori | 3.36 | 18.7 | — | 0.79/1.00/1.00 | Y | n | −0.36 | monotone | +0.097 | 0.00 |
| e20 | rice_cooker | strong | 0.473 | 0.861 | 0.474 | **ori** | 1.74 | 49.8 | — | 1.00/1.00/1.00 | n | n | +1.26 | monotone | +0.088 | 0.00 |
| e15 | wet_floor_sign | strong | 0.454 | 0.591 | 0.701 | pos | 4.26 | 13.3 | — | 0.61/1.00/1.00 | Y | n | −1.26 | monotone | +0.013 | 0.00 |
| e08 | skinny_wood_chair | strong | 0.443 | 0.705 | 0.580 | ori | 3.26 | 13.5 | — | 0.91/1.00/1.00 | Y | n | −0.26 | collapsing | +0.120 | 0.00 |
| e02 | blue_easter_basket | strong | 0.426 | 0.570 | 0.681 | pos | 4.34 | 21.3 | — | 0.49/0.99/1.00 | Y | n | −1.34 | collapsing | +0.000 | 0.00 |
| e16 | spray_bottle | strong | 0.389 | 0.499 | 0.516 | — | 4.95 | 35.4 | — | 0.46/1.00/1.00 | n | n | −1.95 | peaked | +0.006 | 0.00 |
| e11 | light_blue_book | strong | 0.332 | 0.480 | 0.688 | pos | 5.54 | 21.9 | — | 0.49/1.00/1.00 | Y | n | −2.54 | monotone | +0.005 | 0.00 |
| e10 | white_desk | strong | 0.315 | 0.613 | 0.531 | ori | 4.14 | 8.2 | — | 0.56/0.99/1.00 | Y | n | −1.14 | peaked | +0.000 | 0.00 |
| e05 | big_black_box | lost mid | 0.266 | 0.356 | 0.373 | — | 14.56 | 27.6 | 269 | 0.36/0.63/0.82 | n | n | −11.56 | monotone | +0.119 | 0.05 |
| e07 | iron | lost mid | 0.251 | 0.399 | 0.284 | ori | 9.40 | 105.3 | 310 | 0.44/0.59/0.83 | n | n | −6.40 | peaked | +0.000 | 0.07 |
| e17 | traffic_cone | fell back | 0.247 | 0.294 | 0.406 | pos | 20.46 | 53.4 | 292 | 0.30/0.65/0.78 | n | n | −17.46 | peaked | +0.000 | 0.00 |
| e09 | big_red_bowl | lost mid | 0.245 | 0.555 | 0.355 | ori | 4.70 | 41.0 | 854 | 0.54/0.98/0.99 | n | n | −1.70 | monotone | +0.001 | 0.00 |
| e21 | tall_bar_stool | held | 0.206 | 0.563 | 0.242 | **ori** | 4.43 | 28.4 | — | 0.59/0.98/1.00 | Y | n | −1.43 | monotone | +0.000 | 0.00 |
| e01 | vacuum | never locks | 0.200 | 0.180 | 0.393 | pos | 43.61 | 34.2 | 135 | 0.18/0.37/0.39 | n | n | −40.61 | collapsing | +0.000 | 0.00 |
| e13 | snack_box | held | 0.159 | 0.620 | 0.209 | **ori** | 4.07 | 40.2 | — | 0.57/1.00/1.00 | n | n | −1.07 | collapsing | +0.106 | 0.00 |
| e04 | baseball_bat | fell back | 0.147 | 0.251 | 0.232 | — | 116.61 | 95.6 | 355 | 0.26/0.44/0.45 | n | n | −113.61 | peaked | +0.011 | **0.31** |
| e25 | cyan_water_bottle | fell back | 0.134 | 0.137 | 0.232 | pos | 83.60 | 114.0 | 126 | 0.14/0.15/0.15 | n | n | −80.60 | monotone | +0.002 | **0.72** |
| e19 | cane | lost mid | 0.116 | 0.118 | 0.223 | pos | 57.99 | 71.3 | 133 | 0.14/0.15/0.16 | n | n | −54.99 | monotone | +0.119 | **0.19** |
| e14 | round_table | lost mid | 0.109 | 0.115 | 0.184 | pos | 47.14 | 93.6 | 243 | 0.10/0.23/0.29 | n | n | −44.14 | collapsing | +0.058 | **0.31** |
| e28 | squeegee | lost mid | 0.089 | 0.104 | 0.110 | — | 43.70 | 131.3 | 118 | 0.11/0.12/0.14 | n | n | −40.70 | flat | +0.019 | 0.04 |

### Per bucket

| bucket | n | AUC | pos-only | ori-only |
|---|---|---|---|---|
| strong | 18 | 0.531 | 0.689 | 0.717 |
| lost mid-clip | 6 | 0.179 | 0.275 | 0.255 |
| fell back | 3 | 0.176 | 0.228 | 0.290 |
| held, drifting | 2 | 0.183 | 0.592 | 0.226 |
| never locks on | 1 | 0.200 | 0.180 | 0.393 |

## A. AUC decomposition — the clips split cleanly in two (figure d)

Mean AUC 0.3908; with truth orientation **0.5366**, with truth position **0.5381**. Neither channel
dominates *overall*, which is why one number hid this. Per clip it is not close:

- **Orientation-limited, 9 clips** (e00 e06 e08 e09 e10 e13 e20 e21 e26): perfect orientation would
  take them from 0.377 to **0.695**, **+0.318 each**. Their mean position error is already 3.2 cm.
  e20 is the clearest case — 1.74 cm position error, 49.8° angle, AUC 0.473 against a position-only
  0.861.
- **Position-limited, 8 clips** (e02 e03 e10 e11 e15 e24 e27 e29): +0.269 each from perfect
  position, at a mean 3.96 cm.
- **Object gone, 5 clips** (e04 e14 e19 e25 e28): +0.026 and +0.077. **Neither reward channel can
  reach them** — the object is 44–117 cm away, so no shaping term is the lever.

## B. Where the error lives in time (figure a)

Twenty clips never pass the 10 cm gate at all; their AUC is lost to steady pose error, not to
loss. Ten cross it and never come back, between frame 118 and frame 854. For those ten, position
error averages **3.3 cm before** the loss and **63.4 cm after** it: they are tracking well — better
than several clips that never fail — right up to a single moment. That is why the first-loss frame
correlates +0.636 with AUC, and why for this bucket the fix is to survive the moment, not to track
more tightly.

## C. Success margins — this is the headline (figure g)

**MP-SR passes on 5 of 30. Eleven more fail on position alone, by less than 1.5 cm, with
orientation margin to spare**: e27 (−0.02 cm), e22 (−0.04), e08 (−0.26), e06 (−0.36), e23 (−0.48),
e03 (−0.52), e29 (−1.04), e10 (−1.14), e15 (−1.26), e02 (−1.34), e21 (−1.43).

MP-SR is a quarter of the rank sum and sits at 0.167. Moving those eleven clips' time-averaged
position error under 3 cm would take it to **0.533**. That is a larger prize than anything available
in AUC, and it needs tenths of a centimetre, not a new method. The stock object-position reward is
`exp(-|Δp|²/0.2²)`, which is 0.978 at 3 cm and 0.975 at 1 cm — **flat exactly where MP-SR is
decided**. SP-SR has no comparable cluster: of its 13 failures, 11 are clips that lost the object.

## D. Checkpoint curves (figure c)

13 monotone, 10 peaked, 7 collapsing, 1 flat — **17 of 30 do not improve to the end**, so "train
longer" is not free. The registered z-drop rule picks the per-clip best on 9 of 30 and costs
**0.029 mean AUC** against an oracle that reads truth. That matches the independent
`results/selection_rule_check.json` figure (0.3908 against an oracle 0.4201) and caps every
selection idea at +0.029. Within-run spread over each clip's last five checkpoints is 0.0756 mean,
0.2315 max — the noise floor a second-wave arm must beat.

## E. Pantomime, as a number (figure e)

Palm centre to the nearest point of the object's own AUC vertex cloud. Measured in the **simulator
frame**; the palms are the *reference* palms, because the boxes exported object trajectories only
and the achieved body was never recorded. So this measures "the hand went where it should and the
object did not follow", assuming body tracking is good — the recipe's own assumption (body weight
5 against object 1) and what the strip review reports.

The gate is relative: `ee_pose_w` is an end-effector link origin, not a fingertip, so even the
reference sits a mean 2–10 cm off the surface while holding. A frame counts as lost when the
achieved object is more than 10 cm further from the palm than the reference object is.

**Pantomime fraction correlates −0.647 with AUC, the strongest relationship measured.** e25 0.72
(excess 74 cm), e04 0.31 (106 cm), e14 0.31 (19 cm), e19 0.19 (21 cm), e28 0.04 but 31 cm of
excess. Every one of the 18 strong clips scores 0.00. This confirms `docs/FAILURE_REVIEW_VISUAL.md`
from the metrics side rather than from the pictures.

## F. Correlations (figure f)

| against AUC | ρ | n | reading |
|---|---|---|---|
| pantomime fraction | **−0.647** | 30 | the strongest signal: losing the object, not tracking it badly |
| first-loss frame | +0.636 | 10 | surviving longer is most of the score |
| reference ceiling | +0.466 | 30 | real but loose — 5 clips beat their own reference |
| clip length (frames) | −0.449 | 30 | longer clips have more chances to fail |
| bridge anchors | +0.295 | 30 | weak; the two zero-anchor clips (e04, e28) are both bad |
| hands (one/two/foot) | −0.201 | 30 | two-handed is slightly harder, not decisively |
| object radius | −0.191 | 30 | **size is not the problem**, contrary to a reasonable guess |

## Where the next 0.1 of mean AUC is cheapest

Ranked by AUC per dollar, at $3/clip-run. Headroom = what the decomposition says is available if
the channel were perfect; nobody gets all of it.

| # | change | clips | targets | headroom (mean AUC) | cost |
|---|---|---|---|---|---|
| 1 | tighten `object_position_std` 0.2 → 0.08 m | the 18 at ≥0.30 | **MP-SR**, then AUC | +0.072 if half the position gap closes | $54 |
| 2 | orientation reward that is awake at 20–50° | the 9 orientation-limited | AUC, SP-SR ori | **+0.095** at the ceiling, +0.048 at half | $27 |
| 3 | keep the object in hand | e04 e14 e19 e25 e28 | AUC, all four | +0.054 to reach their reference ceilings | $15 |
| 4 | stop training at the peak | the 7 collapsing | AUC | ≤ +0.029 **total**, and no legal rule collects it | $0 |

**One correction to `docs/SECOND_WAVE.md`.** Its proposal 2 put the wider orientation std on the
fell-back clips e04/e17/e25. The decomposition says that is the wrong clip set: those three gain
only +0.104, +0.048 and +0.003 from *perfect* orientation, because their object is metres away.
(+0.085, +0.160 and +0.098 are their gains from perfect *position*, which is what the column
`orientation_only` holds -- truth position substituted, orientation error left in. The clip set
below is the top nine by perfect-orientation gain and is unaffected.) The
clips that are orientation-limited are the ones that **hold** the object — e13, e20, e21, e26, e09,
e10, e08, e00, e06 — and they are worth +0.318 each. The arm was right; the clips were wrong.

**And the biggest single item is not an AUC item at all.** Item 1 moves MP-SR from 5/30 to a
possible 16/30 on a quarter of the rank sum. If only one thing is bought, buy that.
