# Failure taxonomy, the 30 shipping rollouts

Over the entry as it stands: mean kit AUC **0.4631** across 30 clips, 11 of them below 0.35. Each is classed by mechanism, in the fixed order `v2d taxonomy` documents, with the frame the mechanism fires. Position and orientation error are against the Tier 1 clean truth in the released frame, from the same conversion the kit scorer reads.

## The clips below 0.35

| clip | AUC | class | frame | object | mass kg | hands | closure | palm depth cm | ceiling | headroom |
|---|---|---|---|---|---|---|---|---|---|---|
| e28 | 0.0891 | dropped_mid_clip | 118 / 988 | squeegee | 1.0 | two-hand | yes | l0.0, r1.01 | 0.1397 | 0.0506 |
| e19 | 0.1163 | never_grasped | 133 / 886 | cane | 1.0 | one-hand | yes | l0.25, r0.0 | 0.5664 | 0.4501 |
| e25 | 0.1334 | dropped_mid_clip | 126 / 859 | cyan_water_bottle | 1.0 | one-hand | **no** | l1.12, r0.12 | 0.7865 | 0.6531 |
| e04 | 0.1466 | toppled_on_contact | 87 / 803 | baseball_bat | 1.0 | one-hand | **no** | l1.8, r0.0 | 0.4903 | 0.3437 |
| e13 | 0.1594 | held_but_drifting | - / 922 | snack_box | 1.0 | foot | **no** | l0.52, r0.0 | 0.1925 | 0.0331 |
| e01 | 0.2001 | never_grasped | 134 / 747 | vacuum | 1.0 | one-hand | yes | l0.0, r0.0 | 0.3803 | 0.1802 |
| e14 | 0.2055 | reference_ceiling | - / 993 | round_table | 1.0 | two-hand | yes | l3.45, r1.45 | 0.2236 | 0.0181 |
| e21 | 0.2061 | reference_ceiling | - / 678 | tall_bar_stool | 1.0 | one-hand | yes | l2.52, r1.05 | 0.2042 | -0.0019 |
| e07 | 0.2508 | toppled_on_contact | 284 / 896 | iron | 1.0 | one-hand | yes | l0.0, r0.78 | 0.8645 | 0.6137 |
| e05 | 0.2660 | dropped_mid_clip | 269 / 1003 | big_black_box | 1.0 | one-hand | yes | l4.72, r3.59 | 0.4717 | 0.2057 |
| e09 | 0.3003 | held_but_drifting | - / 870 | big_red_bowl | 1.0 | two-hand | yes | l0.33, r0.29 | 0.6188 | 0.3185 |

## AUC headroom by class

`headroom` is each clip's reference ceiling minus what it scores: what a perfect fix to that mechanism could buy, summed over the clips in the class, and what that is worth on the 30-clip mean.

| class | clips | AUC now | headroom | mean-AUC gain if fixed |
|---|---|---|---|---|
| held_but_drifting | 11 (e02 e06 e09 e11 e13 e16 e17 e22 e26 e27 e29) | 5.445 | 2.213 | +0.0738 |
| toppled_on_contact | 2 (e04 e07) | 0.397 | 0.957 | +0.0319 |
| dropped_mid_clip | 3 (e05 e25 e28) | 0.488 | 0.909 | +0.0303 |
| never_grasped | 2 (e01 e19) | 0.316 | 0.630 | +0.0210 |
| reference_ceiling | 12 (e00 e03 e08 e10 e12 e14 e15 e18 e20 e21 e23 e24) | 7.246 | -0.569 | +0.0026 |

## Every clip

| clip | ships | AUC | class | frame | pos err cm max/mean | ori err deg max/mean | travel cm achieved/reference |
|---|---|---|---|---|---|---|---|
| e00 | A | 0.5178 | reference_ceiling | - | 5.2/1.5 | 71.4/17.4 | 76.9/78.7 |
| e01 | first wave | 0.2001 | never_grasped | 134 | 118.9/43.6 | 79.2/34.2 | 2.5/121.4 |
| e02 | first wave | 0.4263 | held_but_drifting | - | 10.9/4.3 | 77.7/21.3 | 99.7/99.8 |
| e03 | A | 0.7806 | reference_ceiling | - | 6.0/2.4 | 66.7/13.7 | 41.3/41.6 |
| e04 | B1 | 0.1466 | toppled_on_contact | 87 | 226.3/102.4 | 179.9/106.2 | 159.0/194.8 |
| e05 | first wave | 0.2660 | dropped_mid_clip | 269 | 56.0/14.6 | 96.4/27.6 | 80.9/85.9 |
| e06 | A | 0.4758 | held_but_drifting | - | 8.5/2.9 | 34.2/20.6 | 38.7/41.2 |
| e07 | first wave | 0.2508 | toppled_on_contact | 284 | 24.1/9.4 | 179.9/105.3 | 24.9/29.4 |
| e08 | A | 0.6654 | reference_ceiling | - | 6.4/1.7 | 37.1/10.3 | 54.9/57.1 |
| e09 | B4x | 0.3003 | held_but_drifting | - | 10.3/4.2 | 70.8/38.4 | 106.3/109.8 |
| e10 | A | 0.4794 | reference_ceiling | - | 9.2/2.9 | 20.6/6.8 | 111.7/106.4 |
| e11 | A | 0.4676 | held_but_drifting | - | 341.3/4.0 | 180.0/22.2 | 125.4/315.9 |
| e12 | A | 0.7904 | reference_ceiling | - | 8.4/2.0 | 16.0/3.7 | 51.9/50.3 |
| e13 | first wave | 0.1594 | held_but_drifting | - | 9.8/4.1 | 95.6/40.2 | 94.1/95.3 |
| e14 | B2 | 0.2055 | reference_ceiling | - | 31.8/5.8 | 94.1/25.9 | 131.4/130.5 |
| e15 | A | 0.6653 | reference_ceiling | - | 5.6/2.7 | 30.3/10.4 | 163.8/163.8 |
| e16 | A | 0.5639 | held_but_drifting | - | 8.8/3.1 | 72.2/32.2 | 113.1/111.2 |
| e17 | B4 | 0.4470 | held_but_drifting | - | 10.1/4.8 | 91.3/25.3 | 191.1/197.4 |
| e18 | A | 0.7385 | reference_ceiling | - | 6.1/1.9 | 17.8/6.3 | 93.3/91.6 |
| e19 | B2 | 0.1163 | never_grasped | 133 | 125.2/58.0 | 178.7/71.3 | 2.7/125.8 |
| e20 | A | 0.7149 | reference_ceiling | - | 6.4/1.6 | 100.6/20.2 | 8.1/4.0 |
| e21 | first wave | 0.2061 | reference_ceiling | - | 10.3/4.4 | 75.2/28.4 | 111.2/112.6 |
| e22 | A | 0.6686 | held_but_drifting | - | 6.7/2.1 | 28.0/11.0 | 122.9/119.6 |
| e23 | A | 0.5881 | reference_ceiling | - | 7.7/2.5 | 33.9/17.4 | 135.4/136.6 |
| e24 | A | 0.8942 | reference_ceiling | - | 4.1/1.5 | 16.6/7.5 | 17.0/17.3 |
| e25 | B1 | 0.1334 | dropped_mid_clip | 126 | 153.9/74.6 | 180.0/98.8 | 101.3/105.2 |
| e26 | A | 0.5930 | held_but_drifting | - | 5.5/1.6 | 56.1/19.0 | 13.3/15.9 |
| e27 | A | 0.7793 | held_but_drifting | - | 8.0/2.4 | 21.0/6.5 | 153.2/151.6 |
| e28 | first wave | 0.0891 | dropped_mid_clip | 118 | 90.3/43.7 | 179.5/131.3 | 20.4/101.9 |
| e29 | A | 0.5638 | held_but_drifting | - | 8.5/4.3 | 24.6/4.3 | 64.7/67.8 |

## Figures

In `results/figures/hail_mary/`, captions in `captions.json`:

- `a_auc_lost_by_class.png` — AUC headroom per failure class, clip counts on the bars.
- `b_failing_clip_error_curves.png` — position error against the clean truth per frame for every clip below 0.35, failure frame dashed, the 10 cm gate dotted.
- `c_mass_vs_auc.png` — mass against AUC beside radius against AUC. Mass is a constant: all 30 object URDFs carry 1.0 kg and inertia 0.01, so it explains nothing, and the figure says so on its face rather than implying a null result.

## The one mechanism worth fixing

**Held but drifting** is the single class holding the most AUC: 11 clips, 2.21 of headroom between them, worth **+0.074** on the 30-clip mean if it were fully closed — more than the next two classes together (toppled_on_contact and dropped_mid_clip). And it is not a grasp problem: these clips never lose the object at all, by either signal, so no amount of work on acquisition or on termination touches them. What they lose is the angle — 7 of the 11 exceed 45° of orientation error while holding position inside 10 cm, which is the same channel docs/METRICS_DEEP_DIVE.md measured at +0.3178 mean AUC of headroom from substituting truth orientation. We also have one causal read on it rather than a correlation: B4 raised the orientation reward's std from 0.3 to 0.8 on e17 and took it from 0.2466 to 0.4470 with SP-SR flipping to pass, one variable, same seed. That is the mechanism to spend the next GPU hour on, and B4x is already testing exactly it on eight more clips. The three loss classes together (toppled, dropped, never grasped) hold 2.50 of headroom over 7 clips, which is worth having but is 7 separate mechanisms — a bat that topples, a bottle that slips, a cane never picked up — and each needs its own change, where the angle is one knob across eleven clips.
