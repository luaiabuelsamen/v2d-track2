# The 3-clip full-budget A/B: what it is and what decides it

Written before the runs. Nothing below is adjusted after seeing a number.

## The question

Does the gated support-contact cleaner earn a 30-clip Tier 2 pass? The 30-clip pass costs
126.9 GPU-hours and $76-86, so it is not something to buy on a 10M-step signal from one clip.

Two arms, stock recipe both times, 349,999,104 steps (the full published recipe), seed 42:

- **RAW** -- stock FlashCHORD on the **released Tier 2 retarget**. This is our stand-in for the
  CHORD baseline: it is what anyone gets by running the published recipe on the released input.
- **CLEAN** -- stock FlashCHORD on the **gated-cleaned reference**.

The arms differ in one thing only, the object trajectory the reference was retargeted from. Same
recipe, same seed, same step count, same evaluation.

## The three clips, and why these three

The gated cleaner changes **7 of the 30 clips** (2, 3, 9, 16, 18, 24); the other 23 it leaves
byte-identical, so RAW and CLEAN would be the same run and the comparison would be empty. e24 is
excluded by instruction, which leaves six. Within those six the diversity the brief asks for is
almost forced:

| clip | object | support | hands | corrected frames | mean tilt |
|---|---|---|---|---|---|
| **e18** | blue_trash_can | **floor only** -- every rest segment at 0 cm | whole body, drag | 525/858 (61%) | 13.8° |
| **e3** | contigo_coffee_mug | **table only** -- every rest segment at 77.7 cm | right hand | 347/781 (44%) | 22.2° |
| **e9** | big_red_bowl | mixed -- 8 segments at 46.9 cm, then floor | **both hands** | 204/870 (23%) | 7.6° |

e18 is the only pure-floor clip the cleaner touches other than e24. e9 is the only two-handed one
it touches. For the table slot, e3 and e25 both qualify and e3 was chosen for having twice the
corrected frames and 2.3x the tilt, so it carries more signal; e25 (cyan_water_bottle, handover,
153/859, 9.5°) is the alternate if e3 fails to retarget.

Everything in that table -- support heights, handedness from the released task captions, corrected
frames, tilt -- is read from the Tier 2 release and the cleaner's own output. **No part of the clip
choice looks at the Tier 1 clean reference**, which the brief forbids as input or supervision. The
clean reference enters once, at scoring, which is what it is for.

Axes the three differ on: support (floor / table / raised-rest-then-floor), handedness (none /
one / both), object scale (trash can / mug / bowl), correction size (13.8° / 22.2° / 7.6°),
length (858 / 781 / 870 frames).

## Step 1: the verified baseline, and what counts as "far off"

Written before the three RAW cells were created. Step 1 is NVIDIA's recipe unchanged, 349,999,104
steps, on the released Tier 2 retarget, for e18 / e3 / e9. No method change of ours is in it. Its
job is to prove our pipeline reproduces something recognisable before any arm is compared.

Reference point: the CHORD baseline's **Tier 2 board average** -- AUC 0.374, SP-SR 0.533,
MP-SR 0.067, MPPE 20.5 cm.

Two things make a strict match the wrong expectation, and both are stated now rather than after
the numbers land:

- that average is over all **30** clips and we are running **3**, and per-clip spread on this
  dataset is enormous -- our Tier 1 entry ranged 0.004 to 0.378 AUC across clips at a fixed budget;
- our RAW arm is a **stand-in** for the baseline, not the baseline: it is our retarget of the
  released Tier 2 trajectory, scored on the host scorer against the clean reference.

So the comparison is on the **mean over the three clips**, and the stop-and-debug lines are set
wide enough that only a broken pipeline trips them:

| | stop and debug if |
|---|---|
| mean AUC | **< 0.187** (half the baseline average) |
| mean MPPE | **> 41 cm** (double the baseline average) |
| SP-SR | **0.0000 on all three clips** -- investigate even if AUC and MPPE look fine |

Why those lines. The Tier 1 entry at 1/850th of this budget scored AUC 0.134 and MPPE 74.2 cm. A
pipeline with a wrong frame, wrong assets or the wrong reference would land near those numbers or
worse, not near 0.374 and 20.5. Half and double the board average sit comfortably between the two
regimes. The SP-SR line is separate because the baseline gets 0.533 there while every run we have
done at every budget has returned exactly 0.0000, so three more zeroes at full budget would mean
either the success metric needs something we are not doing or the recipe needs more than we think
-- and I would rather find that out here than infer it from a cleaner comparison.

**A single clip below a line is not a stop signal**; the mean is. One clip at 0.05 against two at
0.45 is clip variance, which is why the test is pre-committed to the mean.

If step 1 passes, step 2 is the compressed curriculum (~50M steps, VOC faded inside the budget) on
the same three clips, accepted only if it **ranks the clips the same way** the full runs do. If
step 1 trips a line, nothing proceeds until the pipeline is understood.

## What the arms actually differ by (measured after the references were built, before any cell ran)

The gated cleaner tilts a resting object onto its support; it does not move it. Measured on the six
staged references:

| clip | object position delta | object rotation delta (max / mean) | robot root | robot joints |
|---|---|---|---|---|
| e18 | constant 0.589 cm | 20.6° / 14.2° | 0.000 cm | 0.0000 rad |
| e3  | 0.000 cm | 31.8° / 24.4° | 0.000 cm | 0.0000 rad |
| e9  | 0.000 cm | 15.3° / 5.7° | 0.000 cm | 0.0000 rad |

So this is an **orientation** A/B almost entirely -- and the body motion is bit-identical between
the arms on all three clips, because it comes from the human capture rather than from the object.
That is a clean single-channel comparison, and it is not a no-op: AUC and MPPE both read object pose
through mesh vertices, so a rotation-only change moves both. e18's position delta is a constant
offset, which is the floor clip's penetration removal.

It does bound the claim. If CLEAN wins, what won is tilt correction on resting segments, not
"cleaning" in general; the position channel is untested by this experiment.

## The gate

Luai's rule, binding. The 30-clip pass is **proposed** only if all three hold:

1. **mean AUC over the three clips: CLEAN > RAW**, and
2. **mean MPPE over the three clips: CLEAN < RAW** (lower is better), and
3. **full350 on e24 shows the full recipe produces non-zero success** -- SP-SR > 0 or MP-SR > 0.

Condition 3 is not about the cleaner at all. Every run to date, at every budget, has returned
SP-SR and MP-SR of exactly 0.0000. If the full recipe cannot clear a success threshold even once
on a clip we have trained to convergence, then the two success-rate leaderboards are out of reach
whatever the reference, and the rank-sum case for a 30-clip pass changes shape.

Failing the gate does not mean the cleaner is worthless. It means we do not buy the 30-clip pass
on this evidence.

## What I will report alongside it, stated now so it cannot be chosen later

- **Per-clip paired signs.** 3/3 in CLEAN's favour is the strong case. 2/3 with a positive mean is
  the weak case and I will say so in those words.
- **Whether the AUC gain clears 0.014**, the margin pre-registered in `docs/B3_MARGIN.md` from two
  B2 seeds. This is **not** a veto -- the gate above is Luai's and I am not rewriting it -- but if
  the gate passes on a mean gain under 0.014 I will report that it passed on a margin we cannot
  distinguish from noise.
- **The honest limit of the design.** Six runs, one seed per cell, three clips. It can establish
  direction and rough magnitude. It cannot establish significance: B3 showed arm-to-arm seed spread
  of 0.0003 for stock arm A at 10M but 0.1582 for arm C, so seed variance is recipe-dependent and
  we have no 350M replicate to estimate it from. A 3/3 sign sweep with a gain well over 0.014 is
  the only pattern that should be called conclusive at this n.

## Cost

Measured on this hardware: 24,550-25,074 env steps/s for stock arm A at 350M on an A6000
(`v2d-full350`, e24). At a conservative 24,000 steps/s:

| | |
|---|---|
| training, per run | 349,999,104 / 24,000 = 4.05 h |
| provision + image build + staging | ~0.6 h |
| evaluating 7 checkpoints at `world_count=1` | ~0.2 h |
| **per run** | **~4.8 h** |
| 6 runs | 28.8 GPU-hours |
| at $0.68/h (`massedcompute_A6000_plus`) | **~$19.6** |

Wall clock is ~4.8 h if all six run at once. **Six-way concurrency is unverified** -- we measured
4 concurrent A6000s in this org and never tried 6. If 6 is refused the fallback is 4 boxes then 2,
which costs the same GPU-hours and roughly doubles the wall clock; the launcher queues rather than
fails.

Against the budget: $3.91 spent when this was written, with full350 and the sweep still running
and due to add about $2.4 before they are deleted, so ~$6.3 at launch and ~$23.7 left. **$19.6
leaves roughly $4 of margin**, which is thin: one re-provision cycle per box is about $2. If
anything overruns I would rather drop the e9 pair and report a 2-clip A/B than cross $30, and I
will ask before doing either.

---

## Addendum, 2026-10-25: step 1 TRIPPED, and why

The three RAW full-length cells ran to 349,999,104 steps and were evaluated at all seven
checkpoints. Against the lines pre-registered above, on the mean over the three clips:

| line | value | observed | tripped |
|---|---|---|---|
| mean ADD-AUC below | 0.187 | **0.0423** | **yes** |
| mean MPPE above | 41 cm | **55.20 cm** | **yes** |
| SP-SR zero on all three | — | e18 reaches 1.0 | no |

Per clip, at the final checkpoint (evaluator):

| clip | role | AUC | MPPE | SP-SR | distinct AUC values over 7 checkpoints |
|---|---|---|---|---|---|
| e03 | table | 0.0056 | 91.84 cm | 0.0000 | **1** |
| e09 | two-handed | 0.0047 | 68.86 cm | 0.0000 | **1** |
| e18 | floor | 0.1166 | 4.91 cm | 1.0000 | 6 |

**Cause: `scene.support` was hardcoded `false` in every run script.** `brev/run_ab350.sh:53`,
`brev/run_full350.sh:48` and `brev/run_tier1_backup.sh:67` all passed `scene.support=false`, while
NVIDIA's `configs/scene/train.yaml` and `eval.yaml` default to `support: true` and the dataset ships
a `support_surface/<seq>_support.usda` per clip. Without the table in the scene the object falls at
reset and the episode terminates on frame ~0: `termination_cause_fraction` is `object=1.0, wrist=1.0`
at every e03 and e09 checkpoint, and their AUC is *bit-identical* across all seven checkpoints
(0.005601792573623559 and 0.0047413793103448275) — the constant-error signature. e03's support USDA
declares one Cylinder whose top sits at 0.7573 + 0.04/2 = **0.7773 m**, matching the 77.7 cm object
rest height measured from the reference.

e18 and e24 are floor clips whose released support USDA is the empty 125-byte stage, so
`support=false` was already the correct flag for them. That is precisely why they are the two clips
that learned, and it is the control that isolates the bug: **e18's final checkpoint host-scores
ADD-AUC 0.3826, SP-SR 1.0, MPPE 4.17 cm**, at or above the CHORD Tier 2 board average on AUC and far
better on MPPE. The pipeline is sound; the scene was missing a table.

14 of the 30 Tier 2 clips ship support primitives. The same bug explains the ~12 near-zero clips in
the Tier 1 entry already submitted.

Because the trip is explained by a scene-construction bug rather than by the reference arms, step 1
is **not** evidence about RAW vs CLEAN. The fix is the per-clip rule `brev/gate_on_instance.sh`
already carried in `support_flag_for`, and e03 and e09 are relaunched with it; e18 is not relaunched,
since its flag was already correct and its result stands.

One observation recorded without being chased: at e24's 300M and 350M checkpoints the evaluator
reports SP-SR 1.0 while the host scorer reports 0.0. e18 agrees between the two (1.0 and 1.0), so the
disagreement is specific to e24's conversion against the gated-cleaned reference, not a general fault
of either scorer.

---

## Addendum, 2026-10-26: step 1 re-measured with the support fix — the AUC line is crossed, the pipeline is not broken

RAW e03 and RAW e09 were relaunched with the per-clip rule (`brev/support_flag.sh`), both passing
`scene.support=true` and resolving a real support USDA; e18 was not relaunched, since its flag was
already correct. Both cells ran the full 349,999,104 steps and were evaluated at all seven
checkpoints. Full record, including every checkpoint and the superseded numbers, in
`results/ab350_raw_baseline.json`.

Against the lines pre-registered above, on the mean over the three clips:

| line | value | observed | tripped |
|---|---|---|---|
| mean ADD-AUC below | 0.187 | **0.1345** | **yes** |
| mean MPPE above | 41 cm | 33.20 cm | no (was 55.20 cm) |
| SP-SR zero on all three | — | e09 and e18 both reach 1.0 | no |

Per clip, at the final checkpoint:

| clip | role | support | evaluator AUC | evaluator MPPE | evaluator SP-SR | host AUC | host MPPE | host SP-SR | distinct AUC over 7 ckpts |
|---|---|---|---|---|---|---|---|---|---|
| e03 | table | yes | 0.0611 | 89.56 cm | 0.0 | 0.0602 | 89.78 cm | 0.0 | **7** |
| e09 | two-handed | yes | 0.2259 | 5.13 cm | 1.0 | 0.2409 | 5.31 cm | 0.0 | **7** |
| e18 | floor | no | 0.1166 | 4.91 cm | 1.0 | 0.3826 | 4.17 cm | 1.0 | 6 |
| e24 | floor (full350, gated-cleaned) | no | 0.4125 | 4.64 cm | 1.0 | 0.4500 | 4.16 cm | 0.0 | — |

**The 3-clip RAW mean AUC is 0.135 against the 0.187 line. That line was written to catch a broken
pipeline. It did not catch one this time, and is not read as one.** What the trajectories show is a
working pipeline whose remaining failures are mid-episode drops — a policy problem:

- The object starts on its support surface at the reference height and is tracked to about **1 cm**
  over the opening 50 frames (mean 0.94–1.69 cm on 13 of the 14 checkpoints; e03 at 249M is the one
  exception at 4.44 cm). Reset, scene and reference agree.
- Error first exceeds the evaluator's 10 cm object threshold at frame **44–376**, well inside clips
  of 781 and 870 frames. e03's first drop (achieved *z* below 0.15 m while the reference stays above
  it) lands at frame 67–516 depending on the checkpoint; the mug starts on the table at *z* 0.892 m
  and ends on the floor at *z* 0.038 m while the reference keeps it at *z* ≥ 0.876 m throughout.
- Each cell now shows **7 distinct AUC values across 7 checkpoints**. The broken run showed 1.
- e09's final checkpoint holds and places the bowl for the whole clip: SP-SR 1.0,
  `termination_cause_fraction` all 0.0, MPPE 5.13 cm, position error never over 10 cm.
- Both runners passed `scene.support=true`; `flash_chord/scene/setup.py:62` raises
  `FileNotFoundError` when support is enabled and no asset resolves, so getting past scene build is
  itself proof the surface was loaded.

Read `first_drop_frame` next to the other columns, never alone. On e09 it is `None` at 100M and 150M
for the opposite of a good reason — the bowl is never lowered at all (achieved *z* ends at 0.534 m
against a reference 0.044 m), a failure to place rather than a drop — and 250M's frame 655 is the
policy leading the reference down during the place, not a failure.

**The 30-clip Tier 2 run stays NOT approved.** The decision rule above proposes it only if CLEAN
beats RAW on mean AUC and MPPE across the three clips; the CLEAN arm has not been run, so the rule
cannot be evaluated, and it needs Luai's explicit go in any case.

Nothing here compares RAW against CLEAN. RAW is only now measured honestly.

The e24 SP-SR disagreement noted above also appears on e09: the evaluator says 1.0, the host scorer
0.0, on a clip whose MPPE is ~5 cm either way. e18 agrees between the two. Recorded, not chased.

Spend to date: **$21.13 of the $30 standing budget**, nothing left running.
