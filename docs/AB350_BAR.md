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
