# V2D Track 2 — Robotic Grounding

Our entry for [NVIDIA's Video-to-Data Challenge](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/),
Track 2: drive a Unitree G1 + Dex3 to reproduce a human manipulation demonstration, scored on
the object trajectory the robot achieves. Three tiers share one clean reference and differ only
in the input; we target **Tier 2** (synthetic corruption) with **Tier 1** as insurance.

Freeze: **2026-11-04 17:00 EST.**

## The idea

Tier 2 hands you a corrupted reconstruction and scores you against the clean one, so a policy
that tracks its input perfectly still loses. Measured with the organisers' own metric code, that
ceiling is **AUC 0.4247**. Most of the gap is *recoverable before any training happens*, because
the corruption violates physics we can check: an object resting on a surface cannot be tilted
relative to it, and the release ships the floor plane and the support surfaces.

Cleaning the reference lifts the ceiling to **AUC 0.4755** with no metric regressing — and that
is the input to training, so it costs nothing at inference.

**Caveat, measured and load-bearing:** the cleaner recovers only **21–27 %** of a tilt we inject
ourselves, so it fails its own known-answer gate (`docs/GATES.md`). The scored gain is real and
reproducible; the claim that the support-contact *mechanism* is what produces it holds only for
the clips with an unambiguous flat base. Read the ceiling table as a measurement of what the
cleaned reference scores, not as a validated mechanism.

## Results

Every row is reproducible with one command (see below) and backed by a JSON file in `results/`.
Scores come from the kit's own `metric_code/track_2/*.py`, sha256-checked against
`KIT_MANIFEST.json`, on the host's scored grid of 22,989 frames.

### The Tier 2 input ceiling

| reference | AUC ↑ | SP-SR ↑ | MP-SR ↑ | MPPE cm ↓ | hand-object contact ↑ |
|---|---|---|---|---|---|
| Tier 2 as released | 0.4247 | 0.9667 | 1.0000 | 2.3605 | 0.3701 |
| floor-only cleaner | 0.4660 | 1.0000 | 1.0000 | 2.3053 | 0.3841 |
| **support-surface cleaner** (see gate caveat) | **0.4755** | **1.0000** | **1.0000** | **2.2777** | 0.3817 |
| tilt oracle *(not reachable)* | 0.6541 | 1.0000 | 1.0000 | 2.0672 | 0.4352 |
| full-orientation oracle *(not reachable)* | 0.8806 | 1.0000 | 1.0000 | 1.8698 | 0.4709 |
| clean Tier 1 reference | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5744 |

The oracles are built from the clean trajectory. **They are upper bounds, not forecasts**, and a
submission may not use them. Every figure quoted for the pipeline comes from a cleaner that sees
only Tier 2 inputs, the object mesh, the released ground plane and the released support
primitives.

Final rank is the **sum of per-metric ranks**, so MPPE counts as much as AUC. That is why "no
metric regresses" is a hard gate here and not a nicety.

### What Tier 2 actually corrupts

Measured, not assumed (`results/probe_support_root.json`):

| field | corruption |
|---|---|
| `observation.object.pose` | position 1.94 cm mean, orientation 21.4° (15.8° tilt + 11.4° yaw) |
| `observation.human.pose` | **root joint only**, 5.17° mean, 8.00° max |
| `observation.human.translation` | 1.09 cm mean |
| the other 76 human joints | **bit-identical to Tier 1** |

So the human corruption is a rigid, time-varying displacement of the whole body — and only
38.9 % of it is a constant per-clip offset.

### Two cleaners, one of which failed outright

| cleaner | reference it uses | result |
|---|---|---|
| **support contact** (object tilt) | the floor plane and the released support primitives — **external** to the noisy data | **scores better** (AUC 0.4247 → 0.4755) but **fails its known-answer gate** at 0.27 recovery |
| **foot planting** (human root) | "a planted foot does not move" — **internal** to the noisy data | **fails**: root error 5.17° → 5.58°, translation 1.09 → 2.10 cm, improved on 0 of 30 clips |

That contrast is the transferable result: **a cleaner needs an external reference, not merely a
self-consistency condition.** The foot-planting constraint pins *relative* motion, and its
reference pose is the median of the *noisy* foot positions, itself displaced by that segment's
mean corruption — so it scrubs within-segment wobble while redistributing the segment-mean error
into the root. The body's one external reference, the ground plane, is expressed in a frame with
no published transform to the body's own.

The failure is kept in the tree, with its measurement, because the negative result is the useful
part. `v2d clean-root` still runs; it is simply not in the pipeline.

### Scored metrics and the grasp signal come apart

Contact coverage is what a policy actually trains against, and it does **not** track the scored
metrics. Over the 12 episodes the support-surface extension changed, per-clip AUC gain and
contact change correlate at r = +0.481, yet the mean AUC gain is +0.0237 against a mean contact
change of −0.0060. Paired against the floor-only cleaner the whole contact difference is −0.0024
with 6 clips up, 5 down and 19 tied (sign test p = 1.000): **statistically unchanged.**

The reason is the human: a perfect *object* cleaner still leaves 38 % of the contact loss
(0.0777 of 0.2043) on the table, because the retargeted hands follow a root that is still
displaced ~7 cm against a 2 cm contact threshold. Nothing that touches only the object can reach
it. So the cleanup is adopted **for its scored gain**, and no training-signal benefit is claimed.

## Pipeline

```
released episode                      30 clips, Tier 1 (clean) + Tier 2 (corrupted)
  │
  ├─ v2d probe-supports               what the corruption is; whether the primitives are usable
  │
  ├─ v2d clean-support                Tier 2 object pose ──▶ cleaned reference (npz)
  │     --support-primitives            floor + support-surface tilt correction
  │
  ├─ v2d score                        cleaned reference ──▶ AUC / SP-SR / MP-SR / MPPE
  │     --pred npz:DIR                  via the kit's own metric code
  │
  ├─ v2d retarget                     cleaned reference ──▶ G1 motion (motion_v1/single_robot)
  │     --object-npz DIR                SOMA-X body ──▶ Pink IK ──▶ 43 joints, 2 end effectors
  │                                     writes provenance.json; validates every clip
  │
  ├─ brev/gate_launch.sh              G1 motion ──▶ FlashCHORD policy on a rented A6000
  │     --phase full                    refuses an uncleaned reference unless forced
  │     ARM_FLAGS=...                   one-variable recipe arms; see brev/arm_flags.sh
  │
  ├─ v2d pick-checkpoint               7 saved checkpoints ──▶ one per episode
  │     --cell e03=DIR                   on the noisy input reference alone, no ground truth
  │
  └─ v2d pack-tier1                   rollout ──▶ submission CSV (22,989 rows)
        --dry-run                       refuses to pack an entry from the kinematic retarget
```

Retargeting all 60 Tier-1/2 clips runs on this Jetson's CPU in about four hours at two workers,
and every clip validates at **0.0 mm** rigid-fit residual against its input — one rigid transform
explains each retargeted object trajectory exactly, which is what the submission packer has to
undo, so the pipeline is faithful end to end.

## Reproduce

```bash
./scripts/setup_jetson_env.sh            # rootless Python 3.11 via uv; no sudo, no system packages
uv pip install -e ".[dev]"
pytest                                   # known-answer tests
```

Then, with `$DATA` pointing at the released dataset and `$KIT` at the extracted submission kit:

| number | command |
|---|---|
| what Tier 2 corrupts; primitive usability | `v2d probe-supports --dataset-root $DATA` |
| the cleaned reference | `v2d clean-support --dataset-root $DATA --npz-out npz_table --support-primitives` |
| any row of the ceiling table | `v2d score --kit-root $KIT --dataset-root $DATA --pred npz:npz_table` |
| the oracles | `v2d tilt-yaw --dataset-root $DATA --npz-out npz_oracle` |
| retarget to G1 | `v2d retarget --dataset-root $DATA --object-npz npz_table --tiers tier2` |
| contact coverage | `v2d contact --arms-root ARMS --baseline-logs LOGS` |
| the root cleaner's failure | `v2d clean-root --seq-root SEQ --repo REPO --soma-root SOMA` |
| the Tier 1 package | `v2d pack-tier1 --kit-root $KIT --dataset-root $DATA --motion-root MOTION --dry-run` |
| which checkpoint to submit per episode | `v2d pick-checkpoint --cell e03=CELL --cell e09=CELL -o results/checkpoint_selection.json` |

`v2d --help` lists everything. Each command writes its JSON record under `results/`.

### The gates

Every cleaner has to pass three gates in order, with the bar fixed first (`docs/GATES.md`):
known-answer (recover noise we injected), do-no-harm (clean input stays put), then the local
score on real Tier 2.

| gate | floor-only | support-surface | bar |
|---|---|---|---|
| G1 mean recovery of an injected 12° tilt | **0.2137** | **0.2683** | ≥ 0.70 → **FAIL** |
| G1b worst clip | **−0.2758** | **−0.2758** | ≥ 0.20 → **FAIL** |
| G2 movement on clean input | 1.4264° | 1.7397° | ≤ 3.0° → pass |
| G2b position drift on clean input | 0.0000 cm | 0.0000 cm | ≤ 1e-6 cm → pass |

Recovery tracks object shape, not noise: objects with one flat base recover 0.72–0.91, objects
without one (squeegee, luggage, stroller, chair, bat) recover **negatively**. The support facet
is an argmax over hull facets against "down", and when no facet dominates, a 12° tilt makes a
different one win and the cleaner flattens the wrong face. Clips that pass the gate gain
**+0.0755** AUC against raw Tier 2; clips that fail gain **+0.0261** for reasons the mechanism
does not explain.

Per the project rule to choose on synthetic validation rather than on a Tier-1-scored number, the
cleaner is therefore not selected on its AUC. The next variant is gated per clip on synthetic
recovery and is pre-registered in `docs/GATES.md`.

## Method

Results are **pre-registered**. Predictions, thresholds and a decision rule are committed before
an experiment runs, and the rule is then applied mechanically by a script that was also committed
first:

- `docs/EXP_CLEANUP_CONTACT.md` — does cleaning object orientation restore hand-object contact?
  Verdict **PARTIAL**: 5 of 6 predictions passed, and the one that failed missed its
  sign-consistency bar by a single episode. Not upgraded.
- `docs/EXP_SUPPORT_AND_ROOT.md` — the support-surface extension and the root cleaner. **T2, T3,
  T5 passed; T1, T4, T6 failed**, and the decision rule still selected the extension, because T1
  was a prediction about magnitude and the rule is a comparison. Two amendments are recorded, both
  made on calibration data before any outcome was measured, and one prediction (R3) is marked
  **unevaluable as written** rather than quietly reinterpreted.

Three ideas were measured and killed: vertical seating (AUC 0.2786), "motionless means
supported", and left-right common-mode rejection for foot speed. Two post-hoc explanations I
proposed were tested and rejected. They are all written down, because the record of what did not
work is what makes the rest trustworthy.

## Layout

```
src/v2d/           the package: one CLI, shared geometry/dataset/supports core
  clean/           the two cleaners
  retarget/        sequence conversion, batch driver, validation, QA
  score/           the kit's metric code, contact coverage, pre-registration scorers
  submit/          submission packaging
  observe/         W&B + private Hugging Face, both optional and fail-soft
brev/              host-side gate launcher and the script that runs on the instance
docs/              scoping, the 11-day plan, the pre-registrations, the submission path
results/           every committed number
tests/             known-answer tests, including the GPU picker regression
```

## Operating rules

No paid compute without explicit approval of the specific run. Every Brev run deletes its
instance on exit, with an independent watchdog bounded by both `--max-hours` and `--max-usd`. The
GPU picker refuses any card below FlashCHORD's 46 GB floor and has no fallback default. Nothing
is published: the Hugging Face repo is private and the code refuses `private=False`. No
submission, push or email without approval.
