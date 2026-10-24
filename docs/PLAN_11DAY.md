# V2D Track 2: the 11-day plan

Written 2026-10-24. **Freeze: Wed 2026-11-04, 17:00 EST — 11 days.** The 4-week plan in
`SCOPING.md` §6 is superseded by this. Target is **Tier 2**, with **Tier 1 as the guaranteed
valid submission**.

## What changed since the scoping doc

Three things, and each one moves the plan.

**1. The money is real but smaller, and conditional.** NVIDIA granted **$250 in NVIDIA Brev
credits**, not the $300 of self-funded RunPod time the scoping doc costed. The grant email is
explicit: *"If your team does not submit a valid submission to at least one track prior to the
November 4, 2026 deadline, the credits will be revoked."* The code expires Nov 13 and only the
team captain can redeem it. So **a valid submission is not the goal, it is the precondition**,
and it gets bought first (Day 4) with the cheapest policies that still qualify.

**2. The field filled up.** At scoping there was one competing team in Track 2. Now
(leaderboard generated 2026-10-23 21:52 UTC):

| Tier | Teams | Leader | AUC | SP-SR | MP-SR | MPPE cm |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 19 | Chivatam | 0.830 | 0.900 | 0.833 | 6.72 |
| 2 | 13 | Xueyi Wang (FF) | 0.518 | 0.600 | 0.600 | 13.70 |
| 3 | 6 | nima arki | 0.319 | 0.400 | 0.100 | 43.41 |

The CHORD baseline now sits roughly **7th of 14 in Tier 2**, so *reproducing the baseline is a
mid-table finish*, not a foothold. Tier 3 is no longer unsolved (0.002 → 0.319 AUC), so the
"cheapest place to move ahead" argument in the scoping doc is gone. Tier 3 is dropped.

**3. The FAQ answers three of the five open questions** (`assets/v2d_challenge_2026_faqs.pdf`):

- **How metrics combine:** *"We will sum your ranking in each metric and compute final rankings
  as the lowest sum of metric rankings."* So rank is a **sum of per-metric ranks**, and MPPE
  counts as much as AUC. This is the single most useful fact we learned, and it shapes the whole
  target — see below.
- **Tier 3 alignment** may use the first frame of Tier 1 data.
- **Pretrained models and external data are allowed on all tracks.**

Still unanswered: whether one checkpoint may serve several episodes. Not needed; we train
per-episode as the kit expects.

## The target, restated around the rank-sum rule

`results/tier2_input_ceiling_kitscorer.json` scores the released Tier 2 input *as if a policy
tracked it perfectly*, using the host's own metric code. Against the current Tier 2 board:

| | AUC | SP-SR | MP-SR | MPPE cm |
| --- | ---: | ---: | ---: | ---: |
| Tier 2 input, tracked perfectly | 0.425 | **0.967** | **1.000** | **2.36** |
| Current Tier 2 leader | **0.518** | 0.600 | 0.600 | 13.70 |
| CHORD baseline | 0.374 | 0.533 | 0.067 | 20.52 |

A policy that merely tracks its own noisy input well would rank **1st in SP-SR, 1st in MP-SR and
1st in MPPE** by large margins, and about 5th in AUC. Under rank-sum that is roughly
`1+1+1+5 = 8` against the current leader's `1+2+2+2 = 7`. **Tracking fidelity, not reference
cleaning, is what the tier rewards** — and nobody is close to their ceiling: the leader's MPPE
is 13.70 cm against an input ceiling of 2.36 cm.

So the plan spends its GPU budget on **getting policies to track at all**, and treats reference
cleaning as one bounded experiment, not the thesis.

### Correction to SCOPING.md §5

The scoping doc reported the Tier 2 input ceiling AUC as **0.82**. That was a keypoint proxy.
The real metric transforms **500 sampled mesh vertices** (`v2dlb/vertices.py`: seed 0), and with
the actual meshes the ceiling is **0.425** — the proxy was wrong by a factor that reverses the
conclusion, because a 21° rotation error moves a step-ladder's far vertices metres, while the
six keypoints 5 cm from the origin barely move. The corrected claim: **the Tier 2 input is
better than every team on position-like metrics and worse than the top four on AUC.**

## What the reference cleanup idea is actually worth

Measured today, not assumed (`results/tilt_yaw_split.json`, `results/tier2_{tilt,yaw}_fixed.json`).
Support contact pins two rotational degrees of freedom — the support normal in the object's
frame — and leaves rotation about that normal free. Splitting the Tier 2 orientation error
accordingly:

- total rotation error **21.4°** = **15.8° tilt** (support-recoverable) + **11.4° yaw** (not).
- The released **ground plane is trustworthy**: in the 16 episodes where the object rests on the
  floor, the lowest mesh vertex of the *clean* trajectory sits within **±1.5 cm** of the plane.
- In the other 14 episodes the object rests 17–80 cm above the floor, on a table — and those are
  exactly the 14 episodes with non-empty `support_surface/` files. 28 of 30 episodes follow that
  rule; the 2 exceptions (vacuum, squeegee) are objects *leaning* on something.

Scoring the hypotheticals, all keeping the Tier 2 position so only orientation differs:

| Variant | AUC | SP-SR | MP-SR | MPPE cm |
| --- | ---: | ---: | ---: | ---: |
| Tier 2 input as released | 0.425 | 0.967 | 1.000 | 2.36 |
| yaw fixed, tilt left (not achievable from support) | 0.530 | 1.000 | 1.000 | 2.18 |
| **tilt fixed, yaw left (the achievable arm)** | **0.654** | 1.000 | 1.000 | 2.07 |
| both fixed (clean orientation) | 0.881 | 1.000 | 1.000 | 1.87 |

**A perfect tilt cleaner is worth +0.23 AUC at the ceiling and would put the ceiling above every
team on all four metrics.** It passes the scoping doc's kill criterion (recovering tilt removes
~10° of the 21.4°, well over the 5° bar). It is a real result and it is the ablation on Day 8–9 —
but it only raises the *ceiling*, and no team is near its ceiling, so it is second in priority
behind tracking.

**The support-surface frame problem:** `support_surface/*.usda` is Z-up `g1_retarget_world`,
while object poses are Y-up released-Track-2-world, and no transform ships with the data. The
ground plane *is* in the pose frame, so the 16 floor episodes need no alignment. The 14 table
episodes need a fitted transform — that is why the gate clips are both floor episodes, and why
the ablation starts with floor episodes only.

## Cost model

Cheapest Brev GPU meeting FlashCHORD's "48 GiB with ≥46,000 MiB free": **A6000 48 GB at
~$0.60/h**. Details, the L40S 44-GB caveat and the live-price check are in `brev/README.md`.

| Item | Day | GPU-hours | $ |
| --- | --- | ---: | ---: |
| Gate, probe + full | 1, 3 | 4–9 | 4–6 |
| Insurance: 30 short Tier 1 policies | 4 | 8–15 | 5–10 |
| Tier 2 full pass, 30 clips | 5–7 | 90–120 | 54–72 |
| Tilt-cleanup ablation, 5 clips × 2 arms × 2 seeds | 8–9 | 60–80 | 36–48 |
| Re-runs, eval-only, slack | 10–11 | 10–20 | 6–12 |
| **Total** | | **172–244** | **~$105–148** |

That leaves real headroom in $250. **Wall-clock is the binding constraint:** one 30-clip pass is
90–120 GPU-hours, i.e. 4–5 days on a single instance, so it needs **3–4 instances in parallel**.
Whether a fresh Brev org permits 4 concurrent instances is **unverified and is the first thing to
check on Day 1** — the schedule below collapses to Tier 1 only if it does not.

## The 11 days

Each day names its gate. A failed gate changes the plan that evening, not on Nov 3.

**Day 0 — Oct 24 (done, $0).** Local scorer built on the host's own metric code and validated;
Tier 2 ceiling re-measured with real AUC; tilt/yaw decomposition; ground-plane validation; Brev
gate kit and cost model; this plan.

**Day 1 — Oct 25.** *Luai redeems the credit code (only the captain can).*
- `$0` Stand up `soma_to_g1.py` on CPU. It imports no Isaac Lab or omni code — only numpy,
  torch, trimesh, viser, scipy and `soma` — so it can run here, but `py-soma-x` needs `pxr`
  (USD) and `usd-core` ships aarch64 wheels **only for cp311+** while this Jetson is on 3.10.
  Fix: `sudo apt install python3.11 python3.11-venv` (3.11.17 is already available for arm64 from
  the configured deadsnakes PPA), then a 3.11 venv with `usd-core`, `py-soma-x==0.2.1`,
  `pin`, `pin-pink==4.2.0`, torch, trimesh. `pinocchio` has cp310/cp311 aarch64 wheels, confirmed.
- `$0` Run the whole submission path offline on the kit's sample CSV: `tools/preflight.py` then
  `scripts/score_track2.py --pred csv:...`. Finds format faults now rather than on Nov 4.
- `~$0.40` Brev **probe**: `brev/gate_launch.sh --phase probe --go`. Also check the concurrent
  instance quota and that `brev search` prices match `brev/README.md`.
- **Gate G1:** projected 30-clip cost ≤ 120 A6000-hours. If not → Tier 1 only, single pass.
- **Gate G2:** `soma_to_g1.py` retargets one episode. If not by end of day → retargeting moves
  to a Brev instance and the Day 8–9 ablation is cut.

**Day 2 — Oct 26 ($0 if Day 1 worked).** Retarget all 30 episodes on CPU, parallel over cores.
Verify each output against the repo's `robot_name=g1/data.parquet` schema. This is the real
critical path: **no retargeted reference means no training at all.**

**Day 3 — Oct 27 (~$2–5).** Brev **full** gate on e24 watering_can: train 350 M steps →
`evaluate_policy.py` → `pack_submission.py` → score locally. First true per-clip score and the
first proof the whole chain works. **Gate G3:** per-clip local AUC > 0.20 and the packed CSV
passes preflight.

**Day 4 — Oct 28 (~$5–10). Buy the insurance.** Short-budget policies (10–20 M steps) for all 30
**Tier 1** clips, packed and submitted. This satisfies the credit condition — *learned policies
for all 30 trajectories* — and puts a real entry on the Tier 1 board. Uses 1 of 5 weekly Tier 1
slots. **Gate G4: until Kaggle accepts this submission, nothing else gets spent.**

**Days 5–7 — Oct 29–31 (~$54–72).** The Tier 2 pass: full-budget policies for all 30 Tier 2
clips across 3–4 parallel instances. Score locally, submit Tier 2, and record the 5 worst clips.
**Gate G5:** if policy MPPE > 20 cm, the tracker is the whole problem — cancel the ablation and
spend the rest on a second pass with a larger step budget.

**Days 8–9 — Nov 1–2 (~$36–48).** The one experiment: **tilt-projected reference vs raw**, on the
5 worst floor-resting clips, 2 seeds each, plus a smoothed-reference arm as the
"why not simpler" control (smoothing is already known not to help the *input*, so this tests
whether it helps the *policy*). This is the ablation the STANDARD demands.

**Days 10–11 — Nov 3–4 (~$6–12).** Final passes using the unlimited Nov 2–4 submission window.
Freeze `results/`, write the README and the short tech report (both required for any prize),
check every number traces to a committed results file. **Hard internal cutoff 12:00 EST Nov 4**,
five hours before the real one. Luai approves the push and the final submissions.

## What success looks like

Ranked by how much of it is in our control:

1. **A valid submission before Nov 4.** The credits survive. Bought on Day 4, not hoped for.
2. **A Tier 2 entry that beats the CHORD baseline** on all four metrics. Mid-table or better.
3. **A frozen, released measurement** — the local scorer, the corrected input ceiling, the
   tilt/yaw decomposition and the ground-plane validation — with a tech report. This is the
   deliverable that survives whatever the rank is, and it is the honest fit to
   `_ops/STANDARD.md`: one named benchmark, numbers against a named baseline, every figure
   traceable to a results file.
4. **A tier win.** Possible under rank-sum *if* the policies track anywhere near the input
   ceiling, but with 13 teams in Tier 2 and 11 days, this is the stretch, not the plan. The
   scoping doc's "20–30 % chance of a top-2 track finish" was written against a field of one
   and should be read as **well under 10 %** now.

## Standing rules

No paid compute until Luai redeems the code and approves the specific run. No registration,
submission, posting, pushing or email without his approval. Every Brev run tears its instance
down. Keep MemAvailable above 2.5 GB and 20 GB free on the Jetson; check
`~/projects/_ops/GPU_RELEASED` before any local GPU use.
