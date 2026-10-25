# STATUS — v2d (NVIDIA V2D Challenge, Track 2)

## Blocked — needs Luai, nobody else can do either

1. **Kaggle rules not accepted on the 4 Tier 1 competitions**: open each of
   `kaggle.com/c/v2d-challenge-track2-tier1-auc/rules`, `-tier1-sp-sr/rules`, `-tier1-mp-sr/rules`,
   `-tier1-mppe/rules` and click accept — Kaggle has no API for this, and `--check-entry` reported
   all four not accepted on 2026-10-24 (`tier2-auc` is already accepted, so the flow works).
2. **No `code_commit_url`**: create a GitHub remote, push the commit that produced the CSV, and if
   the repo is private invite `Nvidia-V2D-Challenge` — every submitted row must carry an openable
   `/commit/<sha>` URL and this repo has no remote, so no such URL exists yet.

Everything else for the Tier 1 entry is done or in flight: the packer, the grid check and the
local scorer are proven, so the upload is one command once both are cleared. Note the uploader
sends one CSV to all four competitions of a tier and each upload counts against every one of
their quotas even when it fails to score, so nothing unvalidated gets sent.
Goal: A valid Track 2 submission before the freeze (this keeps the $250 Brev credit), then the best Tier 2 rank-sum we can reach in 11 days, and a frozen released measurement whatever the rank.
Finish line: Tier 1 insurance submission accepted; Tier 2 submitted; results/ frozen; README + short tech report; every number traceable to a committed results file.
Deadline: leaderboard freeze 2026-11-04 17:00 EST (internal cutoff 12:00 EST that day)
Milestone: GO, and the first paid compute has run. Gate G1 passed long since (all 60 clips retargeted on CPU at 0.0 mm). **Brev is live**: three probe attempts, two real bugs found and fixed, the hardware confirmed, ~$0.22 spent of a $2 cap. The reference cleaner now has a **known-answer gate, and the original cleaner failed it** -- a gated variant that passes is built and scored. Code is one package with one CLI, ruff/format clean, 76 tests. Observability is live to W&B and a private Hugging Face repo.
Last result: **B3 ran: three training arms on the gated-cleaned Tier 2 reference, e24, 10M steps, two seeds
each.** Arm A (stock) 0.2015 AUC, arm B (+reference-noise injection) 0.2017, arm C (+bounded
object-target correction) **0.2814**. C beats stock by +0.0799, six times the pre-registered
0.014 margin -- and it is **not established**, because C's own seed spread is 0.1582, sixteen
times the 0.0100 the margin was derived from, giving its two-seed mean a standard error (0.112)
larger than the advantage itself. C_s42 scored 0.3605 and C_s43 0.2023; the second is
indistinguishable from A and B. The evaluator's own metrics order the arms identically and every
rollout has its full 748 frames with the frame check at 2e-06 cm, so it is a real effect with
unquantified variance rather than an artefact. A and B differ by 0.0003 and are the same arm at
this budget. **Also measured: this org runs at least 4 concurrent A6000s**, which turns a
126.9 GPU-hour pass from ~127 h of wall clock into ~32 and is what makes a full pass fit before
the freeze. **And a safety bug worth naming: `brev delete` has no --yes flag, so every watchdog
I had armed was a silent no-op** -- the independent protection against runaway GPU billing never
worked, and only the teardown path's fallback ever deleted anything. Fixed in all five call
sites, both watchdogs now retry through rate limits.
Blocked: nothing. Brev is logged in and B0 is approved; W&B and Hugging Face are logged in and detected. Needs Luai: (1) decide the gated-vs-ungated reference set, since that is a ranking call not a scientific one, (2) re-save the quality-gates / dev-confirm section of ~/projects/_ops/briefs/v2d_allhands.md -- it is not on disk, the file ends at the Rules bullets and nothing under _ops mentions it, so I followed the Rules section's three-gate requirement instead and did not guess at the rest.
Next: arm C is the only arm worth more compute, and it needs seeds rather than steps: at least four more at 10M to pin its variance before any ranking is claimed. Then the Tier 1 backup run, which needs Luai's separate approval. The full Tier 2 pass is now costed at 126.9 GPU-hours and $76-86, about a third of the credit, and fits in ~32 h of wall clock at 4 concurrent instances.
Updated: 2026-10-24 18:15

## Rules
No paid compute until Luai redeems the code and approves the specific run. No registration, submission, posting, pushing or email without his approval. Every Brev run deletes its instance on exit. Jetson: MemAvailable above 2.5 GB, 20 GB disk floor, check ~/projects/_ops/GPU_RELEASED before local GPU use.

## Paid compute used
~$2.7 of $250 as of 18:15 PT (4 probe attempts, a 3 h dev box, 3 quota probes): four probe attempts (~$0.35) plus a kept-alive dev box. Every
instance deleted or watchdog-bounded; `brev ls` confirmed empty between runs. Measured full-run
cost is 126.9 GPU-hours, $76-86 per 30-clip pass.

Superseded:  $250 Brev credit granted 2026-10-24, not yet redeemed. Revoked unless a valid submission lands before Nov 4; code expires Nov 13.

## Day 1 log — 2026-10-24
- **All 60 clips retargeted on CPU, zero GPU, zero dollars** (`results/retarget_all.json`). Worst rigid-fit residual across all 60 is **0.0 mm** and worst orientation residual **0.043°**, i.e. one rigid transform explains each retargeted object trajectory exactly — which is precisely what the submission packer has to undo, so the pipeline is faithful end to end. Output is `motion_v1/single_robot`, 43 robot joints, left/right_hand_palm_link end effectors. ~3–4 min per clip at ~4 frames/s per worker.
- **Environment is rootless** (`scripts/setup_jetson_env.sh`, idempotent). Python 3.11 via uv's python-build-standalone because `usd-core` ships aarch64 wheels only for cp311+; `py-soma-x --no-deps` because its `cholespy` has no aarch64 wheel and nothing on the retargeting path imports it; CPU-only torch because the default aarch64 wheel drags ~3.6 GB of CUDA 13 libraries this box's CUDA 12.6 driver cannot use; G1 meshes via GitHub's `/raw/<sha>/` endpoint, which resolves LFS where `git lfs pull` silently did not. **Everything runs under `python -I`**: the machine's global PYTHONPATH holds a PyTorch source tree that otherwise shadows the venv.
- **Validated before batching, three independent ways.** The reconstructed body's sole sits 0.5 mm from the released ground plane; the retargeter puts the object 1 mm above the plane at frame 0; and `validate_retarget.py` returns 0.0 mm / 0.03° on a single clip. Only then did the batch run.
- **Two bugs found, both mine, both fixed.** (1) Episode 11's final frame is the one invisible frame in the release — zero-filled, so a zero quaternion reads back as the identity rotation at the world origin. The converter now carries the last visible pose across it; the validator now excludes it. One such frame was dragging the global Kabsch fit to a 113 mm residual and faking a 24–36° orientation error. It is also the frame the organizers leave out of the scored grid. (2) The first batch lost 28 clips to OOM kills at 3 workers on this 15.6 GB shared box; 2 workers is the ceiling.
- **The two zero-contact clips are explained, not broken.** Of 30 Tier-1 clips exactly one has no hand contact — `rolling_luggage_left_foot_push_06`, whose task name says the object is pushed with a **foot**. Every hand-manipulated clip sits at 0.35–0.86. The other flag, `vacuum_left_lean_02`, has 0.485 contact in Tier 1 and 0.003 in Tier 2: the flag is the noise, not the retarget.
- **New result: Tier 2's injected noise destroys 36 % of the hand–object contact signal** (`results/retarget_qa.json`). Paired across all 30 episodes, mean contact coverage falls 0.574 → 0.370; 26 of 30 clips degrade and 8 lose more than half. The same clip retargeted from the clean and noisy trajectories differs only by that noise, so this is a direct measure of what Tier 2 costs the grasp signal the policy trains on. It is independent evidence for the cleanup idea, which so far rested only on the scoring ladder.
- **Testable prediction, CPU-only and free:** if the 15.8° tilt component is the recoverable part, retargeting the `tilt_fixed/` references should restore most of that lost contact, while `yaw_fixed/` should not. That is the next experiment and it needs no GPU and no money.
- **Grounding is uniform and safe.** All 60 clips ground off the reconstructed plane, applied ground correction never exceeds the 0.01 m cap, plane-leveling tilt reaches 11.8°, and the worst sole Z is +1.4 mm — the robot never ends up standing through the floor.

## Day 1, second half — the reference cleanup, pre-registered and shipped
- **Predictions were committed before any arm ran** (`docs/EXP_CLEANUP_CONTACT.md`, commit `9f19399`), along with the decision rule and the scoring script (`c2f67ef`) that applies it mechanically. One design flaw was caught and the doc amended *before* running (`c039838`): the planned ceiling arm also removed Tier 2's 1.94 cm position noise, which would have depressed recovery by construction, so a `rot_clean` arm (Tier 2 position + clean orientation) was added as the honest denominator.
- **What Tier 2 actually perturbs, measured before predicting.** Object pose; the human **root joint only** (mean 5.21°, max 8.00°); human translation (mean 1.09 cm). Identity, scale, bone lengths and **all 76 non-root joints are bit-identical**. So the human half of the noise is a rigid root transform displacing the hands ~7 cm against a 2 cm contact threshold — which is why it costs so much contact. Also: 44 % of the orientation error is a fixed per-clip offset, and the position error is nearly constant within a clip (0.73 cm std vs 1.94 cm mean).
- **Four arms, 30 clips each, all at 0.0 mm.** Contact coverage: `tier1` 0.5744, `object_clean` 0.4967, `rot_clean` 0.4709, **`tilt_fixed` 0.4352**, `yaw_fixed` 0.3825, `tier2` 0.3701. Tilt recovers **0.646** of the orientation ceiling and yaw **0.123** — **0.00412 contact per degree removed against 0.00109**, 3.8x more efficient for an angle only 1.39x larger. Tilt is the recoverable component, as the support-plane argument predicted.
- **Verdict PARTIAL, by the rule as written.** 5 of 6 predictions pass; P2 required mean gain >= +0.03 **and** sign consistency in >= 20/30 and got **+0.0527 with 19/30**. The rule was not re-read to upgrade it. A tidy post-hoc explanation for the 11 flips ("yaw wins where yaw is the bigger error") was tested and **rejected**: true at group level, but contact gain correlates with degrees removed at only r = +0.107, against r = +0.848 with how much the full cleanup can restore on that clip. Per-clip recovery is clip geometry, not angular magnitude; the mean is the honest unit.
- **The deployable cleaner** (`scripts/clean_support_contact.py`) detects rest segments on a smoothed trajectory, fits a support height per segment, keeps only floor-level segments, picks one support hull facet, and applies the minimal rotation onto the support normal — a pure tilt correction with yaw untouched, Slerped across moving frames. It recovers real furniture heights from noisy data alone (`traffic_cone_ground_table` reads -7 cm then 74.8 cm). Scored gain above; contact gain only +0.0141, **below the `yaw_fixed` control's ceiling and not reliably positive per clip**, so the scored gain is the only claim made for it. The 5 clips it declines to touch show contact change of exactly 0.0000, which is how we know the measurement is clean.
- **Three ideas measured and killed.** `--seat-vertical` (interpolating a support height across unsupported frames) craters everything: AUC 0.2786, MPPE 18.94 cm — off by default. A "motionless means supported" rest test gave `watering_can` seven bogus segments 13-14 cm in the air while pouring; the floor-level gate fixed it. And the floor gate itself was initially backwards, rejecting exactly the penetrating frames worth correcting (`traffic_cone` at -5 to -7.6 cm; a 20° error on an 86 cm box buries a corner ~15 cm).
- **Wired in, not just measured.** `retarget_all.py` writes `provenance.json` recording the object source; `brev/gate_launch.sh --phase full` **refuses a raw-reference parquet or a missing provenance file** (verified all three ways), overridable only by an explicit `--allow-raw-reference`. The gate cannot silently train on uncleaned references.

## Day 1, third phase -- Brev goes live, and the cleaner gets gated

### Paid compute: $0.22 of $2, three attempts, two bugs
- **The GPU picker was broken and would have burned the budget.** The dry run proposed
  `paperspace_V100` at **$2.81/h for a 16 GB card** on a job needing 48 GB. The picker was awk
  guessing which column was which: it read `TARGET_DISK` (100) as the VRAM and `CAPABILITY` (8.6)
  as the hourly price, so a V100 sorted ahead of an A6000 on a number that was not a price,
  against a floor check on a number that was not VRAM. Replaced with a header-offset parser
  (`src/v2d/brev.py`, 17 tests). Column *offsets* matter, not tokens: 12 of 343 rows name the GPU
  "RTX PRO Server 6000", and splitting on whitespace shifts the price on exactly the rows
  describing the biggest cards. No fallback default any more -- guessing is what broke it.
- **Attempt 1 died on readiness.** The wait loop was `brev ls | grep "$NAME.*RUNNING"`, and brev
  reports RUNNING while the SHELL column still says NOT READY, so the payload went into a closed
  port. ~7 min, ~$0.08. Now waits on the SHELL column and proves it with a real `ssh true`;
  predicate in `v2d.brev.is_shell_ready` with tests built from the rows observed that day.
- **Attempt 2 reached the hardware, which is the half that matters.** `NVIDIA RTX A6000, driver
  580.126.09, 48,539 MiB free` -- clears FlashCHORD's 46,000 MiB floor and its >=580 driver
  requirement -- docker usable, image built in **229 s**. Then every container call failed with
  `the input device is not a TTY`, because the repo's `docker/run.sh exec` passes `-it` and stdin
  is never a terminal under `ssh 'bash -s'`. ~12 min, ~$0.136, logged to W&B.
- **A hard dollar cap now exists.** `--max-usd` aborts before creating if the estimate exceeds it,
  and bounds the watchdog by cap/hourly as well as by `--max-hours`, so a hung run cannot overspend
  even if the launcher dies. Verified: `--max-usd 0.10` aborts at exit 3 having created nothing.
- Every attempt deleted its instance on exit; `brev ls` confirmed empty between runs.

### Observability, live
W&B project **v2d-track2** (run `tazofyiy` carries the probe) and the private repo
**luaia/v2d-track2**, created `private=True` and verified by reading it back. Credentials are
*discovered*, not created: this machine has W&B in `~/.netrc` and Hugging Face in its cached
token rather than in environment variables, so an env-only check would have reported "not
configured" on a ready machine. Nothing logs or copies a token, `open_store` raises on
`private=False`, and every method is a working no-op without a credential, because GPU minutes
are the expensive thing and telemetry must never break a run. The container runs
`WANDB_MODE=disabled`: a rented VM is not a place to put a token, so the host logs from the files
copied back.

### Code quality
`scripts/` held 19 standalone tools, each with its own argparse and its own copy of kabsch, the
moving average, the quaternion reorderings, the parquet readers and the contact parsing -- they
could not import each other under `python -I`. Now one `v2d` CLI with 18 stages over a shared
core, **ruff check and format clean**, **76 tests**, and a README with the pipeline, the ceiling
table and a command per row. Verified against the committed science, not just linted: the cleaner
reproduces both variants' npz **bit-for-bit** across 30 episodes, the scorer reproduces
0.4246950476 / 0.4754869389 / 1.0 to the tenth decimal, and the contact module reproduces the
whole ladder and its sign test. Linting surfaced four genuine defects, including a `pool.map`
lambda closing over a loop variable.

**One self-inflicted cost:** moving files into the package while a retarget was running killed 11
of 30 clips, because the batch driver resolved its child stages by path at spawn time. Child
stages are now spawned as `python -m v2d.retarget.sequences`, and the 11 clips were re-run.

## Downloads
Kit (0.7 MB) and FAQ; Track 2 tier 1+2 parquets, meshes, ground planes, support surfaces and metadata (427 MB, scratchpad only, not committed). Videos (686 MB/tier) not downloaded. Project venv at .venv311 and the 60 retargeted motion parquets (scratchpad, ~270 MB) are not committed; `scripts/setup_jetson_env.sh` plus `scripts/retarget_all.py` rebuild both from scratch.

## Day 0 log — 2026-10-24
- **Scorer.** `scripts/score_track2.py` wraps the kit's own `metric_code/track_2/*.py` after checking each file's sha256 against `KIT_MANIFEST.json`, so local numbers are the host's by construction. Validated: perfect tracking → AUC 1.0 / SP-SR 1.0 / MP-SR 1.0 / MPPE 0.0; a 2 cm translation → MPPE exactly 2.0000 cm; the MP-SR 30° bar trips between 29° and 31°; 29° of yaw fails SP-SR, confirming SP-SR's 0.5 rad is *tighter* on orientation than MP-SR's 30° while far looser on position. It reproduces the previously committed SP-SR/MP-SR/MPPE to every digit.
- **AUC is exactly reproducible after all**, which the scoping doc had wrong: the kit ships the evaluator's vertex-sampling convention in `v2dlb/vertices.py` (500 vertices, seed 0). The host samples the simulation mesh instead of the released `.glb`, so expect disagreement around the 4th decimal.
- **Correction:** the Tier 2 input ceiling AUC is 0.425, not the 0.82 keypoint proxy in SCOPING.md §5. Noted in that section.
- **Submission grid.** The released data has 22,990 frames but the scored grid has 22,989: **episode 11's last frame (f000900) is absent from the kit's sample submission**, and the uploader checks rows against that sample. Open question 5 answered, and it would have failed an upload. The scorer defaults to the sample's grid.
- **FAQ answers 3 of the 5 open questions.** Final rank is the **sum of per-metric ranks** (lowest wins), so MPPE counts as much as AUC — this is what makes the Tier 2 target reachable. Tier 3 may use Tier 1's first frame for alignment. Pretrained models and external data are allowed everywhere.
- **Ground plane validated.** In the 16 floor-resting episodes the clean trajectory's lowest mesh vertex sits within ±1.5 cm of the released plane. The other 14 episodes rest 17–80 cm up, on tables, and are exactly the 14 with non-empty `support_surface/` files (28/30 agree; the 2 exceptions are *leaning* objects). Support surfaces are Cube/Cylinder primitives in Z-up `g1_retarget_world` while poses are Y-up, and no transform ships — so the 14 table episodes need a fitted frame alignment.
- **Field check.** Track 2 went from 1 competing team at scoping to 19/13/6 across tiers. The CHORD baseline is now ~7th of 14 in Tier 2, and Tier 3 is no longer unsolved (0.002 → 0.319 AUC), so Tier 3 is dropped. Top-2 odds are well under the scoping doc's 20–30 %.
- **Retargeting is the critical path, and it is feasible here.** `soma_to_g1.py` imports no Isaac Lab or omni code. `py-soma-x` and `pin-pink` are pure-Python and `pinocchio` has aarch64 wheels, all confirmed installed; the blocker is `pxr`, since `usd-core` ships aarch64 only for cp311+ and this box is on 3.10. python3.11.17 is already available for arm64 from the configured deadsnakes PPA.
- **Brev.** `brev/` holds a two-phase gate (cheap throughput probe, then one full clip), dry-run by default, deleting its instance on every exit path with an independent watchdog. Cheapest card meeting the ≥46,000 MiB floor is the **A6000 48 GB at ~$0.60/h**; NVIDIA's own docs list the L40S at 44 GB, which would fail that floor, so the A6000 is preferred on both price and certainty. Budget: a 30-clip pass is 90–120 GPU-hours ≈ $54–72, and the whole plan is ~$105–148 of $250. Wall-clock, not money, is binding: a pass needs 3–4 parallel instances, and that quota is unverified.

## History
- 2026-10-06: scoping complete, conditional GO (docs/SCOPING.md).
- 2026-10-07: Luai emailed v2d_challenge@nvidia.com about credits and the rules questions.
- 2026-10-24: all 60 Tier-1/2 clips retargeted to G1 on CPU at $0; Tier 2 noise measured to cost 36 % of hand-object contact.
- 2026-10-24: B3 on e24: arm C (+bounded target correction) means 0.2814 AUC against stock's 0.2015, but its own seed spread is 0.1582, so not established; needs more seeds.
- 2026-10-24: org runs at least 4 concurrent A6000s, so a 30-clip pass is ~32 h wall clock.
- 2026-10-24: `brev delete --yes` does not exist; every watchdog was a no-op until fixed. No instance was ever actually orphaned, but only by luck of the teardown fallback.
- 2026-10-24: B0 MEASURED on an A6000: 22,977 env steps/s, 40,156 MiB peak, 4.23 h/clip, 126.9 GPU-hours for 30 clips ($76-86) -- above the scoping estimate.
- 2026-10-24: ten Brev bugs found and fixed, four on a kept-alive dev box; git-lfs absent from the image (no training could ever start) and scene.support crashing 16 of 30 clips were the two that would each have destroyed a full run.
- 2026-10-24: both root cleaners dead; the floor pins foot height, not foot tilt (clean sole sits 7-14 deg off the plane).
- 2026-10-24: first paid compute. Brev probe attempts 1-3; GPU picker, instance-readiness and container-TTY bugs found and fixed; A6000 with driver 580 and 48.5 GB confirmed; ~$0.22 spent.
- 2026-10-24: known-answer gate added; BOTH cleaners failed it (0.21/0.27 vs 0.70), diagnosed to facet selection; a per-clip gated variant passes at 0.7498 recovery and AUC 0.4423.
- 2026-10-24: reference cleanup pre-registered, run (PARTIAL), and shipped; Tier 2 input ceiling AUC 0.4247 -> 0.4660 with no metric regressing, and the Brev gate now refuses an uncleaned reference.
- 2026-10-24: NVIDIA granted $250 in Brev credits, V2D use only, revoked without a valid submission by Nov 4. Plan compressed from 4 weeks to 11 days (docs/PLAN_11DAY.md).
