# STATUS — v2d (NVIDIA V2D Challenge, Track 2)
Goal: A valid Track 2 submission before the freeze (this keeps the $250 Brev credit), then the best Tier 2 rank-sum we can reach in 11 days, and a frozen released measurement whatever the rank.
Finish line: Tier 1 insurance submission accepted; Tier 2 submitted; results/ frozen; README + short tech report; every number traceable to a committed results file.
Deadline: leaderboard freeze 2026-11-04 17:00 EST (internal cutoff 12:00 EST that day)
Milestone: GO. Day 0 of 11 complete, all at $0: local scorer validated against the host's own metric code, Tier 2 ceiling re-measured, cleanup idea sized, Brev gate kit written and dry-run.
Last result: Tier 2 input tracked perfectly scores AUC 0.425 / SP-SR 0.967 / MP-SR 1.000 / MPPE 2.36 cm. That is 1st by a wide margin on the three position-like metrics (leader: 0.600 / 0.600 / 13.70) but only ~5th on AUC. The 21.4° orientation error is 15.8° support-recoverable tilt + 11.4° unrecoverable yaw; fixing only tilt lifts the ceiling to AUC 0.654, above every team on all four metrics.
Blocked: nothing on this machine. Needs Luai: (1) redeem the Brev code (captain only), (2) approve the Day-1 probe run (~$0.40), (3) one `sudo apt install python3.11 python3.11-venv` for the local retargeting path.
Next: Day 1 — stand up soma_to_g1 on CPU under Python 3.11, dry-run the submission plumbing on the kit's sample CSV, then the Brev throughput probe and the concurrent-instance quota check (gates G1/G2 in docs/PLAN_11DAY.md).
Updated: 2026-10-24 01:40

## Rules
No paid compute until Luai redeems the code and approves the specific run. No registration, submission, posting, pushing or email without his approval. Every Brev run deletes its instance on exit. Jetson: MemAvailable above 2.5 GB, 20 GB disk floor, check ~/projects/_ops/GPU_RELEASED before local GPU use.

## Paid compute used
None. $250 Brev credit granted 2026-10-24, not yet redeemed. Revoked unless a valid submission lands before Nov 4; code expires Nov 13.

## Downloads
Kit (0.7 MB) and FAQ; Track 2 tier 1+2 parquets, meshes, ground planes, support surfaces and metadata (427 MB, scratchpad only, not committed). Videos (686 MB/tier) not downloaded. Project venv at .venv (not committed).

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
- 2026-10-24: NVIDIA granted $250 in Brev credits, V2D use only, revoked without a valid submission by Nov 4. Plan compressed from 4 weeks to 11 days (docs/PLAN_11DAY.md).
