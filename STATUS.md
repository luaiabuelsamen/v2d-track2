# STATUS — v2d (NVIDIA V2D Challenge, Track 2)

## All three tiers are on the board, 2026-10-29 15:52 UTC

| board | rank | us | 1st | gap |
|---|---|---|---|---|
| tier1-auc | 12/23 | 0.52687 | 0.89354 | 0.367 |
| tier1-sp-sr | 12/22 | 0.66666 | 0.96666 | 0.300 |
| tier1-mp-sr | 11/22 | 0.50000 | 0.93333 | 0.433 |
| tier1-mppe | 11/22 | 13.90 cm | 2.78 | 11.1 cm |
| tier2-auc | 5/18 | 0.46522 | 0.56768 | 0.102 |
| tier2-sp-sr | 5/18 | 0.66666 | 0.80000 | 0.133 |
| tier2-mp-sr | 5/18 | 0.46666 | 0.80000 | 0.333 |
| tier2-mppe | **4/18** | 14.13 cm | 9.28 | 4.8 cm |
| tier3-auc | **3/10** | 0.25059 | 0.33311 | 0.083 |
| tier3-sp-sr | 7/10 | 0.16666 | 0.40000 | 0.233 |
| tier3-mp-sr | 6/10 | 0.03333 | 0.10000 | 0.067 |
| tier3-mppe | 6/10 | 54.44 cm | 31.34 | 23.1 cm |

Tier 1 went 0.46523 → **0.52687** today, past CHORD's 0.51038; Tier 3 went from **no entry at all**
to 3rd on AUC. Tier 2 is unchanged because the board already holds our best Tier 2 file.

**`docs/TO_THE_TOP.md` is the costed answer to "what would first place take".** The short version:
our nineteen best Tier 1 clips average 0.7228 while the leader's whole 30-clip entry averages
0.89354, so lifting all eleven weak clips to 0.80 — better than our current average — reaches
0.751 and about 5th. First place needs a better per-clip tracker, not more credit. What money does
buy: the eleven weak Tier 1 clips full-length on clean references, ~$36, for 12th → 5th-8th.

Every public score agrees with the local kit score to five decimals, so the local scorer is a
faithful stand-in and nothing has to be guessed from the board. Receipts and per-board submission
ids are in `results/tier1_submission_v2.json` and `results/tier3_submission_v2.json`; the code
commit named by all 22,989 rows of each entry is `6eb90dc`, pushed and resolvable on `origin/main`.

Tier 1 moved because of warm-start fine-tuning (`docs/T1_STRONG4.md`, `docs/WARM_START.md`): 11 of
23 fine-tuned clips adopted under a rule that requires AUC to improve and forbids any other board
from falling. That is legal on Tier 1 alone, because `config/tracks.json` names the permitted
multiview trajectory as *the configured scoring reference*; on Tier 2 and Tier 3 selection reads
the input only. Tier 1 fine-tuning is now **closed**: every clip above the 0.35 line where
adoption happens has been run, and all seven never run sit below 0.30, where 0 of 4 have ever
adopted.

**Still open for the Nov 4 2026 5:00 p.m. EST freeze:** accept the `Nvidia-V2D-Challenge` GitHub
invite, confirm the registration form, and pick the final two submissions per board. One Tier 3
box (`v2d-t3s6-a`, clips e06 e12 e17) is training on the last ~$5 of Brev credit; if it adopts,
Tier 3 gets a second upload.

## Live, 2026-10-29 10:20 UTC: the Tier 1 fine-tune is working

Tier 1 is the one tier where we sit **below** the CHORD baseline, and the warm-start fine-tune is
closing it. Three of nine scored clips adopt under the registered no-regression rule:

| | AUC | note |
|---|---|---|
| shipped Tier 1 entry (on the boards) | 0.46523 | the Tier 2 file, byte-identical |
| after adopting e00, e12, e24 | **0.48202** | `results/t1_finetune_adoption.json` |
| CHORD Tier 1 baseline | 0.51038 | needs +0.0284 more |

e00 **0.5178 → 0.8654**, e12 **0.7904 → 0.8839**, e24 **0.8942 → 0.9567**, each also improving
MPPE and holding every success gate — not the AUC-for-MPPE trade that killed the per-clip-best
entry. Ten clips are still training on four boxes; at the adopted clips' average gain of +0.168,
five more adoptions would land Tier 1 at about 0.510.

**Which clips it helps is predictable.** Every clip that gained ships its *final* checkpoint; every
clip where z-drop had picked an earlier one lost, because the fine-tune re-runs exactly the late
training stage that selection had rejected (e20 0.715 → 0.480). The three live Tier 1 boxes were
re-queued onto final-checkpoint clips only, dropping 11 predicted-loss clips and about $4.80 that
could not have changed the entry — adoption rejects a regression, so a predicted loss contributes
nothing either way. See `docs/T1_WARMSTART.md`.

**Tier 3 fine-tune, 7 of 10 clips:** pass 1 0.37103 → 0.40060 raw → **0.44226 after adoption**,
clearing its registered 0.3395 bar. The input-only rule adopted the four largest true gains and
rejected the only serious loss while reading no ground truth — the first direct evidence that the
rule we are *required* to use on Tier 2 and Tier 3 is selective, not merely safe.

Spend $14.37 of the $30.38 credit; four boxes live.

## Where the last free 0.1 of score is: Tier 3, which we have never submitted

The final score is a weighted combination of all three tiers, and our Tier 3 is **nothing at all**.
That is the largest gain still available, and most of it is already paid for.

| tier | submitted | AUC | note |
|---|---|---|---|
| Tier 2 | yes | **0.46523** | the wave-2 entry |
| Tier 1 | yes | **0.46523** | the same file; byte-identical, Tier 1 has no input constraint |
| Tier 3 | **no** | — | reconstruction done; needs a policy rollout for honest provenance |

All twelve Track 2 boards are joined, including all four Tier 3, so the upload path is open.

**Consolidated review page** (scores against both baselines, the 30-clip table, the eight figures,
and the costed plan for the remaining credit): https://claude.ai/artifact/VciEvgt14b74q4UC7EQxGx —
generated by `tools/review/build_review_page.py` from the result records, so it reproduces the
published means 0.39076 and 0.46522 exactly rather than restating them.

### Tier 3, reconstructed (2026-10-28)

All 30 clips reconstructed from the released video with NVIDIA's own modules (geocalib → moge →
grounding_dino → a prompt converter we wrote → sam2 → foundation_pose), then placed in the released
world by **one** rigid transform taken from frame 0, which is what FAQ PR 207 permits. Frame-0
residual is 0.0000 cm and 0.000° on all 30, which is also what proves the quaternion convention is
right: `Transform3d` is imported from upstream rather than reimplemented.

Scored raw against truth: **AUC 0.228 / SP-SR 0.333 / MP-SR 0.033 / MPPE 66.6 cm** over the full
22,989-frame grid. Quality is bimodal, not uniformly poor: 10 clips track to under 5° and 16 cm
(e00 e08 e18 e20 e21 e22 e23 e24 e26 e29); 20 lose the object.

**It cannot be submitted as it stands.** The kit's packer tolerates an empty `checkpoint_sha256`
but requires a non-empty `eval_code_sha256` (`tools/pack_submission.py:420-445`), and a
reconstruction is not an evaluator run, so there is no honest value for that field. The tier's own
note says to drive a policy from the reconstruction, which is both the rules-clean route and the
one that supplies real provenance. Record: `results/t3_reconstruction.json`.

Rules check, settled rather than assumed: Tier 3's configured `data_root` (`track_2/baseline`) is
not released to participants — the hub has only `tier_1_multiview_caption` and
`tier_2_synthetic_noise`. The support USDAs in those two are **byte-identical**, so the scene
assets are tier-invariant released data; what Tier 3 withholds is the *trajectory*, and ours came
from video.

### Clean Tier 1 input makes these policies worse (2026-10-28)

Tier 1 permits the clean multi-view trajectory, so all 30 shipped actors were re-evaluated on it.
Mean AUC **fell 0.46523 → 0.28937**, SP-SR 0.667 → 0.138, and only e06 and e28 improved. Frame
checks were at or below 1.7e-05 cm, so this is not a conversion artefact: each actor learned to
track one specific trajectory, so a cleaner but different one is a distribution shift. **No Tier 1
resubmission follows**, and the right way to use a new reference is to fine-tune onto it
(`brev/patch_warm_start.py`), not to swap it cold. Record: `results/t1_clean_input_eval.json`.
Still owed: a 2-clip Tier 2-input control through the same eval image, to rule out the image.

### Eligibility

Done: valid Tier 1 and Tier 2 submissions; `LICENSE` (Apache-2.0, as the kit's rules require of a
winner); `docs/TECHNICAL_REPORT.md`; all twelve boards joined.

**Needs Luai, not me:**

1. **Accept the `Nvidia-V2D-Challenge` GitHub invite**, pending since 2026-10-25. Until it is
   accepted the organisers cannot open the commit URL that every one of the 22,989 rows quotes.
2. **One push.** `origin/main` is at `0260cd8`, nineteen commits behind local `HEAD`, so none of
   the Tier 3 code is on the remote. The audit checks a commit URL's *shape*, not whether it
   resolves, so an unpushed sha would validate locally and 404 for a judge. The Tier 3 entry is
   therefore packed and kit-validated against local `HEAD` but is **not submittable until that
   commit is pushed** — it is one command and his word, not a decision I should make for him.
3. **Confirm the registration form** was submitted.
4. **Select the final two submissions** before the freeze — `max_final_submissions: 2`, and the
   Kaggle API exposes no "selected" flag, so it is a UI action. Freeze: **November 4 2026,
   5:00 p.m. EST**.

A Tier 3 submission is the remaining score item and is in progress.


## Submitted — the wave-2 entry is on all eight Track 2 leaderboards

**Uploaded 2026-10-28, seven days before the freeze: one upload per tier, both tiers scored, every
public score matching what was measured here.**

| board | submission | public | board before this upload |
|---|---|---|---|
| `tier2-auc` | 57742766 | **0.46522** | 0.39076 |
| `tier2-sp-sr` | 57742768 | **0.66666** | 0.56666 |
| `tier2-mp-sr` | 57742770 | **0.46666** | 0.16666 |
| `tier2-mppe` | 57742773 | **14.12760** | 17.35839 |
| `tier1-auc` | 57742805 | **0.46522** | 0.13430 |
| `tier1-sp-sr` | 57742806 | **0.66666** | 0.0 |
| `tier1-mp-sr` | 57742808 | **0.46666** | 0.0 |
| `tier1-mppe` | 57742811 | **14.12760** | 74.20 |

Three of four metrics match the local score exactly to the five decimals the board prints; Tier 2
and Tier 1 AUC come back 0.46522 where ours truncates to 0.46523, one unit in the fifth decimal.
Both tiers carry the same 22,989 rows — the kit CSV has no tier field, so the files are
byte-identical (`9997634267c66726961720f30724756389b50ee31f1f0040751364f2fb9b4305`) and the tier is
only which competition they went to. On Tier 1, which has no input constraint, that is worth
**+0.33 AUC** over the 409,600-step insurance entry it replaces, and MP-SR goes 0 → 14 clips of 30.

Against the CHORD baseline's published Tier 2 board average (0.374 / 0.533 / 0.067 / 20.5 cm):
ahead on all four, and MP-SR is 7× it. Four submissions left today on each of the eight.

Record: `results/wave2_submission.json` (receipts, per-board local-vs-public, ships by arm).
Every row quotes `0260cd8ce5c81c9e4280e2ec5b34bba59609f98b`, which is `origin/main`.

### How the entry was assembled

30 of 30 clips decided under the registered z-drop pick plus input-only adoption — **no ground
truth was read to choose anything that shipped**. Ships by arm: A 16, A re-rolled 1, first wave 7,
B1 2, B2 2, B4 1, B4x 1. 30 checkpoint digests, all 30 distinct and all recomputed from the
surviving safetensors and matched; 30/30 frame counts matching the pre-reboot record.

e29 is a **re-roll, not a recovery**: the 2026-10-28 reboot wiped the scratchpad and that one
rollout existed nowhere else, so it was re-run on arm A's own recipe and came back at kit AUC
0.6270 against the lost rollout's 0.5638 — luck, recorded as luck in
`docs/RECOVERY_2026-10-28.md` and in the entry record's `lost_in_the_reboot` field. A29 cost $2.73.

The selection rule's measured price: six clips hold a rollout that kit-scores higher than the one
that shipped, rejected for sitting further from the noisy input — 0.0103 of mean AUC, the cost of a
rule that reads no ground truth.

The packed entry, both CSVs, both pack records, the restage record and both Kaggle receipts are on
the home disk at `/home/jetson3/projects/v2d-entry-wave2`, which is what the reboot could not
reach. No instances are running; `brev ls` is empty.

## Superseded — the first-wave entry, on the boards 2026-10-27 until wave 2 replaced it

**The Tier 2 entry is on all four Tier 2 leaderboards and scored, 2026-10-27 05:18 UTC, eight days
before the freeze.** The 30-clip combined run, one upload, four competitions:

| leaderboard | submission | public score | our local score |
|---|---|---|---|
| `tier2-auc` | 57690108 | **0.39076** | 0.39076336 |
| `tier2-sp-sr` | 57690112 | **0.56666** | 0.56666667 |
| `tier2-mp-sr` | 57690116 | **0.16666** | 0.16666667 |
| `tier2-mppe` | 57690120 | **17.35839** | 17.35839727 |

Ahead of the CHORD baseline's published Tier 2 board average on all four (0.374 / 0.533 / 0.067 /
20.5 cm), most clearly on MP-SR at 2.5x. CSV sha256 `3a410393…`, 22,989 rows over 30 episodes, one
z-drop-selected checkpoint per episode and 30 distinct checkpoint digests; `main` pushed at
`e8ffe8f`, which contains the `189773f` in the entry's own provenance rows; four of five daily
submissions left on each competition. Record: `results/tier2_submission.json`,
`results/tier2_entry.json`.

**The host scorer is now proven twice, on both tiers, to five decimals.** Every later decision can
be made locally without spending an upload to find out where we stand.

It is a shipping run, not an attribution run: START, ORI, FING and CLEAN moved together, so the
score cannot be split between them. Twelve clips are still under 0.30 AUC and carry the whole
deficit — `docs/SECOND_WAVE.md` triages them into five failure modes and prices the one-variable
tests, $30 for all four at one seed against $141.78 left of the $230.

## The Tier 1 insurance entry — the credit condition was met here

**The Tier 1 entry is on all four Tier 1 leaderboards and scored, 2026-10-24, eleven days before
the freeze.** One upload, four competitions, every one COMPLETE:

| leaderboard | submission | public score | our local score |
|---|---|---|---|
| `tier1-auc` | 57617131 | 0.13434 | 0.13434706 |
| `tier1-sp-sr` | 57617135 | 0.00000 | 0.0 |
| `tier1-mp-sr` | 57617139 | 0.00000 | 0.0 |
| `tier1-mppe` | 57617142 | 74.19795 | 74.1979545 |

Both blockers are cleared: `origin` is `github.com/luaiabuelsamen/v2d-track2` (private,
`Nvidia-V2D-Challenge` invited) with main pushed at `517ad99`, and Luai accepted the four rule
sets. CSV sha256 `e41271d4…`, 22,989 rows over 30 episodes, four of five daily submissions still
left on each competition.

**The leaderboard reproduces all four local numbers to five decimals.** That is the result worth
keeping from today: from here a local score *is* a leaderboard score, so Tier 2 choices can be
made on the host scorer without spending an upload to find out. It is also not a competitive
entry -- 409,600 env steps per clip is about 1/850th of the full recipe, both success rates are
exactly zero -- and that is fine, because it is insurance.

Goal: A valid Track 2 submission before the freeze (this keeps the $250 Brev credit), then the best Tier 2 rank-sum we can reach in 11 days, and a frozen released measurement whatever the rank.
Finish line: ~~Tier 1 insurance submission accepted~~; ~~Tier 2 submitted~~; results/ frozen; README + short tech report; every number traceable to a committed results file.
Deadline: leaderboard freeze 2026-11-04 17:00 EST (internal cutoff 12:00 EST that day)
Milestone: **both tiers are submitted and scored; the competitive entry is Tier 2 at AUC 0.39076, ahead of the published baseline on all four metrics.** Gate G1 passed long since (all 60 clips retargeted on CPU at 0.0 mm). **Brev is live**: three probe attempts, two real bugs found and fixed, the hardware confirmed, ~$0.22 spent of a $2 cap. The reference cleaner now has a **known-answer gate, and the original cleaner failed it** -- a gated variant that passes is built and scored. Code is one package with one CLI, ruff/format clean, 76 tests. Observability is live to W&B and a private Hugging Face repo.
Done (2026-10-27 01:57 UTC): **the 30-clip Tier 2 run with the combined recipe finished, all 30
clips trained to 349,999,104 steps, pulled, kit-scored and selected** --
`results/t2_combined_run_record.json`, `docs/T2_COMBINED_RUN.md`.

| | our 30 clips | CHORD board average |
|---|---|---|
| mean kit AUC | **0.3908** | 0.374 |
| SP-SR | **0.567** | 0.533 |
| MP-SR | **0.167** | 0.067 |
| mean MPPE | **17.36 cm** | 20.5 cm |

**The pre-registered four-clip bar passed**: mean kit AUC on e03/e09/e18/e24 went 0.36635 (stock)
to **0.59932**, and the count of clips with no drop-free checkpoint fell 1 -> 0 rather than rising
(`results/t2_combined_bar_read.json`). It is a shipping run, not an attribution run -- START, ORI,
FING and CLEAN moved together, so the gain cannot be split between them.

Cost **$88.22** of the $230 authorized, 125.0 GPU-h over 11 A6000 boxes, against a $105.67
projection. Every instance deleted; confirmed by five `brev ls` reads over 50 s. Three clips
(e04, e17, e25) have no drop-free checkpoint, and on six failing clips the z-drop rule picked an
early checkpoint (49M-149M of 350M) because the later ones register a drop against the noisy input
reference -- that is the registered rule's cost on the tail, measured and left alone. z-drop is the
submission rule and is `v2d pick-checkpoint`'s default.

Last result (2026-10-26): **step 1 re-measured with the support fix -- the AUC line is crossed,
the pipeline is not.** RAW e03 and e09 were relaunched with the per-clip `scene.support` rule and ran
the full 349,999,104 steps. The 3-clip RAW mean AUC is **0.1345** against the 0.187 stop line, while
mean MPPE is 33.20 cm (line 41 cm, was 55.20 cm broken) and SP-SR is no longer zero on all three.
The AUC line was written to catch a broken pipeline and is not read as one: the object starts on its
support surface and is tracked to ~1 cm over the opening 50 frames, error first passes 10 cm at frame
44-376 of clips 781 and 870 frames long, each cell now shows 7 distinct AUC values across 7
checkpoints (the broken run showed 1), and e09's final checkpoint holds and places the bowl for the
whole clip (SP-SR 1.0, no termination, MPPE 5.13 cm). The remaining failures are mid-episode drops --
a policy problem, not plumbing. Host scores at 350M: e03 AUC 0.0602 / MPPE 89.78 cm, e09 AUC 0.2409 /
MPPE 5.31 cm, against e18 0.3826 / 4.17 cm and e24 0.4500 / 4.16 cm
(`results/ab350_raw_baseline.json`, addendum in `docs/AB350_BAR.md`). **The 30-clip Tier 2 run stays
NOT approved**, and no CLEAN arm has been run, so nothing here compares RAW against CLEAN.
**Spend: $21.13 of the $30 standing budget, nothing left running** (`brev ls` empty).

Last result: **the Tier 1 fallback entry exists and is scored, and arm C failed its own
pre-registered bar.** Thirty short-budget policies (409,600 steps each) trained and evaluated on
one A6000 in under two hours, all 30 rollouts came back byte-verified, and the kit's own packer
and validator both pass on them: 30 episodes, 22,989 rows, **AUC 0.1343, SP-SR 0.0000, MP-SR
0.0000, MPPE 74.198 cm** (`results/tier1_entry.json`). The kit recomputes each episode's AUC from
the packed CSV and agrees with the evaluator's own metrics JSON to the fourth decimal, so the
export and the frame conversion are right; the entry is weak because it is 409,600 steps per clip,
not because anything is broken. It is one command from upload and that command is blocked on the
two items at the top of this file. **Arm C does not ship.** Four confirmation seeds gave an escape
rate of 2/4 against the pre-registered >=3/4 (`docs/SWEEP_BAR.md`), Fisher one-sided 0.214; over
all six C seeds it is 3/6 at 0.167. C's mean AUC 0.2666 does sit above the A/B cluster's 0.2016,
but its SD is 0.0821 against the cluster's 0.0013 (both population; C's sample SD is
0.0900), which is exactly why the bar was written on the
escape rate and not the mean. **C_s46 scored 0.1821, below every A and B run**, so the correction
is bimodal in both directions rather than a free option. **And a finding that costs us a shortcut:
the evaluator's own AUC does not track the clean-reference score across C's seeds** -- Spearman
-0.60 on the four new seeds, -0.14 over six. At n=6 that is not significant either way, and
nothing in the arm's stated mechanism predicts the sign -- vamp corrected me on that, since
amendment 3 predicts how often a run escapes, not which reference an escaped run tracks -- so it
stands as measured and unexplained. The consequence is firm regardless: seed and
checkpoint selection for Tier 2 has to happen on our own synthetic noise, because the Tier 1
reference is forbidden as supervision and the evaluator metric carries no usable signal
(`results/sweep_c.json`).
Blocked: nothing, and two things are explicitly NOT approved, so neither gets started on my initiative: the 30-clip Tier 2 pass, which waits on the 350M arm A result AND Luai's separate go, and pushing closer's packer wrapper to widen the code_commit_url. The quality-gates / dev-confirm section of the all-hands brief is closed for good -- Luai dropped it; the brief's Rules section and its three-gate requirement are what stand.
Next: D/E/F are finishing on the sweep box and are descriptive at two seeds whatever they show; the
350M arm A run on e24 is the only thing that can give a real success rate, since every run so far
has SP-SR and MP-SR at exactly zero. **The 30-clip Tier 2 pass is not started and needs Luai's
word** -- it is the expensive one. Spend so far on the standing budget: $3.91 of $30, two boxes
running, the Tier 1 box deleted the moment its results were verified off it.
Updated: 2026-10-24 21:55

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
