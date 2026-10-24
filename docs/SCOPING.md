# V2D Challenge, Track 2 (Robotic Grounding): scoping

Written 2026-10-06. Leaderboard freeze is **Wed 2026-11-04, 17:00 EST**, which is 4 weeks and 1 day away.

No compute was paid for, no account was registered and nothing was submitted. Downloads were the kit (0.7 MB), the FAQ, small metadata, and the Tier 1 and Tier 2 trajectory parquets (60 MB).

## Recommendation

**Conditional GO.** Enter Tiers 1 and 2 only if both conditions below hold. If either one fails, the decision is NO-GO.

1. **Budget approved.** Luai approves a hard cap of **$300 of rented 48 GB GPU time** (or NVIDIA grants credits; the FAQ says that is "under review").
2. **Day-4 gate passed.** On two Tier 1 clips, our FlashCHORD run must land within 2× of CHORD's published ~2 L40S-hours per clip. It must also land within ±0.1 of the CHORD baseline's per-clip MP-SR outcome. Both are scored locally against the public clean reference.

Tier 3 is a stretch goal, attempted only after the gate passes (see Week 3).

**Why it is worth a look:**
- The field is thin. Track 2 has exactly one non-baseline team on the board.
- The data is small and the scorer can be reproduced locally.
- A $0 measurement on this Jetson already reframes the problem. The Tier 2 input is close enough to the truth that **the tracker is the bottleneck, not the noise**.
- It fits the STANDARD's "named benchmark, one number against the strongest named baseline (CHORD)".

**Why it is not an easy yes:**
- Whole-body G1 + Dex3 loco-manipulation of stools, ladders and luggage is outside every codebase on jetson3.
- Training needs a 48 GB x86 GPU. Nothing trains on the Jetson.
- A 4-week sprint is exactly the anti-pattern the STANDARD warns about, unless it ends in a frozen, released result whatever the rank.

## 1. Submission requirements and evaluation

Sources: the challenge page, the submission kit (`v2d_submission_kit.zip`, `tracks/track_2/README.md`, `config/tracks.json`), and the FAQ PDF.

### Common to all tiers

- **Robot and simulator.** The robot is fixed: Unitree G1 + Dex3 hands. FlashCHORD's canonical recipe is `g1_recon_body_flash_sac`, built on the SONIC whole-body controller with an RL residual. The FAQ says whole-body control is required and "pretrained controllers are allowed".
- **One policy per episode.** "Include all scored episodes and use one checkpoint per episode" (kit `examples/run_policy_evaluation.py`).
- **All 30 episodes are required for a valid submission.** They total 22,990 frames: 440–1,003 per episode, mean 766, 30 fps. There are 29 objects, including a bar stool, vacuum, step ladder, rolling luggage, round table and monitor.
- **Format.** One CSV per tier with one row per frame: `row_id = track_2/eNNNNNN/w000/fNNNNNN/o0`, the object pose (`pos_xyz`, `quat_xyzw`), and five SHA-256 provenance columns plus `code_commit_url`.
  - The kit's sample has **22,989** data rows against 22,990 released frames. Check which frame is dropped before packing.
- **Evaluator.**
  - Run the kit's `flash_chord/scripts/evaluate_policy.py`, pinned to repo commit `8c778f9`, with `world_count=4096`, `reset_mode=explicit` and a deterministic policy.
  - Only `w000` is submitted.
  - `tools/pack_submission.py` converts evaluator poses into the released world frame with one rigid transform.
  - `submit/v2d_submit.py` uploads the same CSV to the tier's four Kaggle competitions.
- **Provenance.** You need a GitHub commit URL, plus hashes of the evaluator, the evaluator code, each checkpoint, the packer and the uploader. Checkpoints are not uploaded now, but "top solutions will be required to provide code to reproduce". For a private repo, share it with the `Nvidia-V2D-Challenge` account.
  - **Prizes require an Apache-2.0 code release and a short technical report.**
- **Limits.** 5 submissions per week **per tier**, unlimited in the final three days (Nov 2–4). The leaderboard page may lag by up to 24 h.

### Metrics

All four are scored against the **clean multi-view object trajectory in every tier**, then averaged over episodes. Thresholds are taken from `metric_code/track_2/*.py`.

| Metric | Definition | Better |
|---|---|---|
| AUC | ADD (mean distance of the object's mesh vertices) integrated over thresholds of 1–9 cm | higher |
| SP-SR | Episode succeeds if mean position error ≤ 10 cm **and** mean orientation error ≤ 0.5 rad (28.6°) | higher |
| MP-SR | Episode succeeds if mean position error < 3 cm **and** mean orientation error < 30° | higher |
| MPPE | Mean error of 6 keypoints at ±5 cm along the object's axes, in cm | lower |

- **How tiers combine is unpublished.** The site says "weighted combination of the three tiers", with each tier "weighted equally". How the 4 metrics combine into a single rank is not stated. Ask the organizers (Discord or v2d_challenge@nvidia.com) before optimizing for one metric.

### Per tier

| | Tier 1 | Tier 2 | Tier 3 |
|---|---|---|---|
| Input | Clean SOMA-X human and object trajectory (`tier_1_multiview_caption/`) | Same 30 episodes, human and object poses perturbed (`tier_2_synthetic_noise/`) | **No trajectory supplied.** Reconstruct from the exo video by any method |
| Pack flag | `--track track_2_tier1` | `--track track_2_tier2` | `--track track_2_tier3 --frame-references <your recon>` (LeRobot dataset or per-episode `.npz` with `pose_xyzw [T,1,7]` in `released_track2_world`) |
| Constraint | none | Use anything except the Tier 1 clean captions to recover the trajectory | Same. CARI4D is the organizers' off-the-shelf baseline; a "commercial-friendly recreation" of it was used |

**Things we must not do:**
- Feed the public Tier 1 clean trajectory into the Tier 2 or Tier 3 pipelines, directly or as supervision. The reference is public only so local scoring works.
- Assume the Tier 2 trajectory may initialize Tier 3. It is derived from the clean reference, so it is almost certainly out of bounds.
- Assume Tier 3 may use the released `mesh/` assets. That is **ambiguous**: the FAQ forbids Track 1 from using Track 2 assets, but says nothing about Tier 3. Ask before relying on it.

### Leaderboard

Snapshot from `leaderboard.json`, generated 2026-10-07 02:24 UTC.

| Tier | Entry | AUC | SP-SR | MP-SR | MPPE (cm) |
|---|---|---:|---:|---:|---:|
| 1 | Xueyi Wang (FF) | 0.557 | 0.700 | 0.667 | 8.22 |
| 1 | CHORD baseline | 0.510 | 0.700 | 0.200 | 17.74 |
| 2 | Xueyi Wang (FF) | 0.421 | 0.533 | 0.267 | 15.34 |
| 2 | CHORD baseline | 0.374 | 0.533 | 0.067 | 20.52 |
| 3 | Xueyi Wang (FF) | 0.153 | 0.000 | 0.000 | 77.67 |
| 3 | CHORD baseline (CARI4D recon) | 0.002 | 0.000 | 0.000 | 70.17 |

- **One competing team so far**, last submission on 2026-10-06.
  - Do not assume who is behind "Xueyi Wang (FF)". The DexTrack and AdaDexTrack author is Xueyi *Liu*. AdaDexTrack (CVPR 2026) adapts the reference during execution because "noisy references snowball", which is exactly what Tier 2 rewards.
- **Tier 3 is effectively unsolved.** Both entries score zero success, and it counts for a third of the track.
- For context, Track 1 has 4 teams and Track 3 has 2.

## 2. Data sizes

Read from the HF tree API without downloading; my pager is `hf_tree.py`. The dataset is `nvidia/video_to_data_challenge` (CC-BY-4.0), 4.14 GB in total.

| Path | Files | Size |
|---|---:|---:|
| `track_2/tier_1_multiview_caption/` | 156 | 1.080 GB |
| ↳ `data/` (pose parquets) | 30 | 31 MB |
| ↳ `videos/` (exo RGB, 1536×1152 H.264, byte-identical across tiers) | 30 | 686 MB |
| ↳ `mesh/` (29 textured GLB) | 29 | 361 MB |
| ↳ `support_surface/`, `ground_plane/`, `meta/` | 65 | 2 MB |
| `track_2/tier_2_synthetic_noise/` | 156 | 1.080 GB (same layout) |
| Tier 3 | none | Reconstruct from the Tier 1 video; no CARI4D output is released |
| `track_1/` | 67 | 0.51 GB |
| `track_3/` | 169 | 1.47 GB |

- Each parquet holds SOMA-X human data: 77 local rotation vectors, translation, identity, scale and bone parameters. It also holds one object pose `[x,y,z,qw,qx,qy,qz]` (world_T_object, OpenCV world frame) and a visibility flag. Visibility is ≥ 99.9 % in both tiers.
- **No camera intrinsics or extrinsics** ship with the video. That matters for Tier 3.
- The starter repo `nvidia-isaac/video_to_data` is large. A shallow clone passed 3 GB before I stopped it, so read it through the GitHub API or a sparse checkout of `robotic_grounding/`.
- Everything needed for Tiers 1 and 2 fits in about 1.2 GB per tier.

## 3. Compute path

### Two lanes in the official repo

| | Isaac Lab 2.3 / RSL-RL PPO (`robotic_grounding/`, `SonicG1-*` tasks) | **FlashCHORD** (`robotic_grounding/flash_chord/`: Newton + Warp + JAX, FlashSAC) |
|---|---|---|
| Recipe | Three PPO stages, 5k / 50k / 5k iterations at 4,096 envs | `g1_recon_body_flash_sac`: 2,048 worlds, **350.0 M env steps** per clip |
| Measured or published time per clip | **"about 2 days for one task" on one RTX 6000 Ada**. A user reported this in issue #148, and the organizers said it "aligns with the training times we see on our end for IsaacLab and PPO" | **"∼2 hours" per sequence on a single L40S** (CHORD paper, arXiv 2607.00033, "Training Details"). No wall-clock is published for the G1 V2D recipe specifically |
| GPU requirement | 48 GB recommended; driver 580.126, CUDA 13 | **"48 GiB GPU with ≥46,000 MiB free"**; driver ≥580; Docker |
| Is it the submission evaluator? | No | **Yes.** The kit's `evaluate_policy.py` is FlashCHORD |
| Verdict | Not viable: 90 policies × 48 h = 4,320 GPU-hours | **The only viable lane** |

- The FlashDexRetarget paper (arXiv 2610.01849, floating hands only, no G1) quotes "about 3,000 GPU-hours for CHORD" on 50 motions, which works out to about 60 h per motion. That **contradicts** CHORD's own ~2 h figure, so I budget with CHORD's number and treat the gap as a risk.
- FlashDexRetarget's own number is about 30 GPU-h for 50 motions on RTX 3090s, roughly 0.6 h per motion, with a single shared multi-motion policy.
- The V2D evaluator expects one checkpoint per episode, so a shared policy would need evaluator changes. Not planned.

### GPU choice

- **4090 (24 GB) and 5090 (32 GB) do not meet the stated 48 GiB requirement.** That memory figure is for 2,048 worlds plus a 10 M-transition replay buffer on the GPU. Shrinking either changes the recipe, so I do not count on it.
- **The cheapest viable cards are the RTX A6000 48 GB and the L40S 48 GB.**
- Prices: RunPod pricing page marked "Updated September 27, 2026"; Modal per-second rates converted to hourly.

| GPU (48 GB) | RunPod community | RunPod secure | Modal | Assumed time per clip |
|---|---:|---:|---:|---:|
| L40S | $0.79/h | $1.09/h | ~$1.95/h | 2–4 h (CHORD's 2 h, up to 2× for 766-frame whole-body clips) |
| RTX A6000 | **$0.33/h** | $0.53/h | not offered | 3–8 h (assumes it is 1.5–2× slower than L40S; **unmeasured**) |
| RTX 6000 Ada | $0.74/h | $0.84/h | not offered | ≈ L40S |

On these assumptions, the **cost per clip-policy is about $1.6–3.2 on an L40S and $1.0–2.6 on an A6000**. The A6000 is cheaper but slower in wall-clock time.

### Total for a competitive entry

| Item | Policies | L40S-hours |
|---|---:|---:|
| Day-4 gate: reproduce 2 Tier 1 clips | 2 | 4–8 |
| Development: 5-clip subset × ~6 variants (reward, termination, reference processing) | 30 | 60–120 |
| Early full pass on Tiers 1 and 2 (puts a valid entry on the board) | 60 | 120–240 |
| Final pass on Tiers 1 and 2, plus Tier 3 if attempted | 60–90 | 120–360 |
| Tier 3 reconstruction with model-based object tracking plus HMR, not CARI4D (CARI4D is ~45 min per 300 frames on an A100, ~57 A100-h for all 30 clips) | n/a | ~5–10 |
| **Total** | **~150–180** | **~310–740** |

**Dollar cost:**
- L40S community: 310–740 h × $0.79 = **~$245–585**.
- A6000 community: 465–1,480 h × $0.33 = **~$155–490**.
- **A $300 cap therefore covers Tiers 1 and 2 at the low-to-middle end of the range.** The gate measurement on Day 4 replaces these assumptions with a real number before most of the money is spent.
- **Wall-clock is the binding limit, not money.** One full 90-policy pass is about 180–360 L40S-hours. To land it in about 3 days needs **3–5 GPUs in parallel**.
- **Data movement is free.** The Track 2 inputs are 31 MB per tier, plus meshes.

## 4. What runs on the Jetson (Orin 16 GB, aarch64, JetPack R36.4.3, driver 540, CUDA 12.6) and on a Mac

| Task | Jetson | Mac |
|---|---|---|
| FlashCHORD training or evaluation | **No.** Needs driver ≥580, 48 GiB, and an x86 Docker image. Newton and Warp support aarch64 CUDA, and JAX lists aarch64 GPU, but not on Orin's driver; 16 GB also rules out 2,048 worlds | **No.** Warp and Newton are CPU-only on macOS |
| Isaac Lab | No: no Jetson support (2.3 added DGX Spark aarch64, not Orin) | No |
| Local scoring against the clean reference with the kit's metric code; Tier 2/3 reference analysis | **Yes. Done for this doc** (`scripts/tier2_input_ceiling.py`, seconds on CPU) | Yes |
| SOMA→G1 retargeting (`scripts/retarget/soma_to_g1.py`: Pink IK, torch, trimesh) | Probably on CPU. Unverified: needs the SOMA-X body-model assets, and pinocchio on aarch64 | Probably |
| Reference cleanup research (contact and support projection, penetration probes) | Yes. `opposition-deficit/dextrack/reference_penetration.py` and `audit.py` are NumPy/trimesh on CPU; `penetration_torch.py` runs with CUDA torch | Yes (CPU) |
| Packing, preflight, provenance hashing (`tools/pack_submission.py`, `tools/preflight.py`) | Yes | Yes |
| Tier 3 object pose tracking from video (FoundationPose-class) | Possibly, slowly. Not worth the setup; do it on the rented GPU | No |
| Viser viewers for replay and policies | Viewer yes; simulation no | Viewer yes |

**In short, all research that doesn't need a simulator runs here at $0, and every policy is trained on a rented GPU.**

## 5. Measurement done for this scoping: what Tier 2 actually is

`scripts/tier2_input_ceiling.py` scores the **released Tier 2 object input itself** against the clean Tier 1 reference, using the kit's thresholds. This is the score a policy that tracked its input perfectly would get. Results are in `results/tier2_input_ceiling.json`.

| | AUC (keypoint proxy*) | SP-SR | MP-SR | MPPE (cm) | Mean position error (cm) | Mean rotation error (°) |
|---|---:|---:|---:|---:|---:|---:|
| Tier 2 input, tracked perfectly | 0.82 | **0.97** | **1.00** | **2.36** | 1.87 | 21.4 |
| Leader's actual Tier 2 policy | 0.42 | 0.53 | 0.27 | 15.34 | n/a | n/a |
| Leader's Tier 1 policy (clean input) | 0.56 | 0.70 | 0.67 | 8.22 | n/a | n/a |

\*AUC here uses ADD over the 6 MPPE keypoints rather than mesh vertices. It is a proxy, not the leaderboard number.

> **Correction, 2026-10-24.** The 0.82 proxy is wrong and it reversed a conclusion. The real
> metric transforms 500 sampled mesh vertices, and with the released meshes the Tier 2 input
> ceiling is **AUC 0.425** (`results/tier2_input_ceiling_kitscorer.json`, scored with the kit's
> own metric code). A 21° rotation error barely moves keypoints 5 cm from the origin but moves a
> step ladder's far vertices by metres. The corrected reading: the Tier 2 input beats every team
> on SP-SR, MP-SR and MPPE, and **loses to the top four on AUC**. The orientation error splits
> into 15.8° of support-recoverable tilt and 11.4° of unrecoverable yaw, and fixing only the tilt
> would lift the ceiling to AUC 0.654 (`results/tilt_yaw_split.json`,
> `results/tier2_tilt_fixed.json`). See `docs/PLAN_11DAY.md`.

**The corruption itself:**
- Position error is a **slow offset**: 2.0 cm RMS, of which 1.98 cm sits below about 1 Hz and only **0.24 cm is jitter**.
- Rotation error is a random walk **clipped at exactly 30°**. The 95th percentile is 30.0° in 27 of 30 episodes, and the mean is 21°. That lies just under MP-SR's 30° bar and SP-SR's 28.6° bar.
- I found no dropout in the visibility flag or frozen frames.

**What this implies:**

1. **The tracker, not the noise, is what separates scores now.** The leader's Tier 2 policy loses 0.73 MP-SR relative to its own input. Even on clean Tier 1 it loses 0.33.
2. **Smoothing is useless.** Gaussian filters with σ = 2–20 frames and median filters never improve on raw, and σ = 20 drops MP-SR to 0.40. Snapping to one robust pose per detected rest segment also hurts: MP-SR 1.00 → 0.83 at the loosest threshold (`results/tier2_rest_snap.json`).
   - The error is a persistent offset, so **only an independent signal can remove it**: the video, or physical constraints such as support contact and non-penetration.
3. **Orientation margin is the Tier 2 risk.** The input already carries about 21° of the 30° allowed, so a policy that adds about 10° of its own rotation error fails MP-SR.
   - Recovering orientation, for example making resting objects sit stably on their support plane, buys margin precisely where the metric is tight.

## 6. Four-week plan

Leaderboard rank is not a claim. The deliverable is a frozen, released result: code, a tech report and one number against CHORD. That makes the sprint worth something even if the rank is not.

**The novel angle: physically consistent reference projection.** Before training, project each Tier 2/3 object reference onto the set of physically consistent poses. Then train the policy to track a reference that the object can actually follow. Two signals do the projection:

- **(a) Support contact.** When the object is not in hand, it must rest stably on the released `ground_plane` / `support_surface` without penetrating it. That fixes height, roll and pitch, two of the three rotational degrees of freedom.
- **(b) Hand-object contact consistency.** While grasped, the object must touch the retargeted Dex3 fingers without penetrating them. This reuses `opposition-deficit`'s penetration probe and reference-penetration code.

Why this rather than the alternatives:
- It targets the measured failure mode: a slowly varying rotation bias against a tight 30° margin.
- It uses only allowed inputs.
- It feeds Tier 3 too, since any reconstruction has the same failure modes.
- It is distinct from what opposition-deficit ruled out. That project tested penetration-weighted *retargeting* and a penetration *reward* on Allegro/DexTrack; this projects the *object reference*.

**Week 1 (Oct 7–13): $0 work, then the gate.**
- [$0, Jetson] Get the official scorer running locally on all three tiers.
- [$0, Jetson] Download the 29 meshes (361 MB). Measure Tier 2's per-episode support-plane penetration and tilt against the clean reference, to check whether support projection can recover orientation. **Kill the angle if it recovers under 5° of mean rotation error.**
- [$0, Jetson] Get `soma_to_g1.py` running on CPU for all 30 Tier 1 episodes.
- [≤$15 GPU] **Day 4 gate.**
  - Run the FlashCHORD G1 recipe on 2 Tier 1 clips, one easy and one hard; MP-SR is measured locally.
  - Record steps per second, wall-clock and peak memory on A6000 vs L40S.
  - Write everything to `results/`.
- **Go/stop decision on Oct 11.**

**Week 2 (Oct 14–20): a valid entry on the board.**
- Run Tier 1 and Tier 2 on all 30 clips with the stock recipe, about 60 policies on 3–4 parallel GPUs.
- Submit once per tier; that uses 2 of the 5 weekly slots.
- Score locally to pick the 5 worst clips for the development subset.

**Week 3 (Oct 21–27): the angle and an ablation.**
- Run projected vs raw references on the 5-clip subset with ≥2 seeds. This is the "why not simpler" ablation, so it must include a smoothed-reference arm.
- Run Tier 3 only if mesh use is cleared by the organizers:
  - Track the object from the video with the known mesh (FoundationPose-class), fit the human with off-the-shelf HMR, then retarget SOMA to G1.
  - Even AUC > 0.15 or a single SP-SR success would lead that tier.

**Week 4 (Oct 28–Nov 4): final pass and freeze.**
- Run the final full passes, then use the unlimited submissions Nov 2–4.
- Freeze `results/`, write the README and the short tech report (both required for any prize), and push only with Luai's approval.

### Expected leaderboard position (low confidence)

- **Tiers 1/2.** Reproducing CHORD gives about baseline: MP-SR around 0.2 on Tier 1 and 0.07 on Tier 2. Matching the current leader needs roughly a 3× improvement in Tier 1 MP-SR, which is real tracker work. A realistic outcome is **2nd–4th** on Tiers 1/2 in a field I'd expect to grow to 3–8 teams by the freeze.
- **Tier 3.** If it works, it is the cheapest place to move ahead, because the bar is near zero.
- **Overall.** Chance of a top-2 track finish (a prize) is roughly **20–30 %**.
- **Industry Innovation Award.** It is judged on cost and data-flywheel arguments, not rank. A physically consistent reference cleaner is a credible fit, and **this is the more likely payoff.**

## 7. Open questions for the organizers

These need a Discord or email question. Nothing is to be posted without Luai's approval.

1. How are the 4 metrics and 3 tiers combined into the final Track 2 rank?
2. May Tier 3 use the released object meshes? And the Tier 2 human trajectory?
3. May one checkpoint serve several episodes (a multi-motion policy) if `evaluate_policy.py` is unchanged?
4. Are GPU credits available? The FAQ says "under review".
5. Why does the sample CSV have 22,989 rows when the release has 22,990 frames?

## Sources

- Challenge page: https://nvidia-isaac.github.io/video_to_data/v2d_challenge/
- Kit and FAQ: `assets/v2d_submission_kit.zip` and `assets/v2d_challenge_2026_faqs.pdf` on the challenge page
- Leaderboard data: https://raw.githubusercontent.com/MVerghese/v2d-leaderboard-data/main/leaderboard.json
- Dataset: https://huggingface.co/datasets/nvidia/video_to_data_challenge (`track_2/README.md` and the tier READMEs)
- Repository: https://github.com/nvidia-isaac/video_to_data (`robotic_grounding/flash_chord/README.md`, `V2D_INTEGRATION.md`, `configs/experiment/g1_recon_body_flash_sac.yaml`, issue #148)
- CHORD: https://arxiv.org/abs/2607.00033
- FlashDexRetarget: https://arxiv.org/abs/2610.01849
- FlashSAC: https://arxiv.org/abs/2604.04539
- CARI4D: https://arxiv.org/abs/2512.11988
- AdaDexTrack (CVPR 2026)
- RunPod pricing (page updated 2026-09-27): https://www.runpod.io/pricing
- Modal pricing: https://modal.com/pricing
- Warp compatibility: https://nvidia.github.io/warp/latest/user_guide/compatibility.html
- Newton on PyPI: https://pypi.org/project/newton/
