# Tier 3: the reconstruction, after MegaPose died

Tier 3 is a third of Track 2 and we have nothing on it. The challenge page is explicit that this
is not a missing *score* but a missing *submission*: **"A valid Track 2 submission consists of a
separate submission to each tier"**, and **"Your final score is a weighted combination of the
three tiers."** The kit agrees — `track_2_tier3` is `"Tier 3: bring your own reconstruction"`,
`input_path: null`, scored against the clean multi-view truth, `align_initial_pose: false`.

## What is legal here, quoted rather than assumed

`nvidia-isaac/video_to_data` **PR 207** (merged, "Update challenge FAQ PDF") adds one Track 2 FAQ
entry:

> since no camera calibration is released for Tier 3, teams may use the first frame of Tier 1 data
> to compute alignment, provided the Tier 1 caption isn't otherwise used in Tier 3.

So frame-0 alignment against Tier 1 is sanctioned, and it is the *only* Tier 1 read in the plan
below. The object meshes are released Tier 3 assets (`mesh_root: track_2/baseline/mesh`), so the
object's name — `baseball_bat`, `big_red_bowl`, 29 of them on disk — is available without touching
a caption. Nothing in this plan reads a Tier 1 or Tier 2 *object trajectory* beyond frame 0.

## The original plan is dead, and this is why

`brev/t3_recon.sh` was built around MegaPose's RGB refiner. **Its weights no longer exist on any
reachable host:**

| source | result |
|---|---|
| `www.paris.inria.fr/archive_ylabbeprojectsdata/...` (the only upstream) | TCP connect fails, 45 s timeout — the host root fails too, not just the path |
| the same host via an independent third-party prober | 522, so it is not this Jetson's network |
| `gepettoweb.laas.fr/data/happypose/` (happypose's second mirror) | 403 on the tree, **404 on `megapose/`** — the mirror does not carry the models |
| `happypose.s3...`, HF (`ylabbe/*`, `megapose/*`, `bop-benchmark/megapose`) | 404 / 401 / BOP training data, no checkpoint |
| GigaPose's `download_megapose.py` | points at the same dead INRIA URL |

HappyPose's own downloader lists exactly two mirrors and one of them is gone, so this is not a
path error.

## The replacement is NVIDIA's own stack, which is better than what it replaces

`nvidia-isaac/video_to_data` ships `reconstruction/modules/` — the pipeline that *produced this
dataset's ground truth*: `v2d_foundation_pose`, `v2d_sam2`, `v2d_grounding_dino`, `v2d_unidepth`,
`v2d_moge`, `v2d_geocalib`, `v2d_bundlesdf`, `v2d_hoi_object_reconstruction`. Single exo camera,
RGB only, known mesh — the chain is:

1. **`v2d_geocalib run_video_to_calibration`** — intrinsics and gravity from the video alone. No
   Tier data, which removes the one rules ambiguity in the old design (its camera solve fitted
   Tier 2 human joints across 40 frames; this needs none).
2. **`v2d_grounding_dino run_video_to_object_bboxes`** — the object, prompted with its released
   mesh name.
3. **`v2d_sam2 run_video_to_masks`** — the object mask, propagated through the clip.
4. **`v2d_unidepth` or `v2d_moge`** — metric depth per frame, which is what FoundationPose needs
   and what our RGB-only camera does not give.
5. **`v2d_foundation_pose run_video_to_poses`** — 6-DoF pose per frame against the released mesh,
   in the camera frame.
6. **Frame-0 alignment** (FAQ PR 207) — one rigid transform placing the camera-frame track into
   the released world, fixed so frame 0 equals the Tier 1 frame-0 object pose.

Every weight is on a reachable host, checked: FoundationPose at `nvidia/foundationpose` (ONNX) and
`gpue/foundationpose-weights` (the two PyTorch run dirs `2023-10-28-18-33-37` and
`2024-01-11-20-02-45`, which are the same runs the official downloader pulls from Google Drive);
SAM2 at `facebook/sam2.1-hiera-*`; GroundingDINO from IDEA-Research's GitHub releases; UniDepth and
MoGe from HF; FoundationPose also from NGC for the TensorRT backend.

## Cost, and what it buys

| | |
|---|---|
| one A6000 at $0.68/h: build 4–5 images, download weights, run 30 clips × ~766 frames | **$3–5, cap $8** |
| then T3-VALID: 409,600 steps × 30 clips on the reconstruction | **≈$2** |
| credit available | ≈$35 |

That is the whole eligibility gap in money terms. The rest of it is free and not mine to do: join
the four Tier 3 Kaggle competitions (`--check-entry` says none are joined), the registration form,
and getting the `Nvidia-V2D-Challenge` GitHub invite accepted — it was sent 2026-10-25 and is
**still pending**, which means the organisers cannot currently open the `code_commit_url` that
every row of both submitted entries points at.

## Already done, free

- 30 exo videos downloaded (655 MB, `$S/t3/videos_dl`).
- 29 object meshes and the Tier 1 frame-0 poses are on disk.
- The upstream repo's tree is readable through the GitHub API without a full clone (4,932 paths),
  which is how the module paths and weight sources above were checked rather than guessed.

## The bar, pre-registered

The reconstruction is scored **against Tier 1 truth on the Jetson, free**, before any policy
trains on it: per clip, ADD-AUC and mean position error of the reconstructed track. A clip whose
reconstruction is worse than the Tier 2 noisy input (AUC 0.425 / 2.36 cm) is reported as such
rather than hidden, because Tier 3's score is bounded by its input. T3-VALID then trains on it
and the entry ships under the same registered z-drop rule as the other two tiers.
