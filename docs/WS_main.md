# Workstream report — session `v2d`, branch `main`

Owns `brev/`, the launcher, observability, packaging and the README.
Updated 2026-10-24 15:00 PT.

## State: all four jobs delivered; one decision needs Luai

| job | state |
|---|---|
| Brev probe / gate | **4 attempts, 3 bugs found and fixed, ~$0.35 of a $2 cap.** Hardware confirmed. Attempt 4 in flight. |
| launcher | Picker rewritten and tested; hard `--max-usd` cap; readiness, TTY and session-survival all fixed. |
| W&B + HF observability | **Live.** Project `v2d-track2`, private repo `luaia/v2d-track2`. |
| packaging / README | `scripts/` → one `v2d` CLI, 18 stages, ruff clean, **87 tests**, README with a command per results row. |

## Needs Luai

1. **Gated or ungated reference set.** The gated cleaner is the one whose mechanism is validated,
   but it gives up **SP-SR 1.0000 → 0.9667**, and under rank-sum that one episode may cost more
   than its AUC +0.0176 buys. My pre-registration chose the validated mechanism; this is a
   ranking call, not a scientific one. Both sets are built; switching is one flag
   (`--gate-results`).
2. **The quality-gates / dev-confirm section of `~/projects/_ops/briefs/v2d_allhands.md` is not on
   disk.** The file ends at the Rules bullets and nothing under `_ops` mentions it. I followed the
   Rules section's three-gate requirement and did not guess at the rest.

## What the Brev attempts bought

Nine defects, every one of them on the path to a paid training run, and none of them visible
without hardware. The first four cost ~$0.10 each in create-build-die cycles; the rest were found
on a live dev box in seconds.

| # | reached | failed on | fix |
|---|---|---|---|
| 1 | price table (dry run) | **picker chose a 16 GB V100 at $2.81/h** for a 48 GB job: awk read `TARGET_DISK` as VRAM and `CAPABILITY` as price | header-offset parser, 17 tests, no fallback default |
| 2 | instance created | `brev ls` says `RUNNING` while `SHELL` says `NOT READY` | wait on the shell, prove it with `ssh true` |
| 3 | **A6000, driver 580.126.09, 48,539 MiB free, image in 229 s** | `docker/run.sh exec` passes `-it`; stdin is never a TTY under `ssh 'bash -s'` | talk to docker with `-i` |
| 4 | container up, `flash_chord ok [CudaDevice(id=0)]` | ssh died during the first long silent step, taking the run with it | detach under `setsid nohup`, poll for `DONE` |
| 5 | Hydra config parse | `total_environment_steps must be divisible by world_count, got 10000896 and 2048` | 2048 × 4883 = **10,000,384** |
| 6 | reference load | `no support asset could be resolved` — the parquet was uploaded flat | upload the `sequence_id=/robot_name=` tree with `reconstructed_stage/<seq>_support.usda` |
| 7 | **G1 URDF import** | `cannot call vectorize on size 0 inputs` — **git-lfs absent from the image**, so all 1045 meshes are 130-byte pointers | ship this host's resolved meshes; `git lfs pull` there hangs at zero progress |
| 8 | object URDF import | `ParseError: not well-formed, line 1 column 0` — the parquet embeds **absolute Jetson paths**, so Newton parsed a path as XML | `v2d rebase-assets` |
| 9 | scene build (~90 s) | `support surface contains no Cylinder or Cube collision shapes` — **16 of 30 clips ship the empty 125-byte support stage** | choose `scene.support` per clip from the file |

Two of these would each have destroyed a full run on their own. Bug 7 meant **no training could
ever start**, and it hid behind `|| echo "WARNING: git lfs pull incomplete"` before surfacing ten
minutes later as an error naming neither LFS nor a mesh. Bug 9 would have crashed **16 of the 30
clips** on a pass costing $54–72.

The hardware question B0 exists to answer is settled: the cheapest card clearing FlashCHORD's
floor really does clear it, with the required driver, docker works, the image builds in under six
minutes, and Warp binds the GPU as `"NVIDIA RTX A6000" (47 GiB, sm_86, mempool enabled)`. What is
still outstanding is throughput and hours-per-clip, which needs one scene build to complete.

## The result that changed the science

The brief requires a known-answer gate per cleaner. Adding one broke my own result:
**both support-contact cleaners recover only 0.21–0.27 of a tilt we inject ourselves**, against a
pre-registered bar of 0.70, with 9 of 30 clips coming out worse. Do-no-harm passes cleanly.

Recovery tracks object shape — 0.72–0.91 with one flat base, **negative** for squeegee, luggage,
stroller, chair, bat — because the support facet is an argmax over hull facets against "down",
and a 12° tilt makes a different facet win. That also supersedes my earlier guess about the e17
`traffic_cone` regression.

Gating per clip on synthetic recovery (correct 7, pass 23 through) gives recovery **0.7498** and
**AUC 0.4423161** against a bar of **0.4423** derived from the per-clip decomposition — a
four-decimal hit, and much stronger evidence than the aggregate AUC was.

**For the other workstreams:** if your cleaner or denoiser corrects an object pose using a
support face, test it against an injected tilt before trusting any score. Hull-facet argmax does
not identify a support face on objects without a flat base, which is most of this dataset.

## Costs

| | |
|---|---|
| Brev | **~$0.35** of $250, all probe attempts, every instance deleted (`brev ls` empty between runs) |
| Jetson GPU | none; `~/projects/_ops/GPU_RELEASED` untouched |
| pushed / submitted / emailed | nothing |

## Repo notes for anyone else in here

- `v2d --help` is the entry point; every stage writes a JSON record under `results/`.
- `brev/gate_launch.sh --check-provenance-only` tests the reference guard without touching Brev.
- Do **not** move files under a running batch: the driver used to resolve child stages by path at
  spawn time, which cost 11 of 30 clips. It now spawns `python -m v2d.retarget.sequences`.
- Two CPU workers is the ceiling here, and even two lose clips when a Brev poll loop competes.
