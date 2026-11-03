# T1P: Tier 1 precision, from scratch on the clean references

**Pre-registered 2026-11-02, before launch.** One variable against the shipped Tier 1 entry: the
policies are trained from scratch on the *clean tier_1* reference rather than on the noisy Tier 2
retarget that 16 of the 30 shipped clips chase. Recipe otherwise the shipping one — START +
orientation std 0.3 + object_position std 0.08 — seed 42, 349,999,104 steps, `world_count=1`,
`scene.support` per clip from the data.

## Why this clip set, and why this is the only lever left

Three of the four cheapest rungs on all twelve boards are Tier 1, priced against the next
*distinct* score (ties share a rank, confirmed in writing by the organisers on 2026-11-02):

| board | our score | +1 rank needs |
|---|---|---|
| tier1-mp-sr | 0.50000 | **1 clip** crossing the 3 cm / 30 deg gate |
| tier1-mppe | 12.06020 | **+1.9 cm summed** over 30 clips (DAP at 11.99700) |
| tier1-auc | 0.54808 | +1.03 summed AUC |

Seven clips sit just outside the MP-SR gate: **e29 3.85, e11 3.97, e02 4.57, e21 4.79, e09 4.89,
e13 4.98, e17 5.15 cm**. The same seven serve all three rungs — a centimetre each is ~7 cm summed
against the 1.9 cm MPPE needs, and their AUC runs 0.206–0.627, so there is room there too.

**The warm start cannot collect them, and that is measured, not assumed.** Six of these seven were
fine-tuned on the clean Tier 1 reference and every one came back worse on MPPE
(`results/t1_finetune_adoption.json`, `results/t1_rank_sensitivity.json`):

| clip | shipped MPPE | fine-tuned MPPE | shipped AUC | fine-tuned AUC |
|---|---|---|---|---|
| e29 | 3.85 | 8.72 | 0.627 | 0.602 |
| e11 | 3.97 | 4.44 | 0.468 | 0.388 |
| e02 | 4.57 | 35.05 | 0.426 | 0.370 |
| e21 | 4.79 | 10.34 | 0.206 | 0.223 |
| e09 | 4.89 | 11.66 | 0.300 | 0.224 |
| e17 | 5.15 | 5.29 | 0.447 | 0.328 |
| e13 | — | never tried | 0.159 | — |

That arm trades position accuracy for reference tracking, 12 of 12 rejected clips over.

## The mechanism this arm tests, stated as a falsifiable claim

The warm start failed because **re-fitting a converged actor to a different trajectory is
distribution shift** — our own measurement, the cold swap of the clean reference under the shipped
actors took the entry from 0.465 to 0.289 AUC with only 2 of 30 clips improving. That is a fact
about *re-fitting*, not about the reference.

Training from scratch on the clean reference is a different thing and has never been run on
Tier 1. The claim: these policies currently chase a reference carrying **1.94 cm of error against
truth**, so a policy that tracks its reference perfectly still scores ~1.94 cm of MPPE it did not
earn. Removing that floor is worth roughly the gap these seven clips need.

**If the arm fails**, the honest reading is that the deficit is not reference noise but the policy's
own precision, and no reference change collects it.

## The bar, registered now

- **Per clip:** adopt only under the standing Tier 1 no-regression rule — AUC improves and no other
  board regresses. Legal and truth-equivalent, because Tier 1's permitted input *is* the configured
  scoring reference (organisers' answer 2, `results/organiser_answers.json`), so a better rollout is
  simply kept and a worse one is simply not shipped.
- **Arm-level success:** at least one of the three rungs above crosses — one clip under the MP-SR
  gate, or 1.9 cm of summed MPPE, or +1.03 of summed AUC.
- **Downside is the bill, not the entry.** No clip ships unless it beats what is already uploaded.

## Cost

7 clips, 7 boxes, 1 clip each, 4.4 GPU-h + 0.6 h bring-up at $0.68/h = **$23.80**. Checkpoints are
kit-scored on pull and selected per clip; z-drop selection is unnecessary here because the
permitted input is the scoring reference, but it is still recorded for the provenance trail.

## Not in this arm, and why

**Tier 3 is held.** The organisers' answer 3 forbids the ground-truth mesh for training, and our
Tier 3 chain uses it both as FoundationPose's CAD model and as the simulator's object geometry
(`src/v2d/retarget/sequences.py:270`). A follow-up naming both uses is in the same email thread.
Until it is answered, Tier 3 compute buys an entry that may be withdrawn — including the staged
13-clip warm-start batch, which stays staged.

---

## Addendum, 2026-11-03: the first wave trained against a mesh that was not there

Launched 01:03 UTC, dead by 01:06, cost about $1.50. Worth writing down because **every check in
the chain passed** and the run still reported success.

I staged the tree with `--out $S/t1_precision` but without `--mount t1_precision`, so
`stage_t2_combined.sh` kept `MOUNT` at its `t2_combined` default. `--mount` is what gets written
into every parquet's `object_mesh_paths` and `object_urdf_paths` as the *in-container* path, while
`--out` only decides where the tree lands locally. The launcher mounts a stage at
`~/v2d_assets/<basename of the stage dir>`. So all 30 parquets pointed at
`/v2d/assets/t2_combined/...` and the tree arrived at `/v2d/assets/t1_precision/...`.

### Why nothing caught it

| check | what it said |
|---|---|
| the stage's own verification | `checked 30 parquets, 0 problems; 14/30 resolve to a supported scene` |
| the md5 gate on the box | `references verified by md5 on attempt 1` |
| the box | RUNNING, image built, container up, `flash_chord ok [CudaDevice(id=0)]` |
| the cell | `e02 finished after 0 min, 0 trajectories` |

The manifest was internally consistent — it just described a tree that named the wrong mount. And
the final line is the trap: **`run_tier2_full.sh` touches `$CLIP/DONE` on its "produced no
checkpoint" path** (line 119), and `run_clip`'s poller breaks out of its wait the moment that file
exists. So a crash three seconds into training is indistinguishable from a clean four-hour finish,
and the box deleted itself three minutes later, taking the only copy of the traceback with it —
the runner's log lives at `~/t2/clip_eXX.log` on the box and is only pulled on a clean finish.

It was found by diffing the staged parquet against `cr_refs`, where the mount matches, not from
any log.

### Fixed, and what is still owed

- `stage_t2_combined.sh` now **refuses** a `--mount` that disagrees with the stage directory,
  names the fix in the message, and carries `V2D_ALLOW_MOUNT_MISMATCH` for a consumer that really
  does mount elsewhere. Four tests in `tests/test_stage_mount.py`, including the exact invocation
  I got wrong.
- The six live boxes were **re-driven rather than deleted**: the 14 GB image and every mesh and
  USDA were already uploaded, so only the 30 rewritten parquets re-synced.
- **Still owed** (not done here, because `launch_t2_combined.sh` and `run_tier2_full.sh` were in
  use by running boxes and a running bash script must not be edited): the no-checkpoint path
  should write a distinguishable marker — `FAILED` rather than `DONE` — so the poller can tell a
  crash from a finish, and `run_clip` should pull `clip_$TAG.log` before the box is deleted on
  *any* exit, not only on the dead-runner branch. A `scratchpad/catch_clip_logs.sh` poller is the
  stopgap.

### The cap guard bills boxes that no longer exist

Caught at 03:30 UTC, not from a failure. `cr_cap_guard.sh` sums `(now - first_seen)` over **every
box in its ledger** and never asks which are still alive, so b03 (which lived 19 minutes) and b21
(deleted after a hung build) kept accruing as though running. It read 19.87 box-hours and $13.51
when seven live boxes accounted for 15.07 and $10.25.

The direction is safe -- it over-charges, so a cap trips early rather than late -- but the timing
is not. At nine counted boxes per hour the $30 cap would have been reached about 06:15 UTC, which
is when e21 and e29 finish. The guard would have deleted five healthy boxes at ~95% of a 350M-step
run: the exact loss it exists to prevent, by the exact mechanism (acting on its own arithmetic
without doubting it) that the first version failed on.

Worked around for this run by dropping the two dead rows from the ledger and restarting with
`--keep-ledger` and a $36 cap, which moves the trip point to ~09:00 UTC against a ~07:30 finish.

**Owed:** the guard should reconcile its ledger against `brev ls` each pass and stop charging a box
that is gone, or charge it only up to the pass where it last appeared. `tests/test_cap_guard.py`
already has the harness for it -- a fake `brev` whose listing drops a box mid-run is one more case.

### Startup looks like a hang and is not

After `builder.finalize`, `train.log` stops growing for ~10 minutes with the GPU at 0% while Warp
compiles kernels. The tell is CPU, not GPU: `ps -eo pid,pcpu` shows python at ~170%. The
launcher's stall detector allows 900 s of no log growth, which covers it, but a human watching
`nvidia-smi` will conclude the run is dead.
