# Arm CR — the contact-role reward

Written 2026-10-28 00:20 UTC. **Staged, not launched**: the standing order is no `brev create`
without Luai's word per item, and the launch command at the bottom is where this stops.

## What it changes, and why

NVIDIA's ReconBody objective rewards both palms on every frame. `palm_position` and
`palm_orientation` are each summed over the two hands whatever those hands are doing, so a hand the
reference leaves idle is tracked exactly as hard as the hand carrying the object. Our own numbers
say the carrying hand is where the score is: of the nine clips under 0.30 AUC, seven lose the
object while the body keeps tracking, and `docs/FAILURE_MOMENTS.md` names a grip that opens or
slides as the mechanism on five of them.

Arm CR adds two terms, gated per hand on a **role** — *contact*, meaning this hand is supposed to
be holding the object on this frame, or *free* — and makes one existing weight depend on the frame.

| | term | weight | what it is |
|---|---|---|---|
| 12 | `contact_role_palm` | 2.0 | per contact-role hand, `exp(-gap² / 0.05²)` on the palm-to-object-surface gap, averaged over those hands |
| 13 | `contact_consistency` | 1.0 | of those hands, the fraction the **simulator** reports loading the object |
| 6 | `body_joint_position` | 5.0 → **1.0 while either hand is in the contact role** | `V2D_BODY_WEIGHT_IN_CONTACT`, applied by scaling the term, because `set_weights` is per curriculum stage and this has to be per frame |

A free-role hand contributes nothing to either term, which is the point of the arm. Both terms are
averaged rather than summed over the contact-role hands, so they stay in `[0, 1]` like every other
shaped term in this objective — the house weights are calibrated against that range, and a sum
would make a two-handed clip's term twice a one-handed clip's for the same quality of grasp.

Two honest limits, both written into `brev/patch_contact_role.py`:

- **The kernel's surface distance is a sphere**: `|palm − centre| − object_radius`, read once at
  build time from the `CommandBuffers.object_radius` field NVIDIA's own wrench-support launch uses.
  Exact for compact objects, conservative for long ones — a cane is 0.6 m, so a grip on the handle
  reads as ~25 cm of gap and the term saturates low. The per-object point cloud is the better
  answer and a bigger change to the kernel's inputs; it is not what this arm is testing.
- **`contact_consistency` is the sim's report, not the label.** It comes from the wrench support
  the force-closure term already computes, so a hand can be in the right place and still score
  zero here. That is deliberate: the two terms disagreeing is the signal.

## The roles: smoothed, and gated on distance

Roles come from `$V2D_CONTACT_ROLES` — one `eXX.npz` per clip holding `role` of shape (frames, 2),
side order `("left", "right")` — which `v2d contact-roles` writes from the reference the arm trains
on. The reference's raw `hand_contact_active` is not usable as a role for two measured reasons:

- **It drops out mid-grasp.** A hand carrying the object for 300 frames has the flag flicker off a
  few frames at a time, and a role that flickers teaches a reward that flickers. Gaps of **10
  frames or fewer** between two held frames are filled; leading and trailing zeros are left alone,
  because reaching for the object and letting go of it are not dropouts.
- **It is set for hands nowhere near the object.** A flagged frame only counts as a contact role if
  the palm is within **8 cm** of the object's surface — the kit's own AUC vertex cloud, so
  "surface" is the surface the score is computed over.

Both filters run offline, on the Jetson, so what the policy was taught is a file that can be read
and diffed rather than logic buried in a warp kernel. Measured over the eight clips
(`results/contact_roles.json`): **265 gap frames filled, 666 flagged frames dropped as too far.**

| clip | object | frames | contact-role frames (L / R) | filled | dropped |
|---|---|---|---|---|---|
| e01 | vacuum | 747 | 43 / 0 | 4 | 0 |
| e04 | baseball_bat | 803 | 277 / 193 | 47 | **214** |
| e05 | big_black_box | 1003 | 378 / 348 | 36 | 81 |
| e07 | iron | 896 | 0 / 696 | 4 | 1 |
| e09 | big_red_bowl | 870 | 445 / 386 | 86 | 34 |
| e19 | cane | 886 | 159 / 103 | 53 | 26 |
| e25 | cyan_water_bottle | 859 | 177 / 298 | 31 | **277** |
| e28 | squeegee | 988 | 0 / 12 | 4 | 33 |

e04 is the case that prompted the filter and it holds up: the right hand is flagged for 277 frames
at a median of 6.8 cm and a maximum of **19.3 cm** from the bat's surface, and 109 of those frames
are dropped.

### Two clips this arm can barely touch

**e28 has 12 contact-role frames of 988, and e01 has 43 of 747.** On e28 the reference flags only
41 frames at all, and the left hand's 33 all sit 8.2–10.3 cm out, so the gate zeroes them; the
mechanism `docs/FAILURE_MOMENTS.md` found there is not a grip opening but the squeegee being
*flung* at 0.48 m/s against the reference's 0.10. A reward term that applies to 1% of frames cannot
move that clip, and the same is nearly true of e01.

They are staged because they were asked for, and each costs the same ~$3.19 as a clip the arm can
act on. If a clip has to come out of this arm, **e28 and e01 are the candidates, not e09** — e09
has the second-highest coverage of the eight (445 and 386 frames, 386 of them two-handed) and is
one of the clips CR is most likely to move. Dropping those two would make it 6 clips over 3 boxes
at **$19.18**, leaving $8 more of the credit.

## SMOKE TEST

No training ran on this machine and none could: flash_chord lives inside the training image, which
is x86-64 CUDA, while this is aarch64 Tegra with an iGPU, and `docker images` has no such image.
**What ran instead is the real patched kernel source, compiled by the real warp (1.18.0) and
launched on its `cpu` device** against worlds whose answers are computable by hand —
`src/v2d/arms/kernel_harness.py`, pinned by `tests/test_arm_cr_kernel.py` (9 tests).

| check | result |
|---|---|
| the patched objective parses, and the term count matches the names | 14 terms (CR alone), 15 with the orientation arm |
| the kernels compile for the CPU and launch | yes — 7.3 s wall from a cold warp cache, 4.9 s of it codegen |
| every term finite, and the score finite | yes |
| `contact_role_palm` against hand-computed `exp(-gap²/std²)` | 0.852144 vs 0.852144 (2 cm gap); 0.003151 vs 0.003151 (12 cm) |
| `contact_consistency` against the hand-computed touching fraction | 0.5, 1.0, 0.0 — all exact |
| a free-role hand with its palm **on** the object earns nothing | both terms 0.0 |
| `body_joint_position` scaled on contact frames only | 1.0 → **0.2** in contact, 1.0 → 1.0 free |
| the 12 stock terms, bit-identical to the unpatched kernel | yes, `atol=0` |
| `force_closure` itself unchanged by the reducer's new output | `[0.125, 0.25, 0.25]` both |
| the npz `v2d contact-roles` writes is read back by the loader the patch injects | yes, by file and by directory+tag; a (2, frames) array is transposed |
| an unreadable roles file is fatal rather than a silent fall back to the stub | 5 cases, all raise |
| composes with START + ORI in the launcher's canonical order | yes; **the reverse order refuses atomically** |

Whole suite: **467 passed**, ruff clean over everything this arm touches.

### What the smoke test does not cover

Named rather than implied, because the brief asked for a training start and this is not one:

1. **Hydra composition** (`--cfg job`). Needs the package; the config and curriculum are checked
   the only free way instead — every patched file parses, and every curriculum stage's key set
   equals the term-name tuple, which is the invariant `set_weights` raises on.
2. **The reference loader, the SAC loop, throughput.** No steps/s figure from here.
3. **`CommandBuffers.object_radius`'s shape.** The host-side read is tolerant (array or scalar) and
   raises a named error if the field is gone, but its actual value is first seen on a box.

So the first cell's first minutes are still the real gate: GPU >90%, the step counter moving, and
`contact_role_palm` non-zero in the W&B term log. If `object_radius` is not what this assumes, the
run dies in `build()` before training, which is the failure we want.

## Cost and the launch

Eight clips, **4 boxes × 2 clips**, full 349,999,104 steps, seed 42, support per clip, the
bridge_v1 references every second-wave arm trained on (so the reward is the only variable),
10-minute pulls, dead-runner check, deletion authority 24 h, auto-replacement on.

| | |
|---|---|
| box-hours | 0.6 bring-up + 2 × 4.4 = 9.4 h per box |
| **cost at $0.68/box-h** | **$25.57** of the $27 cap |
| credit position | ~$37.77 left; $9 reserved for Tier 3, leaving $27 |
| per clip | $3.19 |

Four boxes of two rather than three of three for a reason worth stating: the cost model prices a
group by its **fullest** box, so 8 clips over 3 boxes is 3 per box — **$28.15, over the cap** —
while 8 over 4 is 2 per box at $25.57, and it finishes in 9.4 h instead of 13.8. Dropping a clip
from 8 to 7 over 3 boxes would have saved **nothing**, which is why the eight are all still here.

Staged and verified: `$S/cr_refs`, 8 clips, 58 files (50 references + 8 roles), 241 MB, manifest
58 lines for 58 files, all eight from interp's fingers on bridge_v1, 5 of 8 resolving to a
supported scene.

    cd ~/projects/v2d-challenge && brev/second_wave.sh --groups CR --go --cap 27

`brev/stage_cr.sh` has already run, so the launcher uses the tree as it stands; re-running it is
free and idempotent. One flag is new on the box and worth knowing about: `run_tier2_full.sh` now
writes `V2D_CONTACT_ROLES` and `V2D_CLIP_TAG` into the training command itself, because
`docker exec` does not inherit the box's environment and the roles are read at training time. New
runner md5 **`2ad58d1bc56dd69a8ab6757fec2c4213`**.

## The bar, pre-registered

One variable against each clip's own record, judged the way every arm since the second wave has
been: **kit AUC on the z-drop-selected checkpoint, per clip, against the record that clip ships
today**, and adoption into the entry only under the registered input-only rule (distance to the
INPUT reference, no truth read).

- **Adopted** on a clip if its z-drop pick beats that clip's current ship on the input-only rule.
  Nothing is adopted on the kit score, which is reported but is not the gate.
- **The arm is worth its 30-clip run** only if it wins on at least **4 of the 8** clips and the
  eight-clip mean kit AUC rises. Three or fewer, or a flat mean, and CR stops here.
- **e28 and e01 are expected to be flat** — 12 and 43 contact-role frames — and are excluded from
  that count: 4 of the 6 clips the arm can actually act on is the real bar.
