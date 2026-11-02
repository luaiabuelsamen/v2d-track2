# One policy per sequence, selected without ground truth

**Track 2 — Robotic Grounding.** Luai Abuelsamen, University of California, Berkeley.
Code: <https://github.com/luaiabuelsamen/v2d-track2> (Apache-2.0). Container: see the form.
Hardware: an NVIDIA Jetson for all CPU work; rented single-A6000 cloud boxes for training.
*A longer version of this report, with the full derivations and the negative results, is
`docs/TECHNICAL_REPORT_LONG.md` in the repository.*

## Introduction

All three Track 2 tiers are scored against the same `multi_view_ground_truth`; only the
permitted **input** differs. We found this by accident and then verified it: our Tier 1 and
Tier 2 public scores came back identical, the two uploaded CSVs are byte-identical, and a
Kabsch fit shows every shipped rollout's reference is an exact (0.000 cm) rigid transform of
the released Tier 2 trajectory. Identical predictions earn identical scores.

That one fact set the strategy. The object trajectory is the deliverable; the policy exists to
make it physically realisable; and the interesting engineering is not the reward but **what you
are allowed to condition on**, because that is the only thing separating the tiers. We entered
all three, and treated the input constraint as the design variable throughout: every per-clip
decision in the Tier 2 and Tier 3 entries reads the *input* reference only, never the truth.

## Approach

SOMA-X human parameters → `soma_to_g1` retarget to a Unitree G1 with Dex3 hands (CPU, 30/30
clips at 0.0 mm residual against the released object trajectory) → FlashCHORD SAC, **one policy
per sequence**, 349,999,104 steps, `world_count=1`, `scene.support` set per clip from the
shipped `support_surface` primitives → `evaluate_policy.py` → per-episode checkpoint selection
→ the challenge kit's own packer.

Four recipe changes ship, each as a patch script that **refuses to apply to an upstream it does
not recognise**, so a silent half-patch is impossible: an object-*orientation* reward term
(weight 1.0, std 0.3) added to `recon_body`, which ships an `object_position_weight` and no
orientation counterpart; a start-curriculum patch; object-position std tightened 0.2 → 0.08 m
on clips already tracking; and finger targets on bridge-smoothed references.

**Selection is the submission rule, registered before the run.** The provenance schema allows a
different `checkpoint_sha256` per episode, so we pick, per clip, the latest checkpoint whose
achieved object never leaves a 10 cm band around the *input* reference ("z-drop"). It reads only
the input, so a reader with no ground truth can reproduce the pick. It cost us real score: 13
rollouts kit-score higher than the ones that ship, and we shipped the rule's picks anyway.

**Tier 3** uses NVIDIA's own modules in their intended order — `v2d_geocalib` → `v2d_moge` →
`v2d_grounding_dino` → a GroundingDINO-to-SAM2 prompt converter we had to write → `v2d_sam2` →
`v2d_foundation_pose`. Placement in the released world is exactly one rigid transform taken from
frame 0, as FAQ PR 207 permits: `world_from_cam = world_from_obj[0] @ inv(cam_from_obj[0])`,
applied unchanged to every later frame, with no scale fitted and no per-frame correction. The
frame-0 residual is 0.0000 cm and 0.000° on all 30 clips.

**Warm-start fine-tuning** (`brev/patch_warm_start.py`) adds `training.warm_start_actor_path`:
after `create_learner` it loads a shipped actor's params and batch stats, sets
`environment_steps` to 299,999,232 so the curriculum sits in its final stage, and runs to the
normal 349,999,104 — about 50M steps, a seventh of a full run, ~35 min and ~$0.57 on an A6000.
The critic starts cold, so the actor's learning rate is scaled ×0.1 for the first 2,000 updates.

Scoring was never reimplemented. `v2d score` hash-checks the kit's four metric files against
`KIT_MANIFEST.json`, imports them and calls their `score()`. Across nineteen uploads the largest
local-to-public disagreement in AUC was 1.1e-05, with the other three metrics matching exactly.

## Experiments & Results

| entry (30 clips) | AUC | SP-SR | MP-SR | MPPE cm | rank |
|---|---|---|---|---|---|
| **Tier 1** | **0.54808** | 0.70000 | 0.50000 | **12.060** | 11 / 23 |
| **Tier 2** | **0.46522** | 0.66666 | 0.46666 | **14.128** | 6 / 18 |
| **Tier 3** | **0.26781** | 0.20000 | 0.03333 | 47.772 | 6 / 12 |
| baseline, Tier 2 | 0.374 | 0.533 | 0.067 | 20.5 | — |
| baseline, Tier 1 | 0.51038 | — | — | — | — |

Ranked by measured contribution rather than by interest:

**`scene.support` per clip — the largest single effect, and it was a bug.** Our first
full-length baseline scored mean AUC 0.043 because all three run scripts passed
`scene.support=false` while NVIDIA's configs default it to `true`. Without the table the object
falls at reset and the episode terminates on frame 0; metrics were bit-identical across all
seven checkpoints, which is the signature. 14 of 30 clips ship a support primitive.

**Warm-start fine-tuning is the one method change that paid, and only where the rules allow
it.** Tier 1 permits the clean multi-view trajectory, which the kit also names as the configured
scoring reference, so a per-clip choice may legally read it. Fine-tuning shipped actors on that
clean reference under a no-regression rule took Tier 1 from 0.46523 to **0.54808** AUC and MPPE
from 14.128 to **12.060 cm**. On Tier 3 the input is our own reconstruction, so selection reads
the input alone; 5 of 10 clips adopted that way, 0.23233 → **0.25059**, turning both success
gates off zero.

**The mechanism is re-fitting to a new reference, not extra training.** A cold swap fails
outright: re-evaluating all 30 shipped actors on the clean Tier 1 trajectory *dropped* mean AUC
from 0.46523 to 0.28937 and SP-SR from 0.667 to 0.138, with only 2 of 30 clips improving (frame
checks ≤1.7e-05 cm, so this is not a conversion artefact). A second generation of fine-tuning on
the *same* reference adopted **0 of 3** and lost e16 outright (0.6469 → 0.2963). Adoption is
predicted by the clip's shipped AUC: 11 of 19 clips above 0.35 adopted at a mean gain of +0.169;
**0 of 4** below 0.35 did.

**Checkpoint selection is worth less than it looks, and an earlier draft overstated it.** Across
all 30 clips with the kit scorer, z-drop is worth **+0.0010** (0.3908 vs 0.3898 final). An oracle
that reads the truth — the ceiling on *any* selection rule — reaches only 0.4201. The rule earns
its place by being reproducible without ground truth, not by the points it adds.

**Where the remaining gap actually lives.** Measured on trajectories rather than on AUC (which
mixes position and orientation, and whose two defects cancel): the noisy Tier 2 reference is
1.94 cm from truth on average while our policies are 13.84 cm, and **19 of 30 clips end up worse
than the reference they were given**. The position error is ours — Tier 2 is policy-bound, not
reference-bound. But only **9 of 30 clips ever leave a 10 cm band** around their reference. So
nine clips drop catastrophically and the rest track a good reference a few centimetres loosely:
**precision across many clips, not robustness on a few**, is where most of the deficit is. Tier 3
is the opposite — eleven clips have *negative* headroom, the policy already tracking better than
the reconstruction it was given.

## Conclusion

A three-tier benchmark that varies only the permitted input rewards treating the input constraint
as the design variable, and we think the transferable results are these: a per-episode selection
rule that reads the input alone is both legal in every tier and reproducible by a reader with no
ground truth; cold-swapping a cleaner reference into a converged policy *loses* score, while
fine-tuning it for a seventh of a run recovers and exceeds it; and the residual failure is
mid-episode release of a well-tracked object, which our own diagnosis traces to the actor being
paid for contact it cannot observe — `envs/recon_body.py` computes the contact state every step
and then discards it, so the finger action through the frames before a drop is necessarily
open-loop.

*Economic viability.* Total spend was **~$144 of cloud credit** for 30 policies across three
tiers and nineteen uploads — a full-length cell is ~$3.00 and a warm-start cell ~$0.57, so the
marginal cost of a new sequence is about the price of a coffee. *Data flywheel.* The selection
and adoption rules read only the input, so every new rollout can be judged and adopted without
any annotation — the loop closes without a human or a ground truth in it. *Complexity reduction.*
One CLI, four patch scripts that refuse unknown upstreams, and a scorer that imports the kit's
own metric code under a hash check, which is why local numbers match the board to five decimals.
*Innovation.* The orientation reward term, the input-only z-drop rule, and warm-start transfer as
a way to move one policy to a different reference cheaply.

Limitations, stated plainly: one seed; no ablation isolates the four recipe changes from one
another, because the shipping run was chosen as a shipping run, knowingly, over an attribution
run; and object inertia is a sim artefact we did not touch — `soma_to_g1` writes every object
URDF as 1 kg with inertia `diag(0.01)`, making the squeegee, cane and vacuum 6–14× too easy to
spin, and those are among our worst clips.
