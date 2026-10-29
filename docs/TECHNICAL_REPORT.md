# V2D Challenge, Track 2 (Robotic Grounding) — technical report

Team: Luai Abuelsamen (University of California, Berkeley).
Code: <https://github.com/luaiabuelsamen/v2d-track2>, released under Apache-2.0 (`LICENSE`).
Platform: an NVIDIA Jetson for all CPU work, and rented single-A6000 cloud boxes for training.

## 1. What the task turned out to be

All three Tier 2 tiers are scored against the same `multi_view_ground_truth`; only the permitted
*input* differs. We verified this the hard way, after noticing our Tier 1 and Tier 2 public scores
were identical: the two CSVs are byte-identical, and a Kabsch fit shows every shipped rollout's
reference is an exact (0.000 cm) rigid transform of the released Tier 2 trajectory. Identical
predictions therefore earn identical scores, and the tiers differ only in what you are allowed to
condition on. That single fact set the whole strategy: the object trajectory is the deliverable,
and the policy exists to produce a physically realisable one.

## 2. Pipeline

SOMA-X human parameters → `soma_to_g1` retarget to a Unitree G1 with Dex3 hands (CPU, 30/30 clips
at 0.0 mm residual against the released object trajectory) → FlashCHORD SAC training, one policy
per sequence, 349,999,104 steps, `world_count=1`, `scene.support` per clip from the shipped
`support_surface` primitives → `evaluate_policy.py` → per-episode checkpoint selection → the kit's
own packer.

Scoring was never reimplemented: `v2d score` hash-checks the kit's four metric files against
`KIT_MANIFEST.json`, imports them and calls their `score()`, so local numbers are leaderboard
numbers by construction. Across eight uploads the largest local-to-public disagreement was
1.1e-05 in AUC, with the other three metrics matching exactly.

## 3. What actually moved the score

Ranked by measured contribution, not by how interesting it was.

**`scene.support` per clip (largest single effect).** Our first full-length baseline scored a mean
AUC of 0.043 because all three run scripts passed `scene.support=false` while NVIDIA's configs
default it to `true`. Without the table the object falls at reset and the episode terminates on
frame 0: metrics were bit-identical across all seven checkpoints, which is the signature. 14 of
30 clips ship a support primitive. This one flag is also why ~12 clips in our first Tier 1 entry
were near zero.

**Per-episode checkpoint selection under an input-only rule.** The provenance schema allows a
different `checkpoint_sha256` per episode. We pick, per clip, the latest checkpoint whose achieved
object never leaves a 10 cm band around the *input* reference ("z-drop"). It reads only the input,
never the truth, so it is reproducible by a reader who has no ground truth — and it is registered
as the submission rule rather than chosen after the fact. It cost us something real: 13 rollouts
kit-score higher than the ones that ship, and we shipped the rule's picks anyway.

**What selection is *not* worth is score, and an earlier draft of this section overstated it.**
On four clips, measured with the evaluator's own metric, z-drop beat the final checkpoint by
+0.0705 (0.2745 against 0.2040), and that figure put selection second in this ranking. Across all
30 clips with the kit scorer it is worth **+0.0010**: z-drop 0.3908, latest-drop-free 0.3946, final
checkpoint 0.3898. An oracle that reads Tier 1 truth — the ceiling on *any* selection rule — reaches
only 0.4201, so checkpoint selection cannot be worth more than +0.029 and no submittable rule
collects even that (`docs/SECOND_WAVE.md`). The rule earns its place by being reproducible without
ground truth, not by the points it adds. The 0.21 of headroom to the reference ceiling is a policy
problem.

**Reward shaping.** An object-orientation term (weight 1.0, std 0.3) added to `recon_body`, which
has an `object_position_weight` but no orientation counterpart; a start-curriculum patch; tightened
object-position std (0.2 → 0.08 m) on the clips that were already working; finger targets and
bridge-smoothed references. Each is a patch script that refuses to apply to an upstream it does not
recognise, so a silent half-patch is impossible.

**Entry.** Mean kit AUC 0.46523 / SP-SR 0.66667 / MP-SR 0.46667 / MPPE 14.128 cm across 30 clips,
against the published CHORD Tier 2 baseline of 0.374 / 0.533 / 0.067 / 20.5 cm. MP-SR passes on 14
of 30 clips, up from 5.

## 4. Tier 3: bring your own reconstruction

NVIDIA's own modules, in their intended order: `v2d_geocalib` → `v2d_moge` → `v2d_grounding_dino`
→ a GroundingDINO-to-SAM2 prompt converter we had to write → `v2d_sam2` → `v2d_foundation_pose`
(`nvlabs_pytorch` backend). MegaPose was the original plan and was abandoned: its only weight host
is unreachable and every mirror we probed returned 403 or 404.

Placement in the released world uses exactly one rigid transform taken from frame 0, which is what
FAQ PR 207 permits: `world_from_cam = world_from_obj[0] @ inv(cam_from_obj[0])`, applied unchanged
to every later frame. No scale is fitted and no per-frame correction is applied. The frame-0
residual is 0.0000 cm and 0.000° on all 30 clips, which simultaneously confirms the quaternion
convention — `Transform3d` is imported from upstream rather than reimplemented.

Reconstruction quality is bimodal rather than uniformly mediocre: 10 clips track to under 5° mean
orientation drift and 16 cm mean position error; 20 lose the object. Scored raw against truth the
reconstruction is AUC 0.228 / SP-SR 0.333 / MP-SR 0.033 / MPPE 66.6 cm.

## 5. Negative results worth recording

**Clean input does not help a policy trained on a different reference.** Tier 1 permits the clean
multi-view trajectory, so we re-evaluated all 30 shipped actors on it. Mean AUC *fell* from 0.46523
to 0.28937 and SP-SR from 0.667 to 0.138; only 2 of 30 clips improved. Frame checks were at or
below 1.7e-05 cm, so this is not a conversion artefact. Each actor learned to track one specific
trajectory, so substituting a cleaner one is a distribution shift, not an upgrade. The implication
is that using a new reference requires fine-tuning, not a cold swap.

**The root cleaner.** A floor/table support cleaner for the Tier 2 references measured worse than
raw and was kept only behind a gate. Its original ~270 cm body-to-plane distance turned out to be
an unflipped frame: under the retargeter's `diag(1,-1,-1)` the feet sit at 2.0 cm, on 30/30 clips.

**A learned denoiser.** Without an absolute anchor it could remove only ~6% of the Tier 2 error,
and a grasping hand is a loose anchor (4.9 cm and 44° of wander), so it was dropped.

## 6. Honest limitations

One seed. No ablation isolates the four recipe changes from each other — the shipping run was
chosen as a shipping run, knowingly, over an attribution run. The remaining failure mode is
mid-episode drops rather than a broken pipeline: trajectories show the object tracked to ~1 cm for
50–100 frames and then released, which is a policy problem. Object inertia is a sim artefact we
did not touch: `soma_to_g1` writes every object URDF as 1 kg with inertia `diag(0.01)`, which makes
the squeegee, cane and vacuum 6–14× too easy to spin.

## 7. Reproducing

`README.md` has the pipeline and the results table. Every number in `results/` is reproducible
from a committed script; every Brev run is a committed runner with its cost in the ledger. Total
training spend was ~$120 of NVIDIA Brev credit across the first wave, the second wave and the
Tier 3 reconstruction.
