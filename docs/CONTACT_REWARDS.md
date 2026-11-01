# Two reward terms upstream already implements and our recipe cannot reach

Desk research, 2026-11-01, no compute. It starts with a hypothesis of mine that turned out to be
wrong, and the thing I found while checking it is better than the hypothesis was.

## The hypothesis, and why it was wrong

CHORD's paper (arXiv:2607.00033, *Learning Dexterous Manipulation Using Contact Wrench Guidance
From Human Demonstration*) describes its contribution as **object-centric contact wrench space
guidance** — "we represent human and robot motions by the forces and torques they can induce on the
object" — and trains with "task-tracking, motion-imitation, and contact-guidance rewards", where
the whole-body variant uses "the reduced force-closure objective". It reports a 90.77% whole-body
success rate.

MBZUAI-RCL's Tier 1 MP-SR is 93.33%, ours is 50%, and the published figure is 90.77%. The obvious
guess was that the leaders implemented the paper's contact objective and we had been training on a
pose-tracking recipe without it.

**That guess is false.** `configs/objective/recon_body.yaml` in NVIDIA's own repo has
`force_closure_weight: 5.0` — tied with `body_joint_position_weight` as the largest positive
weight in the recipe — and `objectives/recon_body.py:158` shows it gated on the reference's own
contact flags (`if contact_active[frame * num_hands + hand] > 0.5`). We have been training with the
paper's contact objective all along. The 90.77% is also on the paper's own 4,739-task benchmark,
not this leaderboard, so the resemblance to 93.33% is not evidence of anything.

## What checking it turned up

`ReconBodyObjectiveConfig` is a **fixed twelve-term objective**: its `weights` tuple and the Warp
composition kernel both hardcode the term list (`objectives/recon_body.py:107-120`,
`objectives/composition.py:110-112`). The objective *library* beside it implements three contact
terms that are not in that list:

| term | upstream implementation | weight where it is used | what it measures |
|---|---|---|---|
| `contact_wrench_support` | `objectives/contact.py:120` | **10.0** in `objective/tracking.yaml` | how much of the commanded wrench the agent can exert |
| **`missed_contact`** | `objectives/contact.py:168` | **−1.0** | "fraction of commanded basis directions per body that the agent failed to cover" |
| `unintended_contact` | `objectives/contact.py:206` | **−10.0** | contact on bodies "the reference doesn't ask for" |

`recon_body` reaches none of them. It has the *positive* force-closure reward and **no penalty for
failing to cover the commanded wrench directions at all.**

That asymmetry lines up with our failure mode exactly. Of thirty clips, ten do not keep hold of the
object — six lost mid-clip, three fell back, one never locks on
(`docs/METRICS_DEEP_DIVE.md`) — and those ten carry most of the Tier 1 deficit. A reward that pays
for force closure while contact is active gives nothing back when the hand quietly stops covering
the directions that hold the object; `missed_contact` is a dense penalty on precisely that, written
in the wrench basis the paper's method is built on.

It is also, notably, **upstream's own implementation of what arm CR hand-rolled.** CR rewarded
palm-to-object-surface distance with a reward we wrote ourselves
(`brev/patch_contact_role.py`); `missed_contact` measures the same failure in force space, is
already written in Warp, and is already exercised by two sibling configs.

## What it would take

A patch of the same shape as the three we already have, and the honest part is that it is the
largest of them. The term list is fixed in three places that must agree — the `weights` tuple, the
`ObjectiveTermConfig` fields, and the `terms[base + n]` indices in the composition kernel — so
adding two terms means extending all three and keeping the index arithmetic consistent. It is
mechanical but it is not a one-line config override, and it must refuse to apply to unknown
upstream the way `patch_object_orientation.py` and `patch_warm_start.py` do.

That work is **free and can be done on the Jetson now**: implement, unit-test the kernel against a
hand-computed value on CPU, and Hydra-compose the arm so it is ready to launch. It cannot be
*validated* without a box.

## Where this sits against the rest of the queue

| | arm | cost | case |
|---|---|---|---|
| cheapest and most certain | the two Tier 1 gate flips (e05, e16) | **~$1.30** | 2 rank points, zero downside, margins far inside the noise floor (`docs/CHEAPEST_RANKS.md`) |
| highest upside | `missed_contact` + `unintended_contact` on the ten drop clips | ~$30 | the only arm aimed at the mechanism that actually loses us Tier 1 |
| already built, never scored | arm CR | $20 | the same idea in distance space rather than force space; killed at 14M of 350M steps by my own cap-guard bug for $0.95 (`results/owed_answers.json`) |

If both exist, the force-space arm should be preferred over CR, and for the same reason the
hypothesis above was worth checking: upstream's version is tested, and ours is not.

## What stays true

This does not change the ceiling finding. Our eighteen working Tier 1 clips average 0.7506 AUC and
first place needs every clip at 0.9567, our single best in thirty. A better contact penalty is a
plausible route to the drop clips, not to the leaders' typical clip.

## Sources

- [CHORD: Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration (arXiv:2607.00033)](https://arxiv.org/abs/2607.00033)
- [ResMimic: From General Motion Tracking to Humanoid Whole-body Loco-Manipulation via Residual Learning (arXiv:2510.05070)](https://arxiv.org/html/2510.05070v1) — residual policy on a tracking prior with a point-cloud object-tracking reward, also on a Unitree G1; the closest published architecture to something that would beat our recipe outright, and out of reach in the time left.
- [V2D Challenge page](https://nvidia-isaac.github.io/video_to_data/v2d_challenge/) — no winning method is published for Track 2; the search found only other entrants' in-progress repositories.
