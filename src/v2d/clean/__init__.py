"""Cleaners for the Tier 2 reference.

Tier 2 perturbs exactly three things: the object pose, the human **root joint only**, and the
human root translation. All 76 non-root joints are bit-identical to Tier 1, so a cleaner that
touched them would be inventing a correction to data with no error.

Two cleaners live here, and the contrast between them is the project's main methodological
result:

- `support_contact` works, because its reference is **external** to the noisy data. A floor
  normal and a support surface's height are geometric facts about the room, independent of the
  trajectory being cleaned. It lifts the Tier 2 input ceiling from AUC 0.4247 to 0.4755 with no
  metric regressing.
- `root_planting` fails, because its reference is **internal**. "A planted foot does not move"
  constrains relative motion only, and its reference pose is the median of the *noisy* foot
  positions, itself displaced by that segment's mean perturbation. It makes the root rotation
  error worse (5.17 -> 5.58 deg) and nearly doubles the translation error.

A cleaner needs an external reference, not merely a self-consistency condition.
"""
