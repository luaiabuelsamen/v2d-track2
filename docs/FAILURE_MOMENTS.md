# What happens at the failure moment

Five clips fail while the object is nearly still, so the speed correlation that explains e04, e07, e14 and e25 cannot explain them. This is the one-second window before each one, measured rather than watched.

**Where each number comes from.** The object track is the policy's own, from the rollout the entry ships. The hands, the contact flags and the body are the *reference's*: the rollout export carries the object alone, so there is no record of where the policy's own hands were. "The hand opens" below always means the hand the policy was tracking opened. Whether its own hand followed is not in the data, and the distinction matters most for exactly these clips -- a policy that never closed its hand and a policy whose reference told it to let go look identical in the object track.

**e05 big_black_box** (AUC 0.266) is lost at frame 602; the window is frames 572 to 602. The object is moving at 0.18 m/s at most in that second, against 1.09 m/s over the clip, and the reference moves it at 0.16 m/s. Its object is within 8.4 cm of the reference when the window opens and 10.3 cm at the failure frame. In the reference, the left palm is 6.8 cm from the surface of the object it is meant to be holding, so the reference has it in hand; the same palm is 4.2 cm from the policy's own object at the window start and 12.7 cm at the failure frame. The reference hand goes from 0.05 rad of mean finger flexion to 0.01 rad, so it does not open. The reference's own contact flags change here: left off (3 changes). The clip ships no support primitive; the object is at 0.23 m at the failure frame, from 0.21 m when the window opened. The body is moving at 0.14 m/s on average, so the robot is walking through this. **Mechanism: contact moves between hands.**

**e09 big_red_bowl** (AUC 0.300) is lost at frame 852; the window is frames 822 to 852. The object is moving at 0.00 m/s at most in that second, against 0.90 m/s over the clip, and the reference moves it at 0.04 m/s. Its object is within 4.7 cm of the reference when the window opens and 4.4 cm at the failure frame. In the reference, the left palm is 74.7 cm from the surface of the object it is meant to be holding, so the reference is not holding it here either; the same palm is 71.1 cm from the policy's own object at the window start and 79.4 cm at the failure frame. The reference hand goes from 0.00 rad of mean finger flexion to 0.00 rad, so it does not open. The reference's contact flags do not change in the window (no hand active). The object is at 0.07 m with the support top at 0.47 m, so it does not reach it. The body is moving at 0.17 m/s on average, so the robot is walking through this. **Mechanism: the shipping rollout does not fail here (still on its reference).**

**e17 traffic_cone** (AUC 0.447) is lost at frame 399; the window is frames 369 to 399. The object is moving at 0.00 m/s at most in that second, against 0.96 m/s over the clip, and the reference moves it at 0.09 m/s. Its object is within 2.7 cm of the reference when the window opens and 2.8 cm at the failure frame. In the reference, the right palm is 25.7 cm from the surface of the object it is meant to be holding, so the reference is not holding it here either; the same palm is 26.1 cm from the policy's own object at the window start and 78.4 cm at the failure frame. The reference hand goes from 0.00 rad of mean finger flexion to 0.00 rad, so it does not open. The reference's contact flags do not change in the window (no hand active). The object is at 0.87 m with the support top at 0.77 m, so it does not reach it. The body is moving at 0.25 m/s on average, so the robot is walking through this. **Mechanism: the shipping rollout does not fail here (still on its reference).**

**e19 cane** (AUC 0.116) is lost at frame 133; the window is frames 103 to 133. The object is moving at 0.00 m/s at most in that second, against 0.10 m/s over the clip, and the reference moves it at 0.30 m/s. Its object is within 1.1 cm of the reference when the window opens and 10.1 cm at the failure frame. In the reference, the right palm is 6.6 cm from the surface of the object it is meant to be holding, so the reference has it in hand; the same palm is 6.8 cm from the policy's own object at the window start and 17.8 cm at the failure frame. The reference hand goes from 0.08 rad of mean finger flexion to 0.08 rad, so it does not open. The reference's contact flags do not change in the window (right active throughout). The clip ships no support primitive; the object is at 0.38 m at the failure frame, from 0.38 m when the window opened. The body is moving at 0.20 m/s on average, so the robot is walking through this. **Mechanism: the object slides out of the grip.**

**e28 squeegee** (AUC 0.089) is lost at frame 118; the window is frames 88 to 118. The object is moving at 0.48 m/s at most in that second, against 0.62 m/s over the clip, and the reference moves it at 0.10 m/s. Its object is within 1.9 cm of the reference when the window opens and 10.7 cm at the failure frame. In the reference, the left palm is 23.5 cm from the surface of the object it is meant to be holding, so the reference is not holding it here either; the same palm is 23.0 cm from the policy's own object at the window start and 18.2 cm at the failure frame. The reference hand goes from 0.00 rad of mean finger flexion to 0.00 rad, so it does not open. The reference's contact flags do not change in the window (no hand active). The object is at 0.54 m with the support top at 0.72 m, so it does not reach it. The body is moving at 0.07 m/s on average, so the robot is walking through this. **Mechanism: the policy's object is flung far faster than the reference moves it.**

## Two of the five frames are from a rollout we no longer ship

The failure frames come from the first wave's triage. **e09** now ships the held but drifting rollout from a second-wave arm, whose object is 4.4 cm off its reference at frame 852 and **e17** now ships the held but drifting rollout from a second-wave arm, whose object is 2.8 cm off its reference at frame 399. Neither has a loss event anywhere in the clip under the taxonomy's rules, so there is no moment to explain: what they lose is the angle, which is the held-but-drifting class and a different fix. The windows are reported above all the same, because 'nothing happens here' is the answer to the question asked.

## The mechanisms

| clip | object | AUC | failure frame | off reference at window start | object speed in window | reference palm on object | mechanism |
|---|---|---|---|---|---|---|---|
| e05 | big_black_box | 0.266 | 602 | 8 cm | 0.18 m/s | 7 cm | contact moves between hands |
| e09 | big_red_bowl | 0.300 | 852 | 5 cm | 0.00 m/s | 75 cm | the shipping rollout does not fail here (still on its reference) |
| e17 | traffic_cone | 0.447 | 399 | 3 cm | 0.00 m/s | 26 cm | the shipping rollout does not fail here (still on its reference) |
| e19 | cane | 0.116 | 133 | 1 cm | 0.00 m/s | 7 cm | the object slides out of the grip |
| e28 | squeegee | 0.089 | 118 | 2 cm | 0.48 m/s | 24 cm | the policy's object is flung far faster than the reference moves it |

## The objects are all 1 kg boxes, and some are far too easy to spin

`soma_to_g1.py` writes every object URDF with mass 1.0 kg and inertia diag 0.01. `I_ratio_sim` is that 0.01 against the object's own shape inertia at the same 1 kg: below 1 means the simulated object spins more easily than the real shape would, and `1/ratio` is how many times too easy. A long thin object is the worst case, because its real inertia about the short axes is large while the URDF's is not.

| clip | object | longest axis m | shape inertia max | I_ratio_sim | times too easy to spin | AUC |
|---|---|---|---|---|---|---|
| e28 | squeegee | 1.54 | 0.1347 | 0.07 | 14.3x | 0.089 |
| e10 | white_desk | 1.03 | 0.1377 | 0.07 | 14.3x | 0.315 |
| e14 | round_table | 0.79 | 0.0839 | 0.12 | 8.3x | 0.109 |
| e05 | big_black_box | 0.86 | 0.0771 | 0.13 | 7.7x | 0.266 |
| e12 | black_platform | 0.81 | 0.0751 | 0.13 | 7.7x | 0.710 |
| e21 | tall_bar_stool | 0.89 | 0.0655 | 0.15 | 6.7x | 0.206 |
| e00 | tall_bar_stool | 0.89 | 0.0655 | 0.15 | 6.7x | 0.498 |
| e01 | vacuum | 1.05 | 0.0613 | 0.16 | 6.2x | 0.200 |
| e19 | cane | 0.91 | 0.0588 | 0.17 | 5.9x | 0.116 |
| e08 | skinny_wood_chair | 0.88 | 0.0594 | 0.17 | 5.9x | 0.443 |
| e29 | rolling_luggage | 0.98 | 0.0600 | 0.17 | 5.9x | 0.593 |
| e18 | blue_trash_can | 0.71 | 0.0512 | 0.20 | 5.0x | 0.736 |

This is reported, not acted on. The URDFs are not edited here: whether object physics may be changed at all is a rules question for Luai, and if the answer is no, the finding still stands as a reason the long thin objects behave as they do.
