# Visual failure review of the 30-clip combined run (2026-10-27)

Source: frame strips in `v2d-viz/out/t2_combined/strips/` (0/25/50/75/100% plus the failure frame
from `results/second_wave_triage.json`), reviewed by the orchestrator for the twelve clips under
0.30 kit AUC and three strong clips (e24, e12, e03). Board entry at the time: AUC 0.39076, 9th of 18.

## Four failure modes, in order of score cost

1. **Pantomime: the body does the motion, the object is left behind.** e04 bat, e25 water bottle,
   e19 cane, e28 squeegee. The robot swings, lifts and sweeps like the human, arms overhead, while
   the object lies where it fell. Body-tracking reward (weight 5) dominates the object term
   (weight 1, std 0.2 m), so once a grasp is marginal the policy earns almost full reward finishing
   the motion without the object. The finger targets were palm-limited on most hands (retarget palm
   inside the object), so closure stayed weak; thin, light one-hand objects are flung on the first
   fast arm motion. This is the ceiling on the whole run and none of the four proposals in
   `docs/SECOND_WAVE.md` addresses it directly.
2. **Late-phase loss after tracking 80-98% of the clip.** e09 bowl dropped as it is set down;
   e05 box rotated correctly then left lying while the reference stands it up; e07 iron tips at
   release. Consistent with the START curriculum: late training almost always starts at frame 0
   and terminates on loss, so the end of the clip is the least-practised part and recovery is never
   learned. Proposal 1 (revert START) targets this.
3. **Heavy floor objects toppled on first contact.** e14 round table, e28 squeegee: knocked over
   in the first quarter by a two-hand approach that does not land together; the body keeps
   performing afterwards. Early bimanual contact, not tracking.
4. **Orientation reward is dead where it is needed.** At std 0.3 rad the term contributes nothing
   at the 50-90 degree errors of the fell-back clips. Proposal 2 (std 0.8) targets this.

## What works

Objects grasped from frame 0 with a secure hold (e24 watering can, e03 mug) and floor-scale objects
moved with both hands flat on them (e12 platform, e08 chair, e06 monitor) track within a few
centimetres throughout.

## Implication for the second wave

Add a fifth arm for failure mode 1: make losing the object expensive on hand-held clips (tighter
object position std or higher object weight), tested on e04 e25 e19 e28 at ~$3 per clip-run,
before the wave launches. Modes 2 and 4 are covered by proposals 1 and 2 as written.
