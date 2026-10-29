# Tier 3, the MPPE push — pre-registered 2026-10-29 17:30 UTC, before staging

## Why this and not something else

The first rank data of the project (`$S/leaderboard_positions.json`, pulled from the twelve public
boards at 17:28 UTC) says where we actually stand:

| board | rank | us | 1st | 2nd | gap |
|---|---|---|---|---|---|
| tier1-auc | 12/23 | 0.52687 | 0.89354 | 0.83047 | 0.367 |
| tier1-mppe | 11/22 | 13.90 | 2.78 | 3.68 | 11.1 cm |
| tier2-auc | 5/18 | 0.46522 | 0.56768 | 0.52267 | 0.102 |
| tier2-mppe | 4/18 | 14.13 | 9.28 | 10.84 | 4.8 cm |
| **tier3-auc** | **3/10** | 0.25059 | 0.33311 | 0.32937 | 0.083 |
| tier3-mppe | 6/10 | 54.44 | 31.34 | 35.01 | 23.1 cm |
| tier3-sp-sr | 7/10 | 0.16666 | 0.40000 | 0.40000 | 0.233 |
| tier3-mp-sr | 6/10 | 0.03333 | 0.10000 | 0.10000 | 0.067 |

Tier 1 is not reachable: closing 0.367 of AUC needs the ~12 weak clips rebuilt, and the measured
rule is that warm-starting a clip below 0.35 never adopts, so that costs full-length training on
clean references — about $36 at $0.68/box-hour, against ~$5 of credit left.

**Tier 3 MPPE is reachable, and it is the largest rank move per dollar on the board.** The Tier 3
fine-tune does not nudge MPPE, it collapses it: e06 went **92.96 → 19.81 cm** and e12 28.17 →
24.15 cm, both adopted on the input-only rule, both improving against truth as well. Pass-1 MPPE
per clip is heavily skewed — e03 150.9, e25 124.1, e04 117.4, e19 88.5, e02 79.5, e10 77.3 — so a
handful of clips carry most of the 54.44 cm mean.

## The arm

Warm-start fine-tune, identical recipe to the one that produced e06/e12/e17: shipped wave-2 actor,
Tier 3 reconstruction as the reference, `start_curriculum,orientation_reward,object_position`,
`object_position_std=0.08`, `object_orientation_std=0.3`, seed 42, steps 299,999,232 →
349,999,104, `scene.support` per clip from the data, one world at eval.

**Clips, worst pass-1 MPPE first:** e03, e25, e04, e19, e02, e10. One box, sequential.

**Cost:** 6 clips x ~35 min + ~25 min bring-up and image build = ~3.9 box-hours = **$2.65** at
$0.68/h, inside the ~$5.03 projected credit and leaving a recovery margin. A second box for
e27 e16 e28 e05 is *not* launched with it: two boxes at once would spend the margin, and if credit
runs out mid-run Brev terminates the box and the clips already paid for are lost.

## The rule, written before the run

Selection is **input-only** (`tools/t3/adopt.py`): a fine-tuned rollout replaces its pass-1
rollout only if it has no drop where pass 1 had one, or it sits closer to the reconstruction it was
driven by. No ground truth enters the verdict — on Tier 3 the input is *not* the scoring reference,
unlike Tier 1, so a truth-based pick would be illegal. Kit scores against truth are computed and
reported, never used to decide.

**Registered expectation.** At the rate measured on the first 12 Tier 3 clips (7 adopted) and the
MPPE collapse seen on e06, six clips project to a mean MPPE of about **40-45 cm** from 54.44, i.e.
3rd or 4th on that board, with AUC rising a little and SP-SR possibly flipping a clip or two as
errors cross the 10 cm gate. **The arm counts as having worked if the entry's mean MPPE falls by
at least 5 cm and no board falls.** The entry is only re-uploaded if that holds.

**Known risk.** e03 and e25 are the two clips whose *Tier 1* fine-tunes behaved worst (e03 is the
only strong-four clip that was rejected), and a 150 cm pass-1 error may mean the reconstruction
itself is wrong rather than the policy. If both come back unadopted, the remaining credit should
not buy more of this arm.
