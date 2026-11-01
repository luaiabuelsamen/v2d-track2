# Gate flips: where the next Tier 1 rank points actually are

> **Superseded in part, 2026-11-01.** This file prices rank gains on the assumption that ties rank behind and that the twelve boards sum into one standing. Both are wrong — NVIDIA's aggregator shares tied ranks and scores the three tiers separately (`docs/RANKING_RULE.md`). Claims marked **[DEAD]** below are worth zero rank points; the measurements are unaffected.

*2026-10-30, against the leaderboard pulled the same hour. Rank sum 86 of 12 boards.*

Every arm this project ran was judged on per-clip **AUC**, and every closure argument was written
in metric space: the entry is Pareto-optimal, nothing we hold improves a board, the remaining gap
is a better tracker bought with training scale. The first two are still true. The third is true of
the leaders and **false of the next two ranks**, and measuring at the gate instead of at the metric
is what shows it.

## The board, with the distance to the next rank

| board | us | rank | next up | distance |
|---|---|---|---|---|
| tier1-auc | 0.54808 | 11/20 | crisiumnih 0.58226 | +0.0342 AUC |
| tier1-sp-sr | 0.70000 | 12/20 | — *tied* with CHORD and wte09fl | **any gain at all** |
| tier1-mp-sr | 0.50000 | 11/20 | wte09fl 0.53333 | +2 clips (0.53333 only ties) |
| tier1-mppe | 12.06020 | 10/20 | DAP 11.99700 | −0.0632 cm |

**[DEAD]** Ties *do* share a rank, so the next two sentences are void; the rung on that
board is 25 of 30 clips, not 22 (`docs/RANKING_RULE.md`).
SP-SR is a three-way tie at 0.70000 and ties do not promote, so **one clip flipping takes 0.70000
to 0.73333 and the rank from 12 to 10 — two rank points for one clip.**

## The rank-objective swap search

The earlier Pareto proof asked whether a swap improves one metric without hurting another. That is
the wrong objective: the challenge ranks by the **sum of per-metric ranks**, so a swap that spends
AUC inside our 0.00625 cushion to buy MPPE past DAP is worth a rank point. Re-ran the search over
all 67 held alternative rollouts across 30 clips, scoring every 1-, 2- and 3-swap against the live
board (`tools/t1/rank_swap.py`).

**No combination lowers the rank sum.** The closest: taking the lowest-MPPE rollout we hold on
every clip reaches 12.01783 cm, **0.0208 cm short** of passing DAP, while holding the other three
ranks. 0.63 cm summed across 30 clips is the entire distance to that rank point.

## The two clips at the gate

SP-SR passes when the time-mean position error is ≤ 0.1 m **and** the time-mean geodesic
orientation error is ≤ 0.5 rad (kit `SP-SR.py:132-141`). Measured per clip on the uploaded entry
(`tools/t1/gate_margins.py`, table in `results/t1_gate_margins.json`):

| clip | pos | orientation | misses SP-SR by |
|---|---|---|---|
| **e16** | 1.86 cm | 0.522 rad | **0.022 rad — 1.3°** |
| **e05** | 10.32 cm | 0.308 rad | **0.32 cm** |
| e09 | 4.19 cm | 0.670 rad | 0.170 rad |
| e13 | 4.07 cm | 0.701 rad | 0.201 rad |
| e07 | 9.40 cm | 1.839 rad | 1.339 rad |
| e01 e19 e25 e04 | 43–102 cm | 0.6–1.9 rad | dropped objects, not near-misses |

e16 is 1.3° of orientation from two rank points, with position 8 cm inside its limit. It also
*passes* MP-SR by 0.09°, so the orientation has to come down, not move. e05 is 0.32 cm of position
from the same two rank points, with orientation comfortably inside.

MP-SR near-misses, all pure position (gate 3 cm): e11 +0.63, e29 +0.84, e13 +1.07, e09 +1.19,
e02 +1.34, e21 +1.43 cm. Two must flip, because 0.53333 only ties wte09fl.

## What it would cost

A 50M-step warm start is ~0.63 GPU-hours, so ~**$0.55 per cell** at $0.68/h with provisioning.

| set | cells | cost | upside |
|---|---|---|---|
| e16, e05 (SP-SR), e29, e11 (MP-SR pair), 2 seeds | 8 | **~$5** | +2 SP-SR, +1 MP-SR, +1 MPPE |
| add e09 e13 e21 e02 | 16 | ~$9 | same, more chances |

**Honest odds:** low per cell. Second-generation warm starts adopted 0 of 3, and e16's own chain
rollout fell 0.6469 → 0.2963. What is different is the *target*: a gate flip of 1.3° or 0.32 cm,
not an AUC record, so the criterion that failed those arms does not apply here. Breadth, not
selection, is what pays — the donor transfer was 3 of 9 and heavy-tailed, and selection was
six-for-six refuted.

**Blocked on Brev credit.** Two creates return `Bad Request`; nothing runs until the account is
topped up. The freeze is 2026-11-04 17:00 EST.

## One caution on the other two tiers

The same margin table could be built for Tier 2 and Tier 3, but it reads the clean truth. On Tier 1
that is legal — `config/tracks.json` makes the released multiview trajectory both the permitted
input and the scoring reference. On Tier 2 and Tier 3 it is not, and using it even to *choose which
clips to retrain* would be reading truth we are not given. If those tiers get spend, the targeting
has to come from the input reference.
