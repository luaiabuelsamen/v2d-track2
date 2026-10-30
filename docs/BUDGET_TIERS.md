# Where we stand, and what each budget buys

*2026-10-30 18:11 UTC. 124 hours to the freeze (Nov 4, 17:00 EST). All leaderboard numbers pulled
this hour; all cost figures measured from boxes we actually ran, not quoted from a price list.*

## Where we stand

Rank sum **86** across the twelve boards. The challenge ranks by the sum of per-metric ranks, so
that number — not any single score — is the thing to move.

| board | us | rank | board | us | rank | board | us | rank |
|---|---|---|---|---|---|---|---|---|
| tier1-auc | 0.54808 | 11/20 | tier2-auc | 0.46522 | 5/18 | tier3-auc | 0.26781 | 4/11 |
| tier1-sp-sr | 0.70000 | 12/20 | tier2-sp-sr | 0.66666 | 5/18 | tier3-sp-sr | 0.20000 | 6/11 |
| tier1-mp-sr | 0.50000 | 11/20 | tier2-mp-sr | 0.46666 | 5/18 | tier3-mp-sr | 0.03333 | 7/11 |
| tier1-mppe | 12.06020 | 10/20 | tier2-mppe | 14.12760 | 4/18 | tier3-mppe | 47.77180 | 6/11 |

It was 84 yesterday. wte09fl uploaded Tier 3 at 14:50 today and took two of our ranks there; nothing
of ours regressed. **The board moves on its own, so standing still loses ground.**

Free moves are exhausted, and now proven against the right objective: over all 67 alternative
rollouts we hold, every 1-, 2- and 3-swap, **nothing lowers the rank sum** (`tools/t1/rank_swap.py`).

## The price list, in rank points

This is what each board costs to climb one place. Ties do **not** promote — Kaggle breaks them by
submission time and everyone we tie with uploaded earlier — so matching the score above us is worth
nothing.

| board | one rank point costs | two points cost |
|---|---|---|
| **tier1-sp-sr** | **+1 clip → +2 points at once** (12 → 10, past CHORD and wte09fl) | — |
| **tier3-mp-sr** | **+1 clip → +2 points** (7 → 5) | +3 clips → **rank 1**, +6 points |
| **tier1-mppe** | **−0.0632 cm** | −0.194 cm |
| tier3-auc | +0.0139 | +0.0616 |
| tier2-auc | +0.0155 | +0.0486 |
| tier1-auc | +0.0342 | +0.0345 *(two teams 0.0003 apart)* |
| tier1-mp-sr | +2 clips | +3 clips |
| tier2-sp-sr | +2 clips | +3 clips → **3 points** (5 → 2) |
| tier2-mp-sr | +2 clips | +3 clips |
| tier3-sp-sr | +2 clips | +3 clips |
| tier2-mppe | −2.60 cm | −3.28 cm |
| tier3-mppe | −5.47 cm | −9.17 cm |

Two of those are nearly free if anything runs at all:

- **e16 misses Tier 1 SP-SR by 1.3° of orientation**, with its position 8 cm inside the limit.
- **e05 misses it by 0.32 cm of position**, with its orientation comfortably inside.

Either one flipping is **two rank points**. This is the best-characterised target this project has
produced, and it was invisible for a week because every arm was judged on AUC and a success rate is
a gate, not a metric (`docs/GATE_FLIPS.md`).

MP-SR near-misses, all pure position against a 3 cm gate: e11 +0.63, e29 +0.84, e13 +1.07,
e09 +1.19, e02 +1.34, e21 +1.43 cm. Two must flip to beat wte09fl.

## What things actually cost

Measured, from `results/brev_spend_warmstart.json` and `results/brev_spend_transfer_run.json`:

| unit | measured cost | what it is |
|---|---|---|
| a box | $0.55 | bring-up, cold image build, asset upload |
| warm-start cell | **$0.57** | 50M steps from an existing actor, eval, pull |
| full-length cell | **~$3.00** | 350M steps from scratch, 4.4 GPU-hours |
| 8 boxes for a day | $131 | the concurrency we have actually run |

And what spend has historically returned:

| arm | spend | result |
|---|---|---|
| donor transfer, Tier 1, 6 clips | $2.23 | 3 of 9 adopted; **1 rank point** plus the SP-SR lift that became another |
| Tier 3 MPPE push | $4.12 | **1 rank point** (mppe 6th → 5th) |
| second wave, Tier 2, full length | ~$85 | Tier 2 AUC 0.39076 → **0.46522**, moved all four Tier 2 boards |
| orientation reward arm | ~$2 | 0 of 2. Collecting orientation broke the 3 cm MP-SR gate |
| second-generation warm start | ~$3 | 0 of 3. Re-fitting the same reference twice does nothing |

**So: roughly $2–4 per rank point while targeted near-misses last, and about $85 for the kind of
broad retraining that moves four boards at once.** The cheap arms are heavy-tailed — three of nine
transfer clips paid for all of it — so breadth is what buys outcomes, not clever selection, which
has been refuted six times out of six on this project.

## Time is not the constraint; money is

124 hours left. At the 8 concurrent boxes we have actually run, that is **$630 of capacity**, or
$590 after reserving 8 hours to pack, score and upload. Anything up to ~$250 fits comfortably in
two days and leaves slack for the one-in-three Brev create failures.

## The tiers

Each builds on the one above. Expected ranges are honest: the low end assumes the historical hit
rates hold, the high end assumes they hold *and* the near-misses flip.

### $12 — the gate flips
20 warm-start cells over 7 boxes, about 6 hours.

- e16 (orientation) and e05 (position), 3 seeds each — the two SP-SR near-misses
- e11 and e29, 3 seeds each — the MP-SR pair
- the four remaining untried Tier 1 transfer clips (e21, e09, e02, e17) for AUC and MPPE
- one Tier 3 box aimed at its MP-SR gate

**Targets:** tier1-sp-sr +2, tier1-mppe +1, tier1-mp-sr +1, tier3-mp-sr +2.
**Expected: 2–5 rank points. 86 → 81–84.**
This is the highest return per dollar on the table and the only tier I would call likely.

### $40 — add the tier the transfer arm has never touched
Plus 40 cells over about 14 boxes, about a day.

The donor transfer has never been run on **Tier 2**, where 19 of 30 clips score worse than their own
input reference. Tier 2 is also where one more flip is worth the most: +3 clips on tier2-sp-sr is
**three rank points** (5th → 2nd). Plus Tier 3 warm starts for its AUC point (+0.0139).

**Expected: 5–9 rank points. 86 → 77–81.**

### $120 — retrain where warm starts plateau
Plus roughly 30 full-length cells, about two days on 8 boxes.

Warm starts can only re-fit an existing policy; they cannot fix a clip that drops the object. The
precision recipe (object_position std 0.08) is what took Tier 2 from 0.391 to 0.465 and **has never
been run on Tier 1 at all**. Its eleven weak Tier 1 clips from scratch is ~$34; a second seed on
Tier 2's tail is another ~$40.

**Expected: 8–13 rank points. 86 → 73–78.**

### $250 — everything, two seeds
About two days on 8–11 boxes, inside the freeze with room to spare.

Adds a second seed across all 30 clips on both Tier 1 and Tier 2, and a real attempt at the
tier3-mp-sr swing: 1/30 → 4/30 takes that board from **7th to 1st**, six rank points, and it is the
single largest rank move available anywhere. It is also the least likely, because 1/30 passing means
our Tier 3 rollouts are nowhere near the 3 cm gate on 29 clips.

**Expected: 12–20 rank points. 86 → 66–74.**

## What no budget buys by Nov 4

- **Tier 1's top three.** MBZUAI is at 0.89354 against our best 19 clips averaging 0.7228. That is a
  better tracker, not more runs of ours.
- **Tracks 1 and 3 of the challenge.** Different data and pipeline; checked and closed
  (`results/other_tracks_feasibility.json`).
- **Tier 2 and Tier 3 oracle selection.** On those tiers the clean reference is not a permitted
  input, so we cannot choose what to ship by the answer, and should not even target spend by it. The
  input-only rule stays.

## Two things worth more than any dollar

1. **Accept the `Nvidia-V2D-Challenge` GitHub invite.** Every one of the 22,989 rows in each entry
   names a commit URL in a private repo. If the organisers cannot read it, the provenance fails.
2. **Confirm the registration form.**

Neither costs anything and both are prerequisites for the entries we have already uploaded counting
at all.

## One trade to be aware of before spending

There is **+0.00945 of AUC** sitting in rollouts we already hold that could be adopted at zero rank
cost today (e14's transfer rollout, e21's fine-tune). It is a third of what a tier1-auc rank point
needs — but taking it costs SP-SR two clips, which destroys the e16/e05 flip worth two points.
**Do not bank it unless the SP-SR play has already failed.** The two are alternatives, not additions.
