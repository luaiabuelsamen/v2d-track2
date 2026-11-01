# Two rank points for about two dollars

Written 2026-11-01, free compute only, no launch. It revises the $41 plan in
`docs/WHAT_IT_TAKES.md` downward by a factor of twenty, and the revision comes from a measurement
that was already in our own files.

## The measurement nobody had mined

Every arm this project ran was judged on a single number per clip, and a single number from a
single training run has an error bar. The 30-clip run saved seven checkpoints per clip, so that
error bar is measurable: the spread of kit AUC over each clip's **last five checkpoints**, with the
clip, the recipe and the reference all held fixed, is

| | |
|---|---|
| std over the last five checkpoints | **0.0809** mean |
| range over the last five | **0.2011** mean, 0.5614 max |

`tools/spend/attempt_value.py`, `results/attempt_value.json`.

That is attempt noise, and it is **larger than most of the differences this project has treated as
results.** Re-reading the donor transfer against it: of nine attempts, **six moved the score by
less than the noise floor** — e01, e04, e07, e13, e19, e25 — and one of those six, e19, was
*adopted*. So "3 of 9 clips are rescuable" was never a property of the clips. It is one draw from a
distribution wide enough to have produced 1 of 9 or 5 of 9.

**This corrects a plan I recommended yesterday.** The $41 breadth plan assumed six attempts per
clip would convert at the measured 1-in-3. But a bootstrap over the same checkpoints says that
repeating *the same arm* is worth almost nothing:

| more draws of the same arm | gain |
|---|---|
| best-of-14 against best-of-7, all 30 clips | **+0.0112** AUC (upper bound) |
| the same over the twelve broken clips | **+0.0076** |

Over twelve clips that is +0.09 of summed AUC, about +0.003 on the entry mean — a tenth of a
tier1-auc rank point for $41. **Repetition is not what paid.** What paid on e05 was a *different*
actor, which moves the distribution rather than resampling it. So attempts are only worth buying
when each attempt is a different intervention, and the $7 donor test was right for exactly that
reason: four *different* donors, not four retries.

## What the noise floor makes cheap

A noise floor that large is bad news for attribution and very good news for **gates**, because a
gate is a step function. Two Tier 1 clips sit a hair from the SP-SR gate
(`results/t1_gate_margins_table.json`), and SP-SR is the board where one clip is worth two ranks:

| clip | position | orientation | misses SP-SR by | the gate |
|---|---|---|---|---|
| **e05** | **10.319 cm** | 0.308 rad | **0.319 cm of position** | ≤ 10 cm **and** ≤ 0.5 rad |
| **e16** | 1.864 cm | **0.5219 rad** | **0.0219 rad = 1.26°** | ≤ 10 cm **and** ≤ 0.5 rad |

Both margins are **far inside the run-to-run spread**. e05's own position error across the two
Tier 1 rollouts we hold is 14.63 cm and 10.37 cm — a 4.3 cm range — and it needs to find 0.32 cm.

And the board arithmetic is unusually kind here:

- tier1-sp-sr is 21/30, **rank 13 of 24**, with two teams sitting on exactly 21 ahead of us on
  submission time. **22 clips takes us to rank 11: two rank points for one clip.**
- 23 clips buys nothing further (rank 10 needs 25), so **only one of e05 and e16 has to land.**
- If e05 improves by 1.90 cm rather than 0.32, the entry mean MPPE crosses 11.997 cm as well and
  tier1-mppe goes 11th to 10th. **Three rank points from one clip.**

## Why the downside is zero, and why that is only true here

`config/tracks.json` names the released Tier 1 multi-view trajectory as the clean trajectory used
as the configured scoring reference, so **on Tier 1 the permitted input is the scoring reference**.
Truth-based per-clip selection is therefore legal there and only there: we keep a new rollout only
if it scores better, and a wrecked attempt costs its own compute and nothing else. On Tier 2 and
Tier 3 the input-only z-drop rule still binds and none of this applies.

That asymmetry is what makes a coin flip worth buying.

## The queue, reordered and repriced

| # | what | cost | what it buys |
|---|---|---|---|
| **1** | **e05 and e16, one fresh attempt each, Tier 1** | **~$1.30** | **2 rank points, 3 if e05 moves 1.9 cm. Zero downside.** |
| 2 | the owed eval-image control (e00, e03 on Tier 2 input) | $0.70 | rules out the eval image behind the clean-input result the roadmap leans on |
| 3 | donor-variance test, 3 clips × 4 **different** donors | $7 | whether a different intervention converts where a repeat does not |
| 4 | the ten remaining broken Tier 1 clips, different arms | ~$15 | the rest of the Tier 1 deficit, at the rate item 3 measures |

**Entry price is about $2, not $50.** Items 1 and 2 together are $2 and fit on one box with its
$0.55 bring-up — call it $2.55 for the box, two gate attempts and the owed control.

Arms for item 1, one variable each, chosen to fit the margin rather than the clip:

- **e05** wants 0.32 cm of position. A donor transfer is the arm with a measured track record of
  moving position on clips that lose the object, and e05 is one; a *second* donor is one draw.
- **e16** wants 1.26° of orientation and already holds position at 1.86 cm, so a donor is the wrong
  tool — it wrecked e13's tight tracking exactly this way (4.98 → 30.21 cm). The gentler arm is a
  warm-start fine-tune of e16's own actor (`brev/patch_warm_start.py`, implemented, never run on a
  box), which perturbs rather than replaces.

## Honest odds

Neither flip is assured. e05 needs 0.32 cm from a draw whose spread is about 4 cm, so call it one
chance in three on a single attempt; e16's 1.26° is a similar order. One attempt each gives roughly
an even chance that at least one lands, and **only one has to land.** At $1.30 against two rank
points, the bet is worth taking whether or not it wins, which is not something that can be said of
the $41 plan it replaces.

The thing that is *not* purchasable has not changed: our eighteen working Tier 1 clips average
0.7506 AUC and the leaders' typical clip beats our second best. First place is a better tracker.
This is about ranks 11 and 13, where step functions live.
