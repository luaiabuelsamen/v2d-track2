# What it takes to win

*2026-10-30 18:30 UTC, 124 hours to the freeze. Every number here is from the full leaderboards
pulled this hour (24 teams on Tier 1, 18 on Tier 2, 11 on Tier 3) or measured on boxes we ran.*

## Where we actually are: 7th of 24

The FAQ is explicit: *"We will sum your ranking in each metric and compute final rankings as the
lowest sum of metric rankings."* Computing that for the whole field — something this project had
never done, having only ever tracked its own sum:

| # | team | rank sum | Tier 1 | Tier 2 | Tier 3 | boards entered |
|---|---|---|---|---|---|---|
| 1 | Izus_systems | **39** | 12 | 17 | 10 | 12/12 |
| 2 | nima arki | 53 | 20 | 23 | 10 | 12/12 |
| 3 | maniphys.ai | 56 | 33 | 13 | 10 | 12/12 |
| 4 | Xueyi Wang (FF) | 57 | 17 | 11 | 29 | 12/12 |
| 5 | MBZUAI-RCL Group | 58 | **4** | **6** | 48 | 8/12 |
| 6 | domoarigato | 72 | 27 | 28 | 17 | 12/12 |
| **7** | **us** | **86** | **44** | **19** | **23** | 12/12 |
| 8 | wte09fl | 98 | 45 | 31 | 22 | 12/12 |

Two facts jump out of that table.

**MBZUAI has the best scores in the challenge and is 5th**, because they never entered Tier 3 and
pay the absence penalty on four boards. Rank-sum scoring pays for breadth, and we are on all twelve
boards. (I checked the sensitivity: under "absence = tied last" we are still 7th and Izus still
leads. Only a rule that ignores unentered boards changes it, and that rule would put a team with
one entry in 2nd, so it is not the rule.)

**Our Tier 1 is our problem.** 44 of our 86 comes from one tier, and it is the worst Tier 1 of
anyone in the top seven.

## The ladder, in rank points from 86

| to reach | must beat | points needed |
|---|---|---|
| 6th | domoarigato 72 | **−15** |
| 5th | MBZUAI 58 | −29 |
| 4th | Xueyi 57 | −30 |
| **3rd** | maniphys 56 | **−31** |
| 2nd | nima arki 53 | −34 |
| 1st | Izus_systems 39 | **−48** |

Note the cluster: 3rd, 4th and 5th sit at 56, 57 and 58. **−31 buys three places at once**, and is
only two points harder than one place.

## Why 1st is not purchasable, stated as a measurement

Beating 39 means averaging rank 3.2 across twelve boards, including Tier 1 where the top four score
0.82–0.89 AUC. Ours is 0.548. The question is whether that is a selection problem or a quality
problem, and it is measurable:

- Our 18 Tier 1 clips that work average **0.7506** AUC. The other 12 are below 0.45.
- If every weak clip were lifted to the average of the good ones, the entry would score **0.7506**
  — **rank 6**, not 1.
- To take rank 1 on that board, every clip would have to reach **0.9567**, which is our single best
  clip in 30.

So the leaders' typical clip is better than our second-best clip. **That is the recipe and the
training scale, not the cleanup, not the reference, and not the selection** — and it is five days
of research, not a purchase. First place is out of reach from here.

## What *is* purchasable, and it is a lot

The same measurement says our Tier 1 problem is **12 specific broken clips**, not a uniform
weakness: e25, e04, e19, e13, e01, e14, e21, e07, e28, e09, e02, e17. They drag all four Tier 1
boards at once — nine of them are the SP-SR failures, four are the 41–102 cm position errors that
own most of our MPPE, and together they cost about 0.20 of entry AUC.

If 8 of those 12 were rescued to ordinary quality:

| board | now | after | points |
|---|---|---|---|
| tier1-sp-sr | 21/30, rank 12 | ~29/30, rank 2 | **−10** |
| tier1-mppe | 12.06 cm, rank 10 | ~4 cm, rank 6 | −4 |
| tier1-auc | 0.548, rank 11 | ~0.65, rank 9 | −2 |
| tier1-mp-sr | 15/30, rank 11 | ~19/30, rank 9 | −2 |

**Tier 1 alone: 44 → ~26, eighteen rank points.** That is 86 → 68, past domoarigato into 6th,
within striking distance of the 56–58 cluster once Tier 2 and Tier 3 breadth is added.

### And we have a measured method for exactly this

The donor transfer — loading another clip's trained actor and fine-tuning briefly, which works
because all 30 shipped actors share one tensor signature — rescued **3 of 9** broken clips at
**$0.37 each**: e05 0.266 → 0.651, e28 0.089 → 0.298 with the object held to frame 745 instead of
118, e19 0.116 → 0.159. One donor per clip, one attempt each.

Truth-based per-clip selection is legal on Tier 1 and only there, because `config/tracks.json` makes
the released multiview trajectory both the permitted input and the scoring reference. So **best-of-N
per clip is a legitimate entry**, and N is what money buys.

At a measured $0.57 per warm-start cell, **12 clips × 6 attempts = 72 cells = $41**. If the
one-in-three rate is per *attempt*, six attempts give each clip an 91% chance of at least one
rescue, and the plan above is funded for the price of a dinner.

### The one thing that could make that wrong, and the $7 test for it

The 1-in-3 came from nine clips with **one** donor each. It is consistent with two different worlds:

- **per-attempt**: any clip can be rescued, it just takes tries. Breadth works, $41 buys ~10 clips.
- **per-clip**: three of those nine were rescuable and six are not, whatever the donor. Then extra
  attempts on a dead clip buy nothing and the plan collapses to the three clips already found.

Everything downstream depends on which it is, and **3 clips × 4 donors = 12 cells = $7** separates
them: if a clip that failed its first donor succeeds on a later one, the world is per-attempt. Buy
that before buying anything else.

## The rank curves, so spend can be aimed

How many clips each success-rate board needs, with ties ranking behind us:

| board | now | next ranks |
|---|---|---|
| tier1-sp-sr | 21/30, rank 12 | +1 clip → **10** · +5 → 7 · +6 → 5 · +7 → 2 · +9 → **1** |
| tier1-mp-sr | 15/30, rank 11 | +2 → 10 · +7 → 9 · +9 → 6 · +11 → 4 · +12 → 2 |
| tier2-sp-sr | 20/30, rank 5 | +2 → 4 · +3 → **2** · +5 → **1** |
| tier2-mp-sr | 14/30, rank 5 | +2 → 4 · +3 → 3 · +5 → 2 |
| tier3-sp-sr | 6/30, rank 6 | +2 → 5 · +5 → 4 · +7 → **2** · +9 → **1** |
| **tier3-mp-sr** | 1/30, rank 7 | **+1 → 5 · +3 → 1** |

And the continuous boards:

| board | now | next ranks |
|---|---|---|
| tier1-auc | 0.54808, 11 | 0.58226 → 10 · 0.58258 → 9 · 0.69517 → 8 |
| tier1-mppe | 12.06020, 10 | 11.997 → 9 · 11.867 → 8 · 7.781 → 7 |
| tier2-auc | 0.46522, 5 | 0.48070 → 4 · 0.51379 → 3 · 0.52267 → 2 |
| tier2-mppe | 14.12760, 4 | 11.524 → 3 · 10.843 → 2 · 9.284 → 1 |
| tier3-auc | 0.26781, 4 | 0.28173 → 3 · 0.32937 → 2 · 0.33311 → 1 |
| tier3-mppe | 47.77180, 6 | 42.304 → 5 · 38.599 → 4 · 36.601 → 3 |

**tier3-mp-sr is the best ratio in the challenge: three clips for six rank points**, 7th to 1st,
because only 11 teams entered and the leaders pass just 3 of 30. It is also the least likely — we
pass 1, and our Tier 3 position error averages 47.77 cm against a 3 cm gate.

## Recombination is finished, now properly

Before spending, the free option was re-tested properly. Multi-start hill climbing over the whole
30-clip entry — 4^30 combinations, 40 restarts, with a plateau term so the search crosses flats
instead of stopping on them — against the rank sum rather than against metric monotonicity:

**44 is a hard floor on what we hold.** The best it finds is 12.01783 cm on MPPE, 0.0208 cm short
of one rank point. Nothing else in 67 rollouts moves a board.

## The answer, in four lines

- **1st place: no.** The leaders' typical clip beats our second-best. That is a better tracker and
  we have five days, not a quarter.
- **3rd place: about $150–300 and nearly everything landing.** Honestly, one chance in four or five.
- **6th place: about $40–60 and the Tier 1 rescue working at the measured rate.** The realistic aim.
- **Doing nothing is not standing still.** wte09fl cost us two rank points this afternoon by
  uploading while we were idle.

Buy the **$7 per-attempt test** first. Everything else is priced off its answer.
