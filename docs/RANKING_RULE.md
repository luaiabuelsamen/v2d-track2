# How Track 2 is actually ranked

*Written 2026-11-01, after finding that the rule this project optimised for three weeks was
not the rule.* This file is the authoritative statement; where any other doc disagrees, it is
wrong and says so at the top.

## The source

The only executable statement of the rule is the official aggregator,
`https://nvidia-isaac.github.io/video_to_data/v2d_challenge/leaderboard.js`
(fetched 2026-11-01, 22,879 B). Four things in it settle what we had been guessing.

| line | text | what we had assumed |
|---|---|---|
| 74-75 | "Standard competition ranking: teams with the same score share a rank, and the next team skips past them, so four teams tied for first are all 1 and the fifth is 5." | that Kaggle's submission-time order applied, so **a tie ranked behind** |
| 251 | "Track 2 is scored separately at" each of its three input tiers | that the **twelve boards summed** into one standing |
| 82 | `row.is_baseline !== true` before ranking | that CHORD Baseline occupied a rank above us |
| 66-71 | "A team without a score on every metric is not ranked. It is listed below every complete team and greyed out like a baseline." | that an absent board had to be charged a last place, which we reported two ways because we could not tell |

There is **no rank sum anywhere in that file** — `grep -i "rankSum\|rank_sum\|lowest sum\|total"` returns nothing. The FAQ's "sum your ranking in each metric and compute final rankings as the lowest sum of metric rankings" is real, but its worked example sums **four** metrics, which is one tier. So the model is: three independent standings, each the sum of four standard-competition metric ranks.

## What that changes

**A metric gain pays nothing until it crosses a *distinct* score.** This is the expensive part. Under losing ties every near-miss looked like a rank for sale; under shared ranks most near-misses are worth zero. Measured on `tier1-sp-sr` the same afternoon the correction landed:

| our SP-SR | clips | rank |
|---|---|---|
| 0.70000 | 21/30 | 6 |
| 0.73333 | 22/30 | 6 |
| 0.76667 | 23/30 | 6 |
| 0.80000 | 24/30 | 6 |
| **0.83333** | **25/30** | **4** |

`docs/CHEAPEST_RANKS.md` had priced one clip at two rank points for ~$1.30. It is worth **zero**; the rung is four clips away, not one. The noise-floor measurement behind that plan was sound — it was applied to a ladder that did not exist.

## Where we actually stand

Recomputed by `tools/board/standings.py` from all twelve boards pulled 2026-11-01, written to `results/standings_tiers_2026_11_01.json`:

| tier | our rank sum | place | leader | ranked teams |
|---|---|---|---|---|
| Tier 1 | 37 | **11th** | MBZUAI-RCL Group, 4 | 23 |
| Tier 2 | 21 | **6th** (tied, nima arki) | MBZUAI-RCL Group, 7 | 18 |
| Tier 3 | 19 | **6th** | Izus_systems, 6 | 12 |

Not "8th of 27 at 96", which was an artifact of our own summing. Two Tier 1 entrants are unranked there for incompleteness and are not teams we are behind.

The shape of the problem is clearer per tier than it ever was in aggregate: **Tier 1 is our worst tier and the one the whole queued plan pointed at.** MBZUAI sits at 4 of a possible 4 — they are first on all four Tier 1 metrics — so Tier 1 is not winnable in the time left. Tier 2 at 6th and Tier 3 at 6th of 12 are where a rank is actually reachable.

## Consequences for pricing, going forward

1. **Price a purchase against the next distinct score on the board, never against the gap to the team above.** Pull the board, list its distinct values, and find the rung. `tools/board/standings.py` does this; ad-hoc arithmetic got it wrong twice on 2026-11-01 alone, once by using the wrong pool (two incomplete teams inflated our MPPE rank from 11 to 12) and once by charging ties.
2. **A tie costs nothing and breaking one downward gains nothing.** We are tied with nima arki at 21 in Tier 2; moving to 20 does not change our place.
3. **Incomplete entries are free of us and we of them.** Entering a board we would come last on cannot hurt our other ranks, and not entering one costs nothing beyond that tier.

## The thing that was inferred is now confirmed

**Settled by the organisers, 2026-11-02.** Shalin Jain (NVIDIA), replying to our question 4:

> All tied teams receive the same rank, and the next rank skips appropriately, there will not be
> consideration of submission time.

So the shared-tie rule is not merely what `leaderboard.js` does — it is what the final ranking
does, stated in writing. Everything in this file now rests on quoted text, and the step-function
consequence is load-bearing rather than cautious: **a metric gain pays nothing until it crosses a
distinct score.** The gate purchases that the submission-time reading would have revived are dead
for good.

<details><summary>What this previously said, kept because four numbers above hung on it</summary>


That the organisers will apply the FAQ's rank sum to the *same* per-metric rank the aggregator computes. Nothing in prose says so — the shared-tie rule is a code comment, not published text. **An email to v2d_challenge@nvidia.com asking whether an equal score is a shared rank for the final sum is worth sending**, because if ties are in fact ordered by submission time the old ladder returns and the gate purchases are live again. Four of the numbers above hang on it. Everything else here is quoted, not inferred.

</details>
