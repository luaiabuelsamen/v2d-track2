"""Track 2 standings, computed the way the organisers compute them: per tier, ties shared.

This file previously summed twelve boards into one number and ranked ties behind, and both
halves were wrong.  The correction comes from the official aggregator, which is the only
executable statement of the rule anywhere:
`https://nvidia-isaac.github.io/video_to_data/v2d_challenge/leaderboard.js`

* **Three tiers are three standings.**  Line 251: "Track 2 is scored separately at" each of
  its three input tiers, and the track list carries `track_2_tier1`, `track_2_tier2` and
  `track_2_tier3` as separate tables.  Nothing in that file sums anything -- there is no
  `rank_sum` in it at all -- and the FAQ's "sum your ranking in each metric" worked example
  sums *four* metrics, i.e. one tier.  A twelve-board total is our invention, and it hid
  which tier a purchase would actually move.
* **Ties share a rank.**  Lines 74-75: "Standard competition ranking: teams with the same
  score share a rank, and the next team skips past them, so four teams tied for first are
  all 1 and the fifth is 5."  We had assumed Kaggle's submission-time ordering applied, so
  a tie ranked behind and every near-miss looked like a rank for sale.  Under the real rule
  a metric gain pays nothing until it crosses a *distinct* score: on tier1-sp-sr, 21, 22 and
  23 of 30 clips are all one rank and the next rung is 25.
* **Baselines do not occupy ranks** (line 82, `row.is_baseline !== true`), so CHORD Baseline
  sitting above us costs us nothing.
* **An incomplete team is not ranked at all.**  Lines 66-71: "A team without a score on
  every metric is not ranked.  It is listed below every complete team and greyed out like a
  baseline."  So a team that entered one board of a tier is not a team we are behind, and
  the old `strict`/`tied_last` columns -- which guessed at this both ways -- are gone.

Kaggle's five-decimal truncation is still handled: a success rate over 30 clips can only be
k/30, so 16/30 publishes as 0.53333 while an entry computes 0.5333333, and compared raw that
reads as a distinct score rather than the tie it is.  Success-rate boards are snapped back to
clip counts before comparing.

Usage:  python tools/board/standings.py <lb_dir> [--json out.json]
where <lb_dir> holds tier{1,2,3}-{auc,sp-sr,mp-sr,mppe}.csv as
`kaggle competitions leaderboard --show --csv --page-size 200` writes them.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

TIERS = ("1", "2", "3")
METRICS = ("auc", "sp-sr", "mp-sr", "mppe")
LOWER_IS_BETTER = {"mppe"}
# A success rate over 30 clips is k/30; snap both sides to k so Kaggle's truncation of
# 0.5333333 to 0.53333 cannot read as a distinct score when it is really a tie.
QUANTISED = {"sp-sr": 30, "mp-sr": 30}
# The aggregator drops `is_baseline` rows before ranking.  The board CSVs carry no such
# column, so the organisers' own entries are matched by name.
BASELINE = re.compile(r"\bbaseline\b", re.IGNORECASE)
OURS = "Luai Abuelsamen"


def quantise(metric: str, value: float) -> float:
    clips = QUANTISED.get(metric)
    return round(value * clips) if clips else value


def read_board(path: Path, canonical: dict[str, str], metric: str = "auc") -> dict[str, float]:
    """Best score per team, baselines dropped.

    `metric` only decides which direction "best" is, for the rare board that lists a team
    twice; a leaderboard CSV carries one row per team, so it almost never matters.

    `canonical` is shared across boards and folds case-variant names onto one team: MBZUAI
    uploaded as both "MBZUAI-RCL Group" and "MBZUAI-RCL GROUP", and counting those as two
    teams would both lose them ranks and hand us a place we have not earned.
    """
    best: dict[str, float] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            team, score = row.get("teamName"), row.get("score")
            if not team or not score or BASELINE.search(team):
                continue
            try:
                value = float(score)
            except ValueError:
                continue
            key = canonical.setdefault(team.casefold(), team)
            prior = best.get(key)
            if prior is None or better(metric, value, prior):
                best[key] = value
    return best


def better(metric: str, value: float, than: float) -> bool:
    return value < than if metric in LOWER_IS_BETTER else value > than


def competition_rank(board: dict[str, float], metric: str, team: str) -> int:
    """Standard competition rank: 1 + the number of *distinct* better scores.

    Teams on the same score share the rank and the next team skips past them, so four teams
    tied for first are all 1 and the fifth is 5 -- `canonicalRanks` in leaderboard.js.
    """
    mine = quantise(metric, board[team])
    ahead = {
        quantise(metric, score)
        for other, score in board.items()
        if other != team and better(metric, quantise(metric, score), mine)
    }
    return 1 + len(ahead)


def tier_standings(boards: dict[str, dict[str, float]]) -> dict:
    """One tier's table: complete teams ranked by the sum of their four metric ranks."""
    entered = {team for board in boards.values() for team in board}
    complete = sorted(t for t in entered if all(t in board for board in boards.values()))
    unranked = sorted(entered - set(complete))

    rows = []
    for team in complete:
        # Only complete teams are ranked, so they are the only ones in the pool.
        ranks = {
            metric: competition_rank({t: board[t] for t in complete}, metric, team)
            for metric, board in boards.items()
        }
        rows.append(
            {
                "team": team,
                "rank_sum": sum(ranks.values()),
                "ranks": dict(sorted(ranks.items())),
                "scores": {m: boards[m][team] for m in sorted(boards)},
            }
        )
    rows.sort(key=lambda r: (r["rank_sum"], r["team"].casefold()))
    # The standings table is itself a ranking, so ties share a place here too.
    previous_sum, previous_place = None, 0
    for index, row in enumerate(rows, 1):
        if row["rank_sum"] != previous_sum:
            previous_place = index
        previous_sum = row["rank_sum"]
        row["place"] = previous_place
    return {
        "boards": sorted(boards),
        "ranked_teams": len(rows),
        "unranked_incomplete": unranked,
        "standings": rows,
    }


def standings(lb_dir: Path) -> dict:
    canonical: dict[str, str] = {}
    tiers = {}
    for tier in TIERS:
        boards = {}
        for metric in METRICS:
            path = lb_dir / f"tier{tier}-{metric}.csv"
            if path.exists():
                boards[metric] = read_board(path, canonical, metric)
        if len(boards) == len(METRICS):
            tiers[f"tier{tier}"] = tier_standings(boards)
    return {
        "rule": "Per tier, sum of four standard-competition metric ranks. Ties share a rank; "
        "baselines are excluded; a team missing any of a tier's four metrics is unranked "
        "(leaderboard.js lines 66-82).",
        "tiers": tiers,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lb_dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out = standings(args.lb_dir)
    for tier, table in out["tiers"].items():
        incomplete = len(table["unranked_incomplete"])
        print(
            f"\n=== {tier}: {table['ranked_teams']} ranked teams, "
            f"{incomplete} unranked (incomplete)"
        )
        print(f"{'#':>3}  {'team':30} {'sum':>4}  " + "  ".join(f"{m:>6}" for m in METRICS))
        for row in table["standings"]:
            mark = " <-- us" if row["team"] == OURS else ""
            ranks = "  ".join(f"{row['ranks'][m]:>6}" for m in METRICS)
            print(f"{row['place']:>3}  {row['team'][:30]:30} {row['rank_sum']:>4}  {ranks}{mark}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
