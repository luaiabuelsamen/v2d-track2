"""Rank-sum standings for every team in Track 2, computed from the twelve boards.

The challenge FAQ fixes the ranking rule: "We will sum your ranking in each metric and
compute final rankings as the lowest sum of metric rankings."  Three tiers times four
metrics is twelve boards, so a team's score is a sum of twelve integers and the only thing
that matters about a metric gain is whether it crosses another team.

Two details decide whether the sum is right or flattering:

* Ties do not promote.  Kaggle orders equal scores by submission time, and every team we
  tie with uploaded before us, so a tie ranks *behind*.  Rank is therefore
  1 + (teams strictly ahead) + (teams tied that uploaded earlier).
* Kaggle truncates to five decimals.  A success rate over 30 clips can only be k/30, so
  16/30 publishes as 0.53333 while an entry computes 0.5333333; compared raw, a tie reads
  as a rank gained.  Success-rate boards are snapped back to clip counts before comparing.

Absence is reported both ways, because the FAQ does not say: `strict` counts only the
boards a team entered (which flatters a team that entered one board), `tied_last` gives an
absent team the last rank on that board plus one.  Our own standing is the same under both.

Usage:  python tools/board/standings.py <lb_dir> [--json out.json]
where <lb_dir> holds tier{1,2,3}-{auc,sp-sr,mp-sr,mppe}.csv as
`kaggle competitions leaderboard --show --csv --page-size 200` writes them.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

TIERS = ("1", "2", "3")
METRICS = ("auc", "sp-sr", "mp-sr", "mppe")
LOWER_IS_BETTER = {"mppe"}
# A success rate over 30 clips is k/30; snap both sides to k so Kaggle's truncation of
# 0.5333333 to 0.53333 cannot turn a genuine tie into a rank we do not hold.
QUANTISED = {"sp-sr": 30, "mp-sr": 30}
OURS = "Luai Abuelsamen"


def quantise(metric: str, value: float) -> float:
    clips = QUANTISED.get(metric)
    return round(value * clips) if clips else value


def read_board(path: Path, canonical: dict[str, str]) -> dict[str, tuple[float, str]]:
    """Best row per team: {team: (score, submission time)}.

    `canonical` is shared across boards and folds case-variant names onto one team:
    MBZUAI uploaded as both "MBZUAI-RCL Group" and "MBZUAI-RCL GROUP", and counting those
    as two teams would both lose them ranks and hand us a place we have not earned.
    """
    best: dict[str, tuple[float, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            team, score = row.get("teamName"), row.get("score")
            if not team or not score:
                continue
            try:
                value = float(score)
            except ValueError:
                continue
            key = canonical.setdefault(team.casefold(), team)
            when = row.get("submissionDate", "")
            prior = best.get(key)
            if prior is None or when < prior[1]:
                best[key] = (value, when)
    return best


def rank_of(board: dict[str, tuple[float, str]], metric: str, team: str) -> int:
    """1 + teams strictly ahead + teams tied that uploaded earlier."""
    mine, when = board[team]
    mine = quantise(metric, mine)
    better = 0
    for other, (score, other_when) in board.items():
        if other == team:
            continue
        score = quantise(metric, score)
        ahead = score < mine if metric in LOWER_IS_BETTER else score > mine
        if ahead or (score == mine and other_when < when):
            better += 1
    return 1 + better


def standings(lb_dir: Path) -> dict:
    boards = {}
    canonical: dict[str, str] = {}
    for tier in TIERS:
        for metric in METRICS:
            path = lb_dir / f"tier{tier}-{metric}.csv"
            if path.exists():
                boards[(tier, metric)] = read_board(path, canonical)

    teams = {team for board in boards.values() for team in board}
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    for (tier, metric), board in boards.items():
        for team in board:
            ranks[team][f"tier{tier}-{metric}"] = rank_of(board, metric, team)

    rows = []
    for team in teams:
        held = ranks[team]
        absent = [f"tier{t}-{m}" for t in TIERS for m in METRICS if f"tier{t}-{m}" not in held]
        penalty = sum(len(boards[(b[4], b[6:])]) + 1 for b in absent if (b[4], b[6:]) in boards)
        per_tier = {
            f"tier{t}": sum(v for k, v in held.items() if k.startswith(f"tier{t}-")) for t in TIERS
        }
        rows.append(
            {
                "team": team,
                "strict": sum(held.values()),
                "tied_last": sum(held.values()) + penalty,
                "entered": len(held),
                "per_tier": per_tier,
                "ranks": dict(sorted(held.items())),
            }
        )

    rows.sort(key=lambda r: (r["tied_last"], r["strict"]))
    for place, row in enumerate(rows, 1):
        row["place_tied_last"] = place
    by_strict = sorted(rows, key=lambda r: (r["strict"], -r["entered"]))
    for place, row in enumerate(by_strict, 1):
        row["place_strict"] = place
    return {"boards": len(boards), "teams": len(teams), "standings": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("lb_dir", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    out = standings(args.lb_dir)
    print(f"{out['teams']} teams over {out['boards']} boards\n")
    print(f"{'#':>3}  {'team':28} {'sum':>4} {'strict':>6} {'in':>3}  T1  T2  T3")
    for row in out["standings"]:
        mark = " <-- us" if row["team"] == OURS else ""
        t = row["per_tier"]
        print(
            f"{row['place_tied_last']:>3}  {row['team'][:28]:28} {row['tied_last']:>4} "
            f"{row['strict']:>6} {row['entered']:>3}  "
            f"{t['tier1']:>3} {t['tier2']:>3} {t['tier3']:>3}{mark}"
        )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
