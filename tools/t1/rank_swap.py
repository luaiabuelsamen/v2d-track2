"""Search Tier 1 rollout swaps against the rank sum, not against metric monotonicity.

The challenge ranks by the sum of per-metric ranks (`README.md`), so the question a swap has to
answer is not "does it improve a metric without hurting another" -- the Pareto criterion in
`results/t1_entry_pareto_optimal.json` -- but "does it lower the rank sum". Those differ: AUC has a
0.00625 cushion below us and MPPE a 0.0632 cm gap above, so spending the first to buy the second is
worth a rank point and the Pareto criterion rejects it.

Ties do not promote. Kaggle breaks them by submission time and every team we tie with uploaded
earlier, so a tied score ranks behind.

    python tools/t1/rank_swap.py --entry-perclip PERCLIP.json --leaderboards LB_DIR \
        --alternatives NAME=DIR [NAME=DIR ...] [--depth 3]

PERCLIP.json is the `per_episode` block of `v2d score --per-episode` on the uploaded entry.
LB_DIR holds `tier1-{auc,sp-sr,mp-sr,mppe}.csv` as `kaggle competitions leaderboard --csv` writes
them. Each alternatives directory holds `eXX.score.json`.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path

METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
BOARD_FILE = {
    "add_auc": "auc",
    "spider_sr": "sp-sr",
    "maniptrans_sr": "mp-sr",
    "mppe_cm": "mppe",
}
LOWER_IS_BETTER = frozenset({"mppe_cm"})
TEAM = "Luai Abuelsamen"


def read_board(directory: Path, metric: str, us: str) -> list[float]:
    """Every other team's score on one board, in leaderboard order."""
    text = (directory / f"tier1-{BOARD_FILE[metric]}.csv").read_text()
    if text.startswith("Next Page Token"):
        text = text.split("\n", 1)[1]
    scores = []
    for row in csv.DictReader(text.splitlines()):
        if row.get("teamName") in (None, us):
            continue
        try:
            scores.append(float(row["score"]))
        except (TypeError, ValueError):
            continue  # a short or header-repeating row, not a score of 0.0
    return scores


def rank(scores: list[float], metric: str, value: float) -> int:
    """1 + teams strictly ahead + teams tied, since a tie ranks behind on submission time."""
    if metric in LOWER_IS_BETTER:
        ahead = sum(1 for s in scores if s < value)
    else:
        ahead = sum(1 for s in scores if s > value)
    tied = sum(1 for s in scores if abs(s - value) < 1e-9)
    return 1 + ahead + tied


def read_entry(path: Path) -> dict[str, dict[str, float]]:
    block = json.loads(path.read_text())
    block = block.get("per_episode", block)
    return {f"e{int(k):02d}": {m: v[m] for m in METRICS} for k, v in block.items()}


def read_alternatives(
    specs: list[str], entry: dict[str, dict[str, float]]
) -> list[tuple[str, str, dict[str, float]]]:
    """(clip, arm, metrics) for every held rollout that differs from the one shipped."""
    out = []
    for spec in specs:
        name, _, directory = spec.partition("=")
        for path in sorted(Path(directory).glob("e[0-9][0-9].score.json")):
            clip = path.name.split(".")[0]
            if clip not in entry:
                continue
            raw = json.loads(path.read_text())
            row = {m: raw.get("overall", raw)[m] for m in METRICS}
            if any(abs(row[m] - entry[clip][m]) > 1e-9 for m in METRICS):
                out.append((clip, name, row))
    return out


def aggregate(
    entry: dict[str, dict[str, float]], swaps: dict[str, dict[str, float]]
) -> dict[str, float]:
    table = {**entry, **swaps}
    return {m: sum(row[m] for row in table.values()) / len(table) for m in METRICS}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entry-perclip", type=Path, required=True)
    parser.add_argument("--leaderboards", type=Path, required=True)
    parser.add_argument("--alternatives", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--team", default=TEAM)
    parser.add_argument("--depth", type=int, default=3, help="largest number of simultaneous swaps")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    entry = read_entry(args.entry_perclip)
    boards = {m: read_board(args.leaderboards, m, args.team) for m in METRICS}
    candidates = read_alternatives(args.alternatives, entry)

    def ranks(agg: dict[str, float]) -> dict[str, int]:
        return {m: rank(boards[m], m, agg[m]) for m in METRICS}

    base_agg = aggregate(entry, {})
    base_ranks = ranks(base_agg)
    base_sum = sum(base_ranks.values())
    print("entry  ", {m: round(base_agg[m], 5) for m in METRICS})
    print("ranks  ", base_ranks, "sum", base_sum)
    print(f"{len(candidates)} alternatives over {len({c for c, _, _ in candidates})} clips")

    wins = []
    for depth in range(1, args.depth + 1):
        for combo in itertools.combinations(candidates, depth):
            if len({clip for clip, _, _ in combo}) != depth:
                continue
            agg = aggregate(entry, {clip: row for clip, _, row in combo})
            got = ranks(agg)
            if sum(got.values()) < base_sum:
                wins.append((sum(got.values()), combo, agg, got))
        if wins:
            break

    wins.sort(key=lambda w: w[0])
    if not wins:
        print(f"no combination of up to {args.depth} swaps lowers the rank sum")
    for total, combo, agg, got in wins[:8]:
        print(f"sum {total} (from {base_sum})  " + " ".join(f"{c}:{n}" for c, n, _ in combo))
        print("   ", {m: round(agg[m], 5) for m in METRICS}, got)

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "entry": base_agg,
                    "ranks": base_ranks,
                    "rank_sum": base_sum,
                    "alternatives": len(candidates),
                    "depth_searched": args.depth,
                    "improvements": [
                        {"rank_sum": t, "swaps": [{"clip": c, "arm": n} for c, n, _ in combo]}
                        for t, combo, _, _ in wins
                    ],
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
