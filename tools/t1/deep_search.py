"""Hill-climb the whole Tier 1 entry against the rank sum, not three swaps at a time.

`rank_swap.py` answers "is there a small recombination that gains a rank". It cannot see a
combination that spends a gate on four clips to buy a rank elsewhere and restores it on a fifth.
The space is about 4^30, so this is multi-start hill climbing with a plateau term: when no move
lowers the rank sum, prefer the move that shortens the distance to the next rank on some board,
which lets the search cross flats instead of stopping on them.

    python tools/t1/deep_search.py --entry-perclip PERCLIP.json --leaderboards LB_DIR \
        --alternatives NAME=DIR [NAME=DIR ...] [--restarts 40]

Arguments are as `rank_swap.py` documents them.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

_PEER = Path(__file__).with_name("rank_swap.py")
_spec = importlib.util.spec_from_file_location("rank_swap", _PEER)
swap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(swap)

# How far from the next rank still counts as "close" on each board, for the plateau gradient only:
# one rank step's worth of AUC, three clips of a success rate, a centimetre of MPPE.
PLATEAU_SCALE = {"add_auc": 0.05, "spider_sr": 3.0, "maniptrans_sr": 3.0, "mppe_cm": 1.0}


def next_target(boards: dict[str, list[float]], metric: str, value: float) -> float | None:
    """The score that would gain one rank on this board, or None if we already lead it."""
    scores = [swap.quantise(metric, s) for s in boards[metric]]
    here = swap.quantise(metric, value)
    if metric in swap.LOWER_IS_BETTER:
        ahead = [s for s in scores if s <= here]
        return max(ahead) if ahead else None
    ahead = [s for s in scores if s >= here]
    return min(ahead) if ahead else None


def progress(boards: dict[str, list[float]], agg: dict[str, float]) -> float:
    """Fraction of the way to the next rank, summed over the boards: the plateau gradient."""
    total = 0.0
    for metric in swap.METRICS:
        target = next_target(boards, metric, agg[metric])
        if target is None:
            total += 1.0
            continue
        span = abs(target - swap.quantise(metric, agg[metric]))
        total += max(0.0, 1.0 - span / PLATEAU_SCALE[metric])
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--entry-perclip", type=Path, required=True)
    parser.add_argument("--leaderboards", type=Path, required=True)
    parser.add_argument("--alternatives", nargs="+", required=True, metavar="NAME=DIR")
    parser.add_argument("--team", default=swap.TEAM)
    parser.add_argument("--restarts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    entry = swap.read_entry(args.entry_perclip)
    boards = {m: swap.read_board(args.leaderboards, m, args.team) for m in swap.METRICS}

    options: dict[str, list[tuple[str, dict[str, float]]]] = {
        clip: [("shipped", row)] for clip, row in entry.items()
    }
    for clip, arm, row in swap.read_alternatives(args.alternatives, entry):
        options[clip].append((arm, row))
    clips = list(entry)

    def aggregate(choice: dict[str, int]) -> dict[str, float]:
        rows = [options[c][choice[c]][1] for c in clips]
        return {m: sum(r[m] for r in rows) / len(rows) for m in swap.METRICS}

    def ranks(agg: dict[str, float]) -> dict[str, int]:
        return {m: swap.rank(boards[m], m, agg[m]) for m in swap.METRICS}

    def objective(choice: dict[str, int]) -> tuple[int, float]:
        agg = aggregate(choice)
        return sum(ranks(agg).values()), -progress(boards, agg)

    rng = random.Random(args.seed)
    best: tuple[tuple[int, float], dict[str, int]] | None = None
    for restart in range(args.restarts):
        choice = (
            dict.fromkeys(clips, 0)
            if restart == 0
            else {c: rng.randrange(len(options[c])) for c in clips}
        )
        current = objective(choice)
        improved = True
        while improved:
            improved = False
            for clip in clips:
                for index in range(len(options[clip])):
                    if index == choice[clip]:
                        continue
                    previous, choice[clip] = choice[clip], index
                    trial = objective(choice)
                    if trial < current:
                        current, improved = trial, True
                    else:
                        choice[clip] = previous
        if best is None or current < best[0]:
            best = (current, dict(choice))

    assert best is not None
    _, choice = best
    shipped = aggregate(dict.fromkeys(clips, 0))
    found = aggregate(choice)
    moves = [(c, options[c][choice[c]][0]) for c in clips if choice[c] != 0]
    print(
        "shipped",
        {m: round(shipped[m], 5) for m in swap.METRICS},
        ranks(shipped),
        "sum",
        sum(ranks(shipped).values()),
    )
    print(
        "best   ",
        {m: round(found[m], 5) for m in swap.METRICS},
        ranks(found),
        "sum",
        sum(ranks(found).values()),
    )
    print(f"moves: {len(moves)}  " + " ".join(f"{c}:{a}" for c, a in moves))

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "restarts": args.restarts,
                    "shipped": {
                        "metrics": shipped,
                        "ranks": ranks(shipped),
                        "rank_sum": sum(ranks(shipped).values()),
                    },
                    "best": {
                        "metrics": found,
                        "ranks": ranks(found),
                        "rank_sum": sum(ranks(found).values()),
                    },
                    "moves": [{"clip": c, "arm": a} for c, a in moves],
                },
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
