#!/usr/bin/env python3
"""Move the point where the VOC spring reaches zero, from 60% of training to a later fraction.

The virtual-object-controller spring holds the object on the reference trajectory and is faded
out across the curriculum; `voc_scale: 0.0` is the first stage the policy has to carry the object
entirely itself, and upstream that stage begins at 209,999,872 steps, 60% of the 350M budget.

For the clips that lose a heavy object in the first quarter -- e14's round table, e28's squeegee,
e19's cane, all knocked over on first contact and never recovered -- 60% is early: the policy is
asked to take the whole load before it can place two hands on the object reliably, and the 40% of
training that follows is spent practising the motion without it. This stretches the fade so the
hand-off happens at `--zero-at` (default 0.85) instead.

The ramp (stages 1-6) is rescaled by the same factor, so the shape of the fade is unchanged and
only its timescale moves; the three zero-VOC stages keep their stock 22.5 : 5 : 12.5 proportions
inside whatever is left. Every threshold stays a multiple of the 2,048 world count, as upstream's
are. Nothing but the thresholds line changes, so this composes with the start-curriculum patch,
which edits stage bodies -- but **run patch_start_curriculum.py first**. That patch verifies the
stock thresholds line before touching anything, by design, so it refuses a tree this one has
already rewritten. In that order both land; in the other order the second stops the run, which is
the safe failure but a failure all the same.

    python patch_voc_curriculum.py /workspace/src/flash_chord [--zero-at 0.85] [--check]
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path

MARKER = "# v2d voc-curriculum"
MARKER_LINE = re.compile(rf"^{re.escape(MARKER)} zero_at=([0-9.]+)\s*$", re.MULTILINE)

RECIPE = "recon_body_flash_sac.yaml"
BUDGET = 349_999_104
WORLD_COUNT = 2048
DEFAULT_ZERO_AT = 0.85

#: Upstream's thresholds, as fractions of the budget. Stage 7 (voc 0.0) begins at index 5.
STOCK = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.825, 0.875, 1.0)
STOCK_ZERO_AT = STOCK[5]
STOCK_LINE = (
    "thresholds: [35000320, 70000640, 104998912, 139999232, 174999552, 209999872, "
    "288749568, 306249728, 349999104]\n"
)


def snap(fraction: float) -> int:
    """A step count on the world-count grid, as every upstream threshold is."""
    return round(BUDGET * fraction / WORLD_COUNT) * WORLD_COUNT


def thresholds(zero_at: float) -> list[int]:
    """The stretched schedule: same fade shape, same tail proportions, new timescale."""
    ramp = [snap(zero_at * fraction / STOCK_ZERO_AT) for fraction in STOCK[:6]]
    tail_room = 1.0 - zero_at
    stock_tail = 1.0 - STOCK_ZERO_AT
    tail = [
        snap(zero_at + tail_room * (STOCK[6] - STOCK_ZERO_AT) / stock_tail),
        snap(zero_at + tail_room * (STOCK[7] - STOCK_ZERO_AT) / stock_tail),
        BUDGET,
    ]
    return ramp + tail


def threshold_line(zero_at: float) -> str:
    return (
        f"{MARKER} zero_at={zero_at:g}\n"
        "thresholds: [" + ", ".join(str(step) for step in thresholds(zero_at)) + "]\n"
    )


def carried(path: Path) -> float | None:
    if not path.is_file():
        return None
    found = MARKER_LINE.search(path.read_text())
    return float(found.group(1)) if found else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="flash_chord source root")
    parser.add_argument(
        "--zero-at",
        type=float,
        default=DEFAULT_ZERO_AT,
        help=f"fraction of training at which voc_scale reaches 0 "
        f"(default {DEFAULT_ZERO_AT:g}, stock {STOCK_ZERO_AT:g})",
    )
    parser.add_argument("--check", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)

    if not STOCK_ZERO_AT < args.zero_at < 1.0:
        raise SystemExit(
            f"FATAL: --zero-at must be between {STOCK_ZERO_AT:g} and 1.0, got {args.zero_at!r}. "
            "Below the stock value this would shorten the fade, which is a different experiment; "
            "at 1.0 the policy never carries the object before the run ends."
        )

    path = args.root / "configs/curriculum" / RECIPE
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} does not exist")
    already = carried(path)
    if already is not None:
        if abs(already - args.zero_at) < 1e-12:
            print(f"already patched at zero_at {args.zero_at:g}")
            return 0
        raise SystemExit(
            f"FATAL: {path} already carries zero_at={already:g}, but {args.zero_at:g} was asked "
            "for. Rebuild the container rather than training an arm nobody chose."
        )

    text = path.read_text()
    if text.count(STOCK_LINE) != 1:
        raise SystemExit(
            f"FATAL: {path} does not carry the thresholds this patch was derived against.\n"
            f"  expected: {STOCK_LINE.strip()}\n"
            "  The stage-to-step mapping would be wrong; re-derive against the installed config."
        )
    steps = thresholds(args.zero_at)
    if any(later <= earlier for earlier, later in itertools.pairwise(steps)):
        raise SystemExit(f"FATAL: the computed schedule is not increasing: {steps}")

    patched = text.replace(STOCK_LINE, threshold_line(args.zero_at), 1)
    if not args.check:
        path.write_text(patched)
    print(f"{'would patch' if args.check else 'patched'}: {path}")
    print(f"voc_scale reaches 0 at {args.zero_at:g} of the budget "
          f"({steps[5]:,} steps; stock {STOCK_ZERO_AT:g}, 209,999,872)")
    print("thresholds: " + ", ".join(f"{step:,}" for step in steps))
    return 0


if __name__ == "__main__":
    sys.exit(main())
