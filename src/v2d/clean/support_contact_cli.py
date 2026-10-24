"""Command-line front end for `v2d.clean.support_contact`.

The cleaner itself exposes `run()` and `save()` so it can be called as a library; this is the
argparse layer, kept separate so the two concerns do not share a file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import FPS
from ..supports import ABOVE_TOLERANCE_CM, BELOW_TOLERANCE_CM
from .support_contact import Options, run, save


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="v2d clean-support",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=FPS)
    parser.add_argument(
        "--rest-speed", type=float, default=0.02, help="m/s below which a frame is at rest"
    )
    parser.add_argument("--min-rest-frames", type=int, default=5)
    parser.add_argument(
        "--smooth-frames",
        type=int,
        default=9,
        help="moving-average window for the speed estimate; 1 disables it",
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=45.0,
        help="reject a per-frame correction above this as a facet mix-up",
    )
    parser.add_argument(
        "--support-tolerance-cm",
        type=float,
        default=ABOVE_TOLERANCE_CM,
        help="how far ABOVE a support a resting object may sit",
    )
    parser.add_argument(
        "--support-penetration-cm",
        type=float,
        default=BELOW_TOLERANCE_CM,
        help="how far BELOW a raised support; unbounded below the floor",
    )
    parser.add_argument(
        "--support-primitives",
        action="store_true",
        help="also accept rest on the released support_surface/ primitives",
    )
    parser.add_argument(
        "--any-support-height",
        action="store_true",
        help="correct every rest segment regardless of height; measured and bad",
    )
    parser.add_argument(
        "--seat-vertical",
        action="store_true",
        help="also shift along the support normal; measured and rejected",
    )
    parser.add_argument("--episodes", type=int, nargs="*", default=None)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    options = Options(
        fps=args.fps,
        rest_speed=args.rest_speed,
        min_rest_frames=args.min_rest_frames,
        smooth_frames=args.smooth_frames,
        max_tilt_deg=args.max_tilt_deg,
        support_tolerance_cm=args.support_tolerance_cm,
        support_penetration_cm=args.support_penetration_cm,
        support_primitives=args.support_primitives,
        any_support_height=args.any_support_height,
        seat_vertical=args.seat_vertical,
    )
    record = run(args.dataset_root, args.npz_out, options, only=args.episodes)
    print(json.dumps(record["summary"], indent=2, sort_keys=True))
    if args.out:
        save(record, args.out)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
