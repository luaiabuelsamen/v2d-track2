"""`v2d` — one entry point for every stage of the pipeline.

    v2d --help

Each subcommand writes a JSON record under `results/`, so every number in the README and in
`docs/` is reproducible from a single command.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from . import __version__


def _brev_pick(args: argparse.Namespace) -> int:
    """Choose a GPU that actually clears the VRAM floor, or refuse to choose one."""
    from .brev import eligible, lookup, parse_search, pick_cheapest

    if args.table:
        text = Path(args.table).read_text()
    else:
        binary = shutil.which("brev")
        if binary is None:
            print("the brev CLI is not on PATH", file=sys.stderr)
            return 3
        done = subprocess.run(
            [binary, "search", "--sort", "price"], capture_output=True, text=True, check=False
        )
        text = done.stdout

    offers, skipped = parse_search(text)
    if not offers:
        print(
            "could not parse any instance from `brev search`. Refusing to guess a default: "
            "guessing is how a 16 GB V100 got proposed for a job that needs 48 GB.",
            file=sys.stderr,
        )
        return 4

    ranked = eligible(offers, args.min_vram)
    if args.show:
        print(
            f"{len(offers)} instance types listed, {len(skipped)} unparsed, "
            f"{len(ranked)} clear the {args.min_vram:g} GB floor:",
            file=sys.stderr,
        )
        for offer in ranked[: args.show]:
            print(f"  {offer.describe()}", file=sys.stderr)

    if args.type:
        chosen = lookup(offers, args.type)
        if chosen is None:
            print(
                f"--type {args.type} is not in the live table. Availability moves: "
                f"hyperstack_A6000 was listed at $0.60/h on 2026-10-24 and was gone hours later. "
                f"Cheapest that fits now: {ranked[0].describe() if ranked else 'nothing'}",
                file=sys.stderr,
            )
            return 5
        if not chosen.meets(args.min_vram):
            print(
                f"--type {args.type} has {chosen.vram_per_gpu_gb:g} GB per GPU, below the "
                f"{args.min_vram:g} GB floor. Refusing.",
                file=sys.stderr,
            )
            return 6
    else:
        chosen = pick_cheapest(offers, args.min_vram)
        if chosen is None:
            print(
                f"nothing on the live table clears {args.min_vram:g} GB per GPU.", file=sys.stderr
            )
            return 7

    if args.json:
        print(json.dumps(chosen.__dict__, indent=2, sort_keys=True))
    else:
        # price, type, vram -- consumed by brev/gate_launch.sh
        print(f"{chosen.usd_per_hour} {chosen.instance_type} {chosen.vram_per_gpu_gb:g}")
    return 0


def _observe_status(args: argparse.Namespace) -> int:
    """Report which observability credentials this machine has, without revealing any."""
    from .observe import hf_available, wandb_available, which_credentials
    from .observe.hf_store import open_store

    found = which_credentials()
    print(f"wandb : {'ready' if wandb_available() else 'absent'}  (found at {found['wandb']})")
    print(f"hf    : {'ready' if hf_available() else 'absent'}  (found at {found['huggingface']})")
    if args.repo and hf_available():
        store = open_store(args.repo, create=False)
        print(f"repo  : {store.repo_id or '-'}  {store.url or store.reason}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="v2d",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"v2d {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    pick = sub.add_parser(
        "brev-pick",
        help="choose the cheapest Brev GPU that clears FlashCHORD's VRAM floor",
        description=_brev_pick.__doc__,
    )
    pick.add_argument("--min-vram", type=float, default=46.0, help="GB per GPU (default 46)")
    pick.add_argument("--type", help="validate this instance type instead of choosing one")
    pick.add_argument("--table", help="parse this file instead of calling `brev search`")
    pick.add_argument("--json", action="store_true", help="emit the full record")
    pick.add_argument(
        "--show",
        type=int,
        default=0,
        metavar="N",
        help="also print the N cheapest eligible types to stderr",
    )
    pick.set_defaults(func=_brev_pick)

    status = sub.add_parser(
        "observe-status", help="report which W&B / Hugging Face credentials are available"
    )
    status.add_argument("--repo", default="v2d-track2", help="also check this HF repo")
    status.set_defaults(func=_observe_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
