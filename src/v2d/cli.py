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


#: Stages that keep their own argparse because each is also spawned as an isolated subprocess.
#: The CLI forwards the remaining arguments to them unchanged, so there is one entry point
#: without a second copy of every flag.
FORWARDED: dict[str, tuple[str, str]] = {
    "clean-support": ("v2d.clean.support_contact_cli", "clean the Tier 2 object pose"),
    "clean-root": (
        "v2d.clean.root_planting",
        "the root cleaner that failed; kept with its measurement",
    ),
    "tilt-yaw": (
        "v2d.clean.tilt_yaw",
        "the tilt/yaw oracles (upper bounds, not usable in a submission)",
    ),
    "retarget": ("v2d.retarget.batch", "retarget every clip to the G1"),
    "retarget-sequences": ("v2d.retarget.sequences", "convert one episode into soma_to_g1 layout"),
    "retarget-validate": ("v2d.retarget.validate", "check a retargeted clip against its input"),
    "retarget-qa": ("v2d.retarget.qa", "aggregate the retargeter's own diagnostics"),
    "score": ("v2d.score.track2", "score a prediction with the kit's metric code"),
    "probe-supports": ("v2d.probe.support_root", "what Tier 2 corrupts; are the primitives usable"),
    "noise-channels": (
        "v2d.analyze.noise_channels",
        "which channels Tier 2 corrupts, how hard, and which ones the metric reads",
    ),
    "deep-dive": (
        "v2d.analyze.deep_dive",
        "a metrics-level reading of a finished run, per clip and per bucket",
    ),
    "deep-dive-figures": (
        "v2d.analyze.deep_dive_figures",
        "the figures behind the metrics deep dive",
    ),
    "triage-weak-clips": (
        "v2d.analyze.triage",
        "why each weak clip is weak, from its own rollout rather than its score",
    ),
    "match-object-reference": (
        "v2d.stage.object_reference",
        "which source's object trajectory each clip of a retargeted tree carries",
    ),
    "pick-checkpoint": (
        "v2d.select.checkpoints",
        "choose one checkpoint per episode without reading any ground truth",
    ),
    "compare-selection-rules": (
        "v2d.select.rules",
        "what the registered checkpoint rule costs against the alternatives",
    ),
    "adopt-check": (
        "v2d.select.adopt",
        "which second-wave clips replace the board, decided on the input reference alone",
    ),
    "prereg-cleanup": ("v2d.score.prereg_cleanup", "apply docs/EXP_CLEANUP_CONTACT.md's rule"),
    "prereg-support-root": (
        "v2d.score.prereg_support_root",
        "apply docs/EXP_SUPPORT_AND_ROOT.md's rule",
    ),
    "pack-tier1": ("v2d.submit.tier1", "pack a Tier 1 submission with the kit's own packer"),
    "taxonomy": (
        "v2d.analyze.taxonomy",
        "class every shipping rollout's failure by mechanism",
    ),
    "stage-entry-wave2": (
        "v2d.submit.wave2",
        "stage the final-candidate Tier 2 entry from the adoption record",
    ),
    "rollout-to-npz": (
        "v2d.submit.rollout",
        "convert a FlashCHORD rollout into the released frame for scoring",
    ),
    "rebase-assets": (
        "v2d.submit.rebase_assets",
        "rewrite a motion parquet's absolute asset paths for another machine",
    ),
}


def _forward(module: str, argv: list[str]) -> int:
    """Run a stage that owns its own argparse, with `argv` passed through.

    A stage refuses with `SystemExit("why")`, which is what Python itself prints before exiting
    non-zero. Passing that string to `int()` -- as this did -- turned every such refusal into a
    ValueError traceback with the real reason buried in it, so a string code is printed and
    reported as exit 2 instead.
    """
    import importlib

    stage = importlib.import_module(module)
    try:
        stage.main(argv)
    except SystemExit as exit_code:
        code = exit_code.code
        if code is None or code is True or code is False:
            return int(bool(code))
        if isinstance(code, int):
            return code
        print(code, file=sys.stderr)
        return 2
    return 0


def _gate_cleaner(args: argparse.Namespace) -> int:
    """Run the known-answer and do-no-harm gates, and check them against docs/GATES.md."""
    from .clean.gates import Injection, run, save
    from .clean.support_contact import Options

    record = run(
        Path(args.dataset_root),
        Options(support_primitives=args.support_primitives),
        Injection(tilt_deg=args.tilt_deg, seed=args.seed),
        only=args.episodes,
    )
    summary = record["summary"]
    checks = {
        "G1_mean_recovery": {
            "bar": ">= 0.70",
            "value": summary["mean_recovered_fraction"],
            "pass": bool((summary["mean_recovered_fraction"] or 0) >= 0.70),
        },
        "G1b_worst_recovery": {
            "bar": ">= 0.20",
            "value": summary["worst_recovered_fraction"],
            "pass": bool((summary["worst_recovered_fraction"] or -1) >= 0.20),
        },
        "G2_do_no_harm_tilt": {
            "bar": "<= 3.0 deg",
            "value": summary["mean_do_no_harm_tilt_deg"],
            "pass": bool(summary["mean_do_no_harm_tilt_deg"] <= 3.0),
        },
        "G2b_do_no_harm_drift": {
            "bar": "<= 1e-6 cm",
            "value": summary["worst_do_no_harm_drift_cm"],
            "pass": bool(summary["worst_do_no_harm_drift_cm"] <= 1e-6),
        },
    }
    record["checks"] = checks
    record["verdict"] = "PASS" if all(c["pass"] for c in checks.values()) else "FAIL"
    print(
        "\n"
        + json.dumps(
            {"summary": summary, "checks": checks, "verdict": record["verdict"]},
            indent=2,
            sort_keys=True,
        )
    )
    if args.out:
        save(record, Path(args.out))
        print(f"wrote {args.out}", file=sys.stderr)
    return 0 if record["verdict"] == "PASS" else 1


def _contact(args: argparse.Namespace) -> int:
    """Hand-object contact coverage across the retargeted arms, with paired sign tests."""
    from .score.contact import coverage, ladder, paired, sign_test

    series: dict[str, dict[int, float]] = {
        "tier1": coverage(Path(args.baseline_logs), "tier1"),
        "tier2": coverage(Path(args.baseline_logs), "tier2"),
    }
    for arm in args.arms:
        found = coverage(Path(args.arms_root) / arm / "logs")
        if found:
            series[arm] = found
    means = ladder(series)
    comparisons = {}
    for arm in args.arms:
        if arm in series and args.against in series:
            difference = paired(series[arm], series[args.against])
            comparisons[f"{arm}_vs_{args.against}"] = {
                "mean": round(sum(difference.values()) / len(difference), 4)
                if difference
                else None,
                **sign_test(difference),
            }
    populated = [value for value in series.values() if value]
    shared = sorted(set.intersection(*(set(value) for value in populated))) if populated else []
    record = {
        "what": "hand-object contact coverage per arm, at the retargeter's 2 cm threshold",
        "episodes_scored": shared,
        "contact_mean": means,
        "paired": comparisons,
        "per_episode": {
            name: {str(episode): round(value[episode], 4) for episode in shared}
            for name, value in series.items()
            if value
        },
    }
    print(
        json.dumps(
            {k: v for k, v in record.items() if k != "per_episode"}, indent=2, sort_keys=True
        )
    )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}", file=sys.stderr)
    return 0


def _brev_ready(args: argparse.Namespace) -> int:
    """Exit 0 if a `brev ls` row is ready for SSH (or, with --dead-check, is unrecoverable)."""
    from .brev import is_dead, is_shell_ready

    if args.dead_check:
        return 0 if is_dead(args.row) else 1
    return 0 if is_shell_ready(args.row) else 1


def _gate_report(args: argparse.Namespace) -> int:
    """Log a finished Brev gate run to W&B and push its artefacts to the private HF repo."""
    from .observe.gate_report import report

    record = report(
        Path(args.directory),
        phase=args.phase,
        instance_type=args.instance_type,
        usd_per_hour=args.usd_per_hour,
        wall_hours=args.wall_hours,
        motion_root=Path(args.motion_root) if args.motion_root else None,
        hf_repo=None if args.no_upload else args.hf_repo,
        project=args.project,
    )
    print(json.dumps(record, indent=2, sort_keys=True))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"wrote {out}", file=sys.stderr)
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

    ready = sub.add_parser(
        "brev-ready",
        help="is a `brev ls` row ready for SSH?",
        description=_brev_ready.__doc__,
    )
    ready.add_argument("--row", required=True, help="one line of `brev ls` output")
    ready.add_argument(
        "--dead-check",
        action="store_true",
        help="instead test whether the row is in an unrecoverable state",
    )
    ready.set_defaults(func=_brev_ready)

    gate = sub.add_parser(
        "gate-report",
        help="log a finished Brev gate run to W&B and the private HF repo",
        description=_gate_report.__doc__,
    )
    gate.add_argument("directory", help="the results/gate directory copied back from the instance")
    gate.add_argument("--phase", default="probe", choices=["probe", "full"])
    gate.add_argument("--instance-type")
    gate.add_argument("--usd-per-hour", type=float)
    gate.add_argument("--wall-hours", type=float)
    gate.add_argument("--motion-root", help="where to look up the reference provenance")
    gate.add_argument("--project", default="v2d-track2")
    gate.add_argument("--hf-repo", default="v2d-track2")
    gate.add_argument("--no-upload", action="store_true", help="skip the Hugging Face upload")
    gate.add_argument("-o", "--out", help="also write the record here")
    gate.set_defaults(func=_gate_report)

    gates = sub.add_parser(
        "gate-cleaner",
        help="known-answer and do-no-harm gates for the object cleaner",
        description=_gate_cleaner.__doc__,
    )
    gates.add_argument("--dataset-root", required=True)
    gates.add_argument("--support-primitives", action="store_true")
    gates.add_argument("--tilt-deg", type=float, default=12.0)
    gates.add_argument("--seed", type=int, default=0)
    gates.add_argument("--episodes", type=int, nargs="*", default=None)
    gates.add_argument("-o", "--out")
    gates.set_defaults(func=_gate_cleaner)

    contact = sub.add_parser(
        "contact", help="contact coverage across arms", description=_contact.__doc__
    )
    contact.add_argument("--arms-root", required=True, help="holds <arm>/logs per arm")
    contact.add_argument("--baseline-logs", required=True, help="the main run's log directory")
    contact.add_argument("--arms", nargs="*", default=["table", "table_root", "root"])
    contact.add_argument("--against", default="tier2", help="series to pair each arm against")
    contact.add_argument("-o", "--out")
    contact.set_defaults(func=_contact)

    # Registered so `v2d --help` lists them, but never parsed through: main() dispatches these
    # before argparse runs, because a top-level parser consumes `--help` before a subparser's
    # REMAINDER can capture it, and `v2d score --help` must reach the stage's own parser.
    for name, (_module, help_text) in FORWARDED.items():
        sub.add_parser(name, help=help_text, add_help=False)

    status = sub.add_parser(
        "observe-status", help="report which W&B / Hugging Face credentials are available"
    )
    status.add_argument("--repo", default="v2d-track2", help="also check this HF repo")
    status.set_defaults(func=_observe_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in FORWARDED:
        return _forward(FORWARDED[arguments[0]][0], arguments[1:])
    args = build_parser().parse_args(arguments)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
