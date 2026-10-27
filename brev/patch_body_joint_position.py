#!/usr/bin/env python3
"""Retune flash_chord's body-joint tracking reward: its weight, and optionally its shaping std.

`body_joint_position_weight: 5.0` is the largest weight in `recon_body`, five times the object's.
That ordering is why a clip can score well on the body while the object is on the floor: the
policy is paid five times more for the arm sweep than for what the arm is carrying, and
docs/FAILURE_MOMENTS.md shows e19's cane leaving a grip the reference never opens while the body
goes on with the motion. Lowering it to 1.0 makes the object and the body worth the same, which is
the one change the nine weak clips have in common.

It is deliberately the same shape as `patch_object_position.py`: the weight that matters at run
time is the **curriculum's**, because `set_weights` applies the curriculum mapping every stage, so
the curriculum is patched too and a weight change that never reaches training is a hard failure
rather than a silent one.

Run inside the training container, before training:

    python patch_body_joint_position.py /workspace/src/flash_chord \\
        --weight 1.0 [--std 1.0] [--check]

Idempotent at the same values, and refuses a tree already patched at different ones rather than
reporting success while the arm on the record is not the arm that trained.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

#: Written into the objective yaml so a patched tree states what it carries.
MARKER = "# v2d body-joint-position"
MARKER_LINE = re.compile(
    rf"^{re.escape(MARKER)} weight=([0-9.eE+-]+) std=(none|[0-9.eE+-]+)\s*$", re.MULTILINE
)

DEFAULT_WEIGHT = 1.0
STOCK_WEIGHT = 5.0
STOCK_STD = 1.0
#: Upstream writes these as literal text; f"{5.0:g}" is "5" and would match nothing.
STOCK_WEIGHT_TEXT = "5.0"
STOCK_STD_TEXT = "1.0"


def objective_py_edits(weight: float, std: float | None) -> tuple[tuple[str, str], ...]:
    edits = [
        (
            f"    body_joint_position_weight: float = {STOCK_WEIGHT_TEXT}\n",
            f"    body_joint_position_weight: float = {weight:g}\n",
        )
    ]
    if std is not None:
        edits.append(
            (
                f"    body_joint_position_std: float = {STOCK_STD_TEXT}\n",
                f"    body_joint_position_std: float = {std:g}\n",
            )
        )
    return tuple(edits)


def objective_yaml_edits(weight: float, std: float | None) -> tuple[tuple[str, str], ...]:
    edits = [
        (
            f"body_joint_position_weight: {STOCK_WEIGHT_TEXT}\n",
            f"{MARKER} weight={weight:g} std={'none' if std is None else f'{std:g}'}\n"
            f"body_joint_position_weight: {weight:g}\n",
        )
    ]
    if std is not None:
        edits.append(
            (
                f"body_joint_position_std: {STOCK_STD_TEXT}\n",
                f"body_joint_position_std: {std:g}\n",
            )
        )
    return tuple(edits)


def curriculum_edits(weight: float) -> tuple[tuple[str, str], ...]:
    """The mapping `set_weights` applies every stage; without this the change never lands."""
    return (
        (
            f"      body_joint_position: {STOCK_WEIGHT_TEXT}\n",
            f"      body_joint_position: {weight:g}\n",
        ),
    )


def carried(objective_yaml: Path) -> tuple[float, float | None] | None:
    """The (weight, std) a tree already carries, or None if it is unpatched."""
    if not objective_yaml.is_file():
        return None
    found = MARKER_LINE.search(objective_yaml.read_text())
    if not found:
        return None
    return float(found.group(1)), (None if found.group(2) == "none" else float(found.group(2)))


def apply(text: str, edits: tuple[tuple[str, str], ...], where: str) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"FATAL: {where}: anchor occurs {count} times, expected exactly 1.\n"
                f"  anchor: {old!r}\n"
                "  flash_chord has changed, or this tree is already patched at other values; "
                "re-derive the patch against the installed source."
            )
        text = text.replace(old, new, 1)
    return text


def plan_file(path: Path, edits: tuple[tuple[str, str], ...]) -> tuple[str, str]:
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} does not exist")
    patched = apply(path.read_text(), edits, str(path))
    if path.suffix == ".py":
        ast.parse(patched)
    return f"patched: {path}", patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="flash_chord source root")
    parser.add_argument(
        "--weight",
        type=float,
        default=DEFAULT_WEIGHT,
        help=f"body_joint_position weight (default {DEFAULT_WEIGHT:g}, stock {STOCK_WEIGHT:g})",
    )
    parser.add_argument(
        "--std", type=float, help=f"shaping std in rad (stock {STOCK_STD:g}); omit to leave it"
    )
    parser.add_argument("--check", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)

    for name, value in (("--weight", args.weight), ("--std", args.std)):
        if value is None:
            continue
        if not (value > 0.0) or value != value or value == float("inf"):
            raise SystemExit(f"FATAL: {name} must be finite and positive, got {value!r}")

    objective_yaml = args.root / "configs/objective/recon_body.yaml"
    already = carried(objective_yaml)
    if already is not None:
        if already == (args.weight, args.std):
            print(
                f"already patched at weight {args.weight:g}, std "
                f"{'unchanged' if args.std is None else f'{args.std:g}'}"
            )
            return 0
        raise SystemExit(
            f"FATAL: {objective_yaml} already carries weight={already[0]:g} "
            f"std={already[1] if already[1] is not None else 'none'}, but "
            f"weight={args.weight:g} std={args.std if args.std is not None else 'none'} "
            "was asked for.\n"
            "  Leaving the first values in place and reporting success would put the wrong arm on "
            "the record. Rebuild the container."
        )

    targets: list[tuple[Path, tuple[tuple[str, str], ...]]] = [
        (args.root / "objectives/recon_body.py", objective_py_edits(args.weight, args.std)),
        (objective_yaml, objective_yaml_edits(args.weight, args.std)),
    ]
    anchor = curriculum_edits(args.weight)[0][0]
    curricula = [
        path
        for path in sorted((args.root / "configs/curriculum").glob("*.yaml"))
        if anchor in path.read_text()
    ]
    if not curricula:
        raise SystemExit(
            f"FATAL: no curriculum names {anchor.strip()!r}, so the weight change would not reach "
            "training; set_weights applies the curriculum mapping, not the config."
        )
    targets.extend((path, curriculum_edits(args.weight)) for path in curricula)

    planned: list[tuple[Path, str]] = []
    report: list[str] = []
    for path, edits in targets:
        message, text = plan_file(path, edits)
        report.append(message)
        planned.append((path, text))
    if not args.check:
        for path, text in planned:
            path.write_text(text)
    for line in report:
        print(("would patch: " + line.split(": ", 1)[1]) if args.check else line)
    print(
        f"body_joint_position weight {STOCK_WEIGHT:g} -> {args.weight:g}"
        + ("" if args.std is None else f", std {STOCK_STD:g} -> {args.std:g}")
        + f"; {len(curricula)} curriculum file(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
