#!/usr/bin/env python3
"""Retune flash_chord's object-position reward: its shaping std, and optionally its weight.

NVIDIA rewards the object's position with `exp(-|p-p*|^2 / std^2)` at `std = 0.2 m`. The metrics
deep dive says that is far too loose for the metric that is actually scored. MP-SR needs the
time-averaged object position error under **3 cm**, and eleven of our clips miss it by less than
1.5 cm while their orientation margin is comfortable; at std 0.2 m the reward is still 0.98 of its
maximum at 3 cm, so the policy earns almost full credit exactly where the competition says it has
failed. Tightening the std puts the gradient back inside the band the score is decided in:

    error     std 0.20    std 0.08
    1 cm        0.975       0.855
    3 cm        0.978*      0.249      (*0.9778 at 3 cm vs 0.9975 at 1 cm: nearly flat)
    5 cm        0.939       0.021
   10 cm        0.779       0.000

`--weight` is separate and defaults to leaving the weight alone. It scales the term's contribution
against body tracking (`body_joint_position: 5.0`), which is what a clip that performs the motion
without the object is exploiting. The weight that matters at run time is the **curriculum's**, not
the objective config's -- `set_weights` applies the curriculum mapping every stage -- so the
curriculum is patched too, and only when a weight is asked for.

Run inside the training container, before training:

    python patch_object_position.py /workspace/src/flash_chord --std 0.08 [--weight 5.0] [--check]

Idempotent at the same values, and refuses a tree already patched at different ones rather than
reporting success while the arm on the record is not the arm that trained.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

#: Written into the objective yaml so a patched tree states what it carries.
MARKER = "# v2d object-position"
MARKER_LINE = re.compile(rf"^{re.escape(MARKER)} std=([0-9.eE+-]+) weight=(none|[0-9.eE+-]+)\s*$",
                         re.MULTILINE)

DEFAULT_STD = 0.08
STOCK_STD = 0.2
STOCK_WEIGHT = 1.0
#: Upstream writes these as literal text; f"{1.0:g}" is "1" and would match nothing.
STOCK_STD_TEXT = "0.2"
STOCK_WEIGHT_TEXT = "1.0"


def objective_py_edits(std: float, weight: float | None) -> tuple[tuple[str, str], ...]:
    edits = [
        (
            f"    object_position_std: float = {STOCK_STD_TEXT}\n",
            f"    object_position_std: float = {std:g}\n",
        )
    ]
    if weight is not None:
        edits.append(
            (
                f"    object_position_weight: float = {STOCK_WEIGHT_TEXT}\n",
                f"    object_position_weight: float = {weight:g}\n",
            )
        )
    return tuple(edits)


def objective_yaml_edits(std: float, weight: float | None) -> tuple[tuple[str, str], ...]:
    edits = [
        (
            f"object_position_std: {STOCK_STD_TEXT}\n",
            f"{MARKER} std={std:g} weight={'none' if weight is None else f'{weight:g}'}\n"
            f"object_position_std: {std:g}\n",
        )
    ]
    if weight is not None:
        edits.append(
            (
                f"object_position_weight: {STOCK_WEIGHT_TEXT}\n",
                f"object_position_weight: {weight:g}\n",
            )
        )
    return tuple(edits)


#: The curriculum's mapping is what `set_weights` applies every stage, so a weight change that
#: does not land here does not happen at all. Patching the anchored block covers the aliased
#: stages with it.
def curriculum_edits(weight: float) -> tuple[tuple[str, str], ...]:
    return (
        (
            f"      object_position: {STOCK_WEIGHT_TEXT}\n",
            f"      object_position: {weight:g}\n",
        ),
    )


def carried(objective_yaml: Path) -> tuple[float, float | None] | None:
    """The (std, weight) a tree already carries, or None if it is unpatched."""
    if not objective_yaml.is_file():
        return None
    found = MARKER_LINE.search(objective_yaml.read_text())
    if not found:
        return None
    return float(found.group(1)), (
        None if found.group(2) == "none" else float(found.group(2))
    )


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


def plan_file(path: Path, edits: tuple[tuple[str, str], ...]) -> tuple[str, str | None]:
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} does not exist")
    text = path.read_text()
    patched = apply(text, edits, str(path))
    if path.suffix == ".py":
        ast.parse(patched)
    return f"patched: {path}", patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="flash_chord source root")
    parser.add_argument(
        "--std",
        type=float,
        default=DEFAULT_STD,
        help=f"shaping std in metres (default {DEFAULT_STD:g}, stock {STOCK_STD:g})",
    )
    parser.add_argument("--weight", type=float,
                        help=f"object_position weight (stock {STOCK_WEIGHT:g}); omit to leave it")
    parser.add_argument("--check", action="store_true", help="report, write nothing")
    args = parser.parse_args(argv)

    for name, value in (("--std", args.std), ("--weight", args.weight)):
        if value is None:
            continue
        if not (value > 0.0) or value != value or value == float("inf"):
            raise SystemExit(f"FATAL: {name} must be finite and positive, got {value!r}")

    objective_yaml = args.root / "configs/objective/recon_body.yaml"
    already = carried(objective_yaml)
    if already is not None:
        if already == (args.std, args.weight):
            print(f"already patched at std {args.std:g}, weight "
                  f"{'unchanged' if args.weight is None else f'{args.weight:g}'}")
            return 0
        raise SystemExit(
            f"FATAL: {objective_yaml} already carries std={already[0]:g} "
            f"weight={already[1] if already[1] is not None else 'none'}, but "
            f"std={args.std:g} weight={args.weight if args.weight is not None else 'none'} "
            "was asked for.\n"
            "  Leaving the first values in place and reporting success would put the wrong arm on "
            "the record. Rebuild the container."
        )

    targets: list[tuple[Path, tuple[tuple[str, str], ...]]] = [
        (args.root / "objectives/recon_body.py", objective_py_edits(args.std, args.weight)),
        (objective_yaml, objective_yaml_edits(args.std, args.weight)),
    ]
    curricula: list[Path] = []
    if args.weight is not None:
        anchor = curriculum_edits(args.weight)[0][0]
        for path in sorted((args.root / "configs/curriculum").glob("*.yaml")):
            if anchor in path.read_text():
                curricula.append(path)
        if not curricula:
            raise SystemExit(
                f"FATAL: no curriculum names {anchor.strip()!r}, so a weight change would not "
                "reach training; set_weights applies the curriculum mapping, not the config."
            )
        targets.extend((path, curriculum_edits(args.weight)) for path in curricula)

    # Plan everything before writing anything: a tree with a patched objective and an unpatched
    # curriculum trains a recipe nobody chose.
    planned: list[tuple[Path, str]] = []
    report: list[str] = []
    for path, edits in targets:
        message, text = plan_file(path, edits)
        report.append(message)
        if text is not None:
            planned.append((path, text))
    if not args.check:
        for path, text in planned:
            path.write_text(text)
    else:
        report = [m.replace("patched:", "would patch:") for m in report]

    print("\n".join(report))
    print(
        f"object-position reward: std {args.std:g} m (stock {STOCK_STD:g}), weight "
        + (f"{args.weight:g} in {len(curricula)} curriculum file(s)" if args.weight is not None
           else "unchanged")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
