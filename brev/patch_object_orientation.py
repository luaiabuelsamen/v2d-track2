#!/usr/bin/env python3
"""Add an object-orientation reward term to flash_chord's ReconBody objective, in place.

NVIDIA's ReconBody rewards the object by POSITION only -- `exp(-|p-p*|^2 / 0.2^2)` -- and has no
orientation term at all, while ADD-AUC is computed over mesh points and so is sensitive to the
object's angle. On the clips that already hold the object we measure 4.2-5.3 cm of position error
against 25-45 degrees of orientation error, so the part that is scored is the part that is never
rewarded. This adds the missing term, mirroring the existing `palm_orientation` term exactly: the
same double-cover-aware geodesic angle and the same `exp(-theta^2 / std^2)` shaping, with its own
std. Weight 1.0 is copied from `palm_orientation` (and `pelvis_orientation`), the house calibration
for an orientation term in this objective.

The std is `--std`, defaulting to 0.3 rad (17.2 deg), i.e. half reward at 14.3 deg. That is tighter
than `palm_orientation`'s 0.4 for one reason: the object's angle is scored and the palm's is not.
Both values are flat across the 64-96 deg our checkpoints actually sit at, so the choice only
changes the slope inside the 0-20 deg band where ADD-AUC is won, and 0.3 doubles it. It is a
parameter because it is the one number in the term that is a guess -- the evaluator's own
orientation gate is 1.5 rad, which is far too loose to read a value off.

The term is APPENDED at index 12 so every existing term keeps its index: a reordering would
silently permute the weight vector. The curriculum config has to be patched too, because
`set_weights` raises unless the mapping's key set matches the term names exactly -- so a half-patch
fails loudly at the first curriculum stage rather than training something unintended.

Run inside the training container, before training:

    python patch_object_orientation.py /workspace/src/flash_chord [--check]

Idempotent: a tree that already carries the term is left alone and reported. Exits non-zero, having
changed nothing, if any anchor is missing or ambiguous.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

MARKER = "object_orientation"

# 0.3 rad = 17.2 deg; see the module docstring for why it is not palm_orientation's 0.4.
DEFAULT_STD = 0.3


def objective_edits(std: float) -> tuple[tuple[str, str], ...]:
    """The edits to `objectives/recon_body.py`, for one choice of std."""
    return (
        # (1) the term name, appended so the existing indices are untouched
        (
            '    "palm_orientation",\n    "force_closure",\n)\n',
            '    "palm_orientation",\n    "force_closure",\n    "object_orientation",\n)\n',
        ),
        # (2) the term count the kernel strides by
        (
            "_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(12)",
            "_RECON_BODY_OBJECTIVE_TERM_COUNT = wp.constant(13)",
        ),
        # (3) the config: a weight and a std of its own
        (
            "    force_closure_weight: float = 5.0\n",
            "    force_closure_weight: float = 5.0\n    object_orientation_weight: float = 1.0\n",
        ),
        (
            "    palm_orientation_std: float = 0.4\n",
            "    palm_orientation_std: float = 0.4\n"
            f"    object_orientation_std: float = {std:g}\n",
        ),
        # (4) validation: finite, and positive like every other shaping std
        (
            "            self.palm_orientation_std,\n            self.force_closure_min_support,",
            "            self.palm_orientation_std,\n            self.object_orientation_std,"
            "\n            self.force_closure_min_support,",
        ),
        (
            "                self.palm_orientation_std,\n            )\n        ):",
            "                self.palm_orientation_std,\n                self.object_orientation_std,"
            "\n            )\n        ):",
        ),
        # (5) the weights tuple, in term order, so index 12 is the new weight
        (
            "            self.force_closure_weight,\n        )",
            "            self.force_closure_weight,\n            self.object_orientation_weight,\n        )",
        ),
        # (6) the kernel needs the reference object quaternion and the new std
        (
            "    object_target_pos_w: wp.array(dtype=wp.vec3),\n",
            "    object_target_pos_w: wp.array(dtype=wp.vec3),\n"
            "    object_target_quat_w: wp.array(dtype=wp.quat),\n",
        ),
        (
            "    palm_orientation_std: float,\n",
            "    palm_orientation_std: float,\n    object_orientation_std: float,\n",
        ),
        # (7) the term itself, at index 12
        (
            "    terms[term_base + 11] = force_closure[world]\n",
            "    terms[term_base + 11] = force_closure[world]\n"
            "\n"
            "    # Our addition. The twelve stock terms track the object by position only, so a "
            "policy\n"
            "    # can hold the object in the right place at the wrong angle and lose ADD-AUC for "
            "free.\n"
            "    # Mirrors palm_orientation: same double-cover-aware geodesic angle, same "
            "exp(-e^2/s^2).\n"
            "    object_orientation = wp.normalize(wp.transform_get_rotation(body_q[body_base + "
            "object_body_id]))\n"
            "    object_orientation_error = quat_geodesic_angle(object_orientation, "
            "object_target_quat_w[world])\n"
            "    terms[term_base + 12] = wp.exp(\n"
            "        -(object_orientation_error * object_orientation_error)\n"
            "        / (object_orientation_std * object_orientation_std)\n"
            "    )\n",
        ),
        # (8) pass both at launch, in signature order
        (
            "                self.command.body_target_pos_w,\n",
            "                self.command.body_target_pos_w,\n                self.command.body_target_quat_w,\n",
        ),
        (
            "                self.config.palm_orientation_std,\n",
            "                self.config.palm_orientation_std,\n"
            "                self.config.object_orientation_std,\n",
        ),
    )


def objective_yaml_edits(std: float) -> tuple[tuple[str, str], ...]:
    """The edits to `configs/objective/recon_body.yaml`, for one choice of std."""
    return (
        (
            "object_position_weight: 1.0\n",
            "object_position_weight: 1.0\nobject_orientation_weight: 1.0\n",
        ),
        (
            "palm_orientation_std: 0.4\n",
            f"palm_orientation_std: 0.4\nobject_orientation_std: {std:g}\n",
        ),
    )


STD_IN_YAML = re.compile(r"^object_orientation_std:\s*([0-9.eE+-]+)\s*$", re.MULTILINE)


def existing_std(path: Path) -> float | None:
    """The std a tree already carries, or None if it is not patched yet.

    A box patched at one std and then asked for another would otherwise keep the first and report
    success, and the arm on the record would be the wrong one.
    """
    if not path.is_file():
        return None
    found = STD_IN_YAML.search(path.read_text())
    return float(found.group(1)) if found else None

# set_weights() compares key SETS, so every curriculum stage that names the weights must name this
# one too. Patching the anchor block covers the aliased stages with it.
CURRICULUM_EDIT = ("      force_closure: 5.0\n", "      force_closure: 5.0\n      object_orientation: 1.0\n")


def apply(text: str, edits: tuple[tuple[str, str], ...], where: str) -> str:
    for old, new in edits:
        count = text.count(old)
        if count != 1:
            raise SystemExit(
                f"FATAL: {where}: anchor occurs {count} times, expected exactly 1.\n"
                f"  anchor: {old!r}\n"
                "  flash_chord has changed; re-derive the patch against the installed source "
                "rather than letting it apply somewhere unintended."
            )
        text = text.replace(old, new, 1)
    return text


def plan_file(path: Path, edits: tuple[tuple[str, str], ...]) -> tuple[str, str | None]:
    """(report, new text) for one file. None means it is already patched and needs no write."""
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} does not exist")
    text = path.read_text()
    if MARKER in text:
        return f"already patched: {path}", None
    patched = apply(text, edits, str(path))
    if path.suffix == ".py":
        # A syntax error here would surface 13 minutes later as a container crash.
        ast.parse(patched)
    return f"patched: {path}", patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="flash_chord source root, e.g. /workspace/src/flash_chord")
    parser.add_argument(
        "--check", action="store_true", help="report what would change, write nothing"
    )
    parser.add_argument(
        "--std",
        type=float,
        default=DEFAULT_STD,
        help=f"shaping std in radians for the new term (default {DEFAULT_STD:g})",
    )
    args = parser.parse_args(argv)

    if not (args.std > 0.0) or args.std != args.std or args.std == float("inf"):
        raise SystemExit(f"FATAL: --std must be finite and positive, got {args.std!r}")

    objective_yaml = args.root / "configs/objective/recon_body.yaml"
    carried = existing_std(objective_yaml)
    if carried is not None and abs(carried - args.std) > 1e-12:
        raise SystemExit(
            f"FATAL: {objective_yaml} already carries object_orientation_std {carried:g}, but "
            f"{args.std:g} was asked for.\n"
            "  The patch is idempotent, so it would leave the first value in place and report "
            "success,\n"
            "  and the arm on the record would not be the arm that trained. Rebuild the container "
            "or\n"
            f"  pass --std {carried:g}."
        )

    targets: list[tuple[Path, tuple[tuple[str, str], ...]]] = [
        (args.root / "objectives/recon_body.py", objective_edits(args.std)),
        (objective_yaml, objective_yaml_edits(args.std)),
    ]
    # Every curriculum that names the ReconBody weights, not just the one recipe we run: a stage
    # whose key set is short by one raises at the moment it is entered, hours into a run.
    #
    # Selected by the ReconBody weight block itself, NOT by the word force_closure. The container
    # ships four curricula naming force_closure and only two are ReconBody: reference.yaml and
    # tracking_250m_mixed_reset.yaml are different objectives entirely (object_keypoints,
    # hand_keypoints, contact_wrench_support) whose force_closure is an interpolation,
    # `${objective.force_closure.weight}`. Selecting on the word made the patch refuse on those
    # files, and because it is atomic, nothing was patched at all -- which stopped every box of
    # the 2026-10-26 relaunch from training and cost seven clip claims. Adding
    # object_orientation to another objective's key set would be wrong anyway: set_weights()
    # compares key sets, so it would break THAT curriculum the moment it was used.
    curriculum_dir = args.root / "configs/curriculum"
    curricula, foreign = [], []
    for path in sorted(curriculum_dir.glob("*.yaml")):
        text = path.read_text()
        if CURRICULUM_EDIT[0] in text or MARKER in text:
            curricula.append(path)
        elif "force_closure:" in text:
            foreign.append(path)
    if not curricula:
        raise SystemExit(
            "FATAL: no curriculum config carries the ReconBody weight block "
            f"({CURRICULUM_EDIT[0].strip()!r}); patch is incomplete"
        )
    targets.extend((path, (CURRICULUM_EDIT,)) for path in curricula)

    # Plan every file BEFORE writing any of them. A patch that edits the objective, then fails on
    # the curriculum, leaves a tree that trains with a 13-term objective and a 12-key curriculum
    # and dies at the first stage change -- hours in, having billed for all of it.
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
    elif planned:
        report = [m.replace("patched:", "would patch:") for m in report]

    print("\n".join(report))
    for path in foreign:
        print(
            f"left alone: {path} -- names force_closure but not the ReconBody weight block, "
            "so it is a different objective"
        )
    print(
        f"object-orientation term: weight 1.0, std {args.std:g} rad "
        f"({args.std * 180.0 / 3.141592653589793:.1f} deg), term index 12 of 13"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
