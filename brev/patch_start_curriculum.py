#!/usr/bin/env python3
"""Start the deployment-exact resets earlier in FlashCHORD's curriculum, in place.

Two things decide where an episode begins. The curriculum's `voc_scale` says how much the virtual
object controller -- an external wrench that drives the object onto its reference pose -- helps,
and `immediate_first_frame_probability` says how often a reset puts the robot at frame 0 in the
deployment-exact way (`lifecycle/reset.py:sample_recon_body_reset`: an immediate reset skips the
shoulder spread, skips the finger opening, sets `steps_since_reset` past the freeze, and takes the
*curriculum* VOC scale; a random-frame reset instead starts at `reset.reset_voc_scale`).

Upstream brings the two together only at the very end: VOC reaches zero at 60% of the budget
(209,999,872 steps) but deployment-exact starts do not appear until 82.5% (288,749,568), and the
terminal 50/50 mixture not until 87.5%. For 78.7M steps -- 22% of the run -- every reset is a
random mid-clip start with the object placed by the wrench, which is a state the submission
evaluator never produces: it resets once, at frame 0, with no help. This patch moves the 0.25
mixture up to the zero-VOC stage and the 0.50 mixture up to 82.5%, so the policy spends the
zero-VOC part of the budget practising the start it is actually scored from.

Stages 1-6 are deliberately untouched: the early curriculum is what teaches the task at all, and
making its starts harder is a different experiment from making the late starts honest.

This is only half the change. A random-frame reset still sets the applied VOC scale to
`reset.reset_voc_scale` (stock 1.0) and holds it for the first 40 of the 50 freeze steps, so in the
zero-VOC stages the object is still being placed for 75% of resets. That half is a hydra override,
not a source edit, and `brev/arm_flags.sh` passes it with this patch.

Run inside the training container, before training:

    python patch_start_curriculum.py /workspace/src/flash_chord [--check]

Idempotent: a tree that already carries the change is left alone and reported. Exits non-zero,
having changed nothing, if the curriculum is not the one this patch was derived against.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MARKER = "# v2d start-curriculum"

# The one recipe this patch is derived against. The thresholds below are specific to the 2,048
# world, 349,999,104-step budget, so patching some other curriculum would move stages to step
# counts that mean nothing there.
RECIPE = "recon_body_flash_sac.yaml"

STAGE_HEADER = "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
EXPECTED_STAGES = 9
EXPECTED_THRESHOLDS = (
    "thresholds: [35000320, 70000640, 104998912, 139999232, 174999552, 209999872, "
    "288749568, 306249728, 349999104]\n"
)
# Stages 1-6, each of which must appear exactly once and must come out of this patch unchanged.
EARLY_VOC_SCALES = ("1.0", "0.75", "0.5", "0.25", "0.1", "0.05")

# Stage 7 (zero VOC, entered at 209,999,872) and stage 8 (entered at 288,749,568), together, so the
# anchor cannot match anywhere else: stage 9 already carries a probability and stage 6 is not 0.0.
OLD = (
    "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
    "    voc_scale: 0.0\n"
    "    objective_weights: *recon_body_weights\n"
    "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
    "    voc_scale: 0.0\n"
    "    objective_weights: *recon_body_weights\n"
    "    immediate_first_frame_probability: 0.25\n"
)
NEW = (
    "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
    "    voc_scale: 0.0\n"
    "    objective_weights: *recon_body_weights\n"
    "    # v2d start-curriculum: deployment-exact starts from the zero-VOC stage (209,999,872,\n"
    "    # 60% of the budget) instead of 288,749,568 (82.5%), and the terminal 50/50 mixture\n"
    "    # from 82.5% instead of 87.5%. Stages 1-6 are untouched. Pair with\n"
    "    # reset.reset_voc_scale=0.0 so the random resets stop placing the object as well.\n"
    "    immediate_first_frame_probability: 0.25\n"
    "  - _target_: flash_chord.configuration.make_curriculum_stage\n"
    "    voc_scale: 0.0\n"
    "    objective_weights: *recon_body_weights\n"
    "    immediate_first_frame_probability: 0.50\n"
)


def verify_structure(text: str, where: str) -> None:
    """Refuse a curriculum that is not the one the step counts above were read from."""
    problems: list[str] = []
    if EXPECTED_THRESHOLDS not in text:
        problems.append(f"thresholds line is not {EXPECTED_THRESHOLDS.strip()!r}")
    if text.count(STAGE_HEADER) != EXPECTED_STAGES:
        problems.append(f"{text.count(STAGE_HEADER)} stages, expected {EXPECTED_STAGES}")
    for scale in EARLY_VOC_SCALES:
        line = f"    voc_scale: {scale}\n"
        if text.count(line) != 1:
            problems.append(f"{text.count(line)} stages at voc_scale {scale}, expected 1")
    if text.count("    voc_scale: 0.0\n") != 3:
        problems.append(
            f"{text.count('    voc_scale: 0.0')} zero-VOC stages, expected 3 (stages 7, 8, 9)"
        )
    if text.count("immediate_first_frame_probability") != 2:
        problems.append(
            f"{text.count('immediate_first_frame_probability')} stages name "
            "immediate_first_frame_probability, expected 2 (stages 8 and 9)"
        )
    if problems:
        raise SystemExit(
            f"FATAL: {where} is not the curriculum this patch was derived against:\n"
            + "".join(f"  - {problem}\n" for problem in problems)
            + "  The stage-to-step mapping above would be wrong. Re-derive the patch against the\n"
            "  installed config rather than moving stages to step counts that mean nothing."
        )


def verify_patched(text: str, where: str) -> None:
    """The change must be exactly the two probabilities, with everything else where it was."""
    problems: list[str] = []
    if EXPECTED_THRESHOLDS not in text:
        problems.append("the thresholds line was disturbed")
    if text.count("immediate_first_frame_probability: 0.25\n") != 1:
        problems.append("expected exactly one stage at 0.25 (stage 7)")
    if text.count("immediate_first_frame_probability: 0.50\n") != 2:
        problems.append("expected exactly two stages at 0.50 (stages 8 and 9)")
    for scale in EARLY_VOC_SCALES:
        if text.count(f"    voc_scale: {scale}\n") != 1:
            problems.append(f"stage at voc_scale {scale} was disturbed")
    if problems:
        raise SystemExit(
            f"FATAL: {where}: the patched curriculum is not what was intended:\n"
            + "".join(f"  - {problem}\n" for problem in problems)
        )


def plan_file(path: Path) -> tuple[str, str | None]:
    """(report, new text) for the curriculum. None means it already carries the change."""
    if not path.is_file():
        raise SystemExit(f"FATAL: {path} does not exist")
    text = path.read_text()
    if MARKER in text:
        return f"already patched: {path}", None
    verify_structure(text, str(path))
    count = text.count(OLD)
    if count != 1:
        raise SystemExit(
            f"FATAL: {path}: the zero-VOC stage pair occurs {count} times, expected exactly 1.\n"
            "  The curriculum has changed shape; re-derive the patch rather than letting it apply\n"
            "  somewhere unintended."
        )
    patched = text.replace(OLD, NEW, 1)
    verify_patched(patched, str(path))
    return f"patched: {path}", patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="flash_chord source root, e.g. /workspace/src/flash_chord"
    )
    parser.add_argument(
        "--check", action="store_true", help="report what would change, write nothing"
    )
    args = parser.parse_args(argv)

    path = args.root / "configs/curriculum" / RECIPE
    message, text = plan_file(path)
    if text is not None and not args.check:
        path.write_text(text)
    elif text is not None:
        message = message.replace("patched:", "would patch:")

    print(message)
    print(
        "start curriculum: immediate_first_frame_probability 0.25 from 209,999,872 "
        "and 0.50 from 288,749,568; stages 1-6 unchanged"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
