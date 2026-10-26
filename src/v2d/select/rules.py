"""What the registered checkpoint-selection rule costs, measured against the alternatives.

`v2d pick-checkpoint` is the submission rule and it reads no ground truth: among the checkpoints
that never drop the object it takes the one closest to the noisy Tier 2 input. That tie-break has a
mechanism working against it -- the input is the noise, so a policy that tracks the noise more
literally looks better -- and this command measures what it actually costs, now that the combined
run is producing kit scores for every checkpoint of every clip.

Four rules, all computed from the same per-checkpoint kit scores:

- `z_drop_registered`  what `pick-checkpoint` chose; the rule on the record.
- `latest_drop_free`   the last checkpoint that never drops, i.e. train longer and keep the end.
- `final_checkpoint`   no rule at all; the 350M checkpoint, whatever it does.
- `oracle_reads_truth` the best-scoring checkpoint. It reads the kit score, which is Tier 1 truth,
                       so it is an upper bound and can never be submitted.

The first three read no ground truth, so any of them may be submitted. Which one SHOULD be is not
this command's call: the rule was pre-registered, and the evidence so far points both ways. On the
four stock-recipe clips the registered rule beat the final checkpoint by a wide margin (four-clip
mean kit AUC 0.36635 against 0.28342, see v2d.select.checkpoints). On the first six clips of the
combined run it loses to it. Per-checkpoint AUC is noisy enough for both to be true -- e08 moves
0.137 -> 0.563 between adjacent checkpoints on one evaluation world -- which is exactly why this
writes a file instead of changing the rule.

    v2d compare-selection-rules --results $RUN/results -o results/selection_rule_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# The order they are reported in: cheapest rule first, the upper bound last.
RULES = ("z_drop_registered", "latest_drop_free", "final_checkpoint", "oracle_reads_truth")
READS_GROUND_TRUTH = {rule: rule == "oracle_reads_truth" for rule in RULES}


def clip_record(selection_path: Path) -> dict[str, Any] | None:
    """One clip's per-checkpoint scores and what each rule would pick, or None if unusable.

    A partially scored clip is skipped rather than scored on what it has: a missing score biases
    each rule differently, which is how a selection comparison quietly becomes a comparison of
    which checkpoints happened to finish scoring.
    """
    tag = selection_path.parent.name
    selection = json.loads(selection_path.read_text())["episodes"][tag]
    checkpoints = sorted(selection["checkpoints"], key=lambda c: c["environment_steps"])
    scores: dict[int, float] = {}
    for checkpoint in checkpoints:
        steps = checkpoint["environment_steps"]
        score = selection_path.parent / f"policy_{steps}/score.json"
        if score.is_file():
            scores[steps] = json.loads(score.read_text())["overall"]["add_auc"]
    if not checkpoints or len(scores) != len(checkpoints):
        return None

    drop_free = [c["environment_steps"] for c in checkpoints if not c["dropped"]]
    # With nothing drop-free, pick-checkpoint's own fallback: the checkpoint that held longest.
    longest_hold = max(checkpoints, key=lambda c: c.get("held_frames") or 0)["environment_steps"]
    return {
        "episode": selection["episode"],
        "kit_add_auc_by_checkpoint": {str(k): round(v, 6) for k, v in scores.items()},
        "dropped_by_checkpoint": {
            str(c["environment_steps"]): bool(c["dropped"]) for c in checkpoints
        },
        "picks": {
            "z_drop_registered": selection["selected_environment_steps"],
            "latest_drop_free": drop_free[-1] if drop_free else longest_hold,
            "final_checkpoint": checkpoints[-1]["environment_steps"],
            "oracle_reads_truth": max(scores, key=lambda steps: scores[steps]),
        },
    }


def compare(results: Path) -> dict[str, Any]:
    clips: dict[str, dict[str, Any]] = {}
    skipped: list[str] = []
    for selection_path in sorted(results.glob("e[0-9][0-9]/selection.json")):
        record = clip_record(selection_path)
        if record is None:
            skipped.append(selection_path.parent.name)
        else:
            clips[selection_path.parent.name] = record

    means = {}
    for rule in RULES:
        if clips:
            means[rule] = round(
                sum(
                    clip["kit_add_auc_by_checkpoint"][str(clip["picks"][rule])]
                    for clip in clips.values()
                )
                / len(clips),
                6,
            )
    return {
        "what": "kit AUC of four checkpoint-selection rules on the clips scored so far",
        "results_dir": str(results),
        "clips": sorted(clips),
        "clips_skipped_not_fully_scored": skipped,
        "mean_kit_add_auc": means,
        "reads_ground_truth": READS_GROUND_TRUTH,
        "prior_measurement": {
            "what": "the same comparison on the four stock clips, from v2d.select.checkpoints",
            "z_drop_registered": 0.36635,
            "final_checkpoint": 0.28342,
        },
        "status": (
            "z-drop stays the registered submission rule; the alternatives are reported, not "
            "adopted. Both read no ground truth, so a switch is legal: a decision, not a fix."
        ),
        "per_clip": clips,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="a run's results dir, holding e??/selection.json and e??/policy_*/score.json",
    )
    parser.add_argument("-o", "--out", type=Path, help="write the record here as well as stdout")
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        raise SystemExit(f"FATAL: no results dir at {args.results}")
    record = compare(args.results)
    if not record["clips"]:
        raise SystemExit(
            f"FATAL: {args.results} has no clip with every checkpoint scored"
            + (
                f" (skipped: {', '.join(record['clips_skipped_not_fully_scored'])})"
                if record["clips_skipped_not_fully_scored"]
                else ""
            )
        )

    for rule in RULES:
        truth = " (reads truth, upper bound)" if READS_GROUND_TRUTH[rule] else ""
        print(f"{rule:22s} mean kit AUC {record['mean_kit_add_auc'][rule]:.4f}{truth}")
    print(f"over {len(record['clips'])} clips: {' '.join(record['clips'])}")
    if record["clips_skipped_not_fully_scored"]:
        print(f"skipped, not fully scored: {' '.join(record['clips_skipped_not_fully_scored'])}")
    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
