"""Pick one checkpoint per episode, using no ground truth.

A 350M-step run saves seven checkpoints and the last one is not the best: across our four clips the
final checkpoint scores a four-clip mean kit AUC of 0.28342, while picking per clip lifts it to
0.36635 -- the largest free win on the board, and it applies to all 30 clips at submission time.

The catch is that choosing by score would be choosing on the answer. The only reference available
at submission time is the noisy Tier 2 input the policy trained against, which the evaluator writes
into `objects.parquet` as the `reference_*` columns, so that is the only thing this command reads.
Tier 1 is never opened; there is no argument for it. What survives is a rule with two parts:

1. **Reject a checkpoint that loses the object.** Two definitions are available, because they
   disagree and the disagreement is on the record:
   - `divergence` (default): the achieved object must stay within `--divergence` metres of its
     reference at *every* frame. 0.10 m is the evaluator's own `object_position_threshold_m`, so
     this rejects exactly the checkpoints that have stopped tracking by the evaluator's yardstick.
   - `z-drop`: the achieved object sits `--margin` below its reference *and* under `--drop-z` for
     `--min-frames` consecutive frames. This is the rule registered in docs/AB4_DESIGN.md; the
     margin and the run length are there because the bare `z < 0.15 m` form fires on a 2-frame,
     7 mm lift lag and threw away 6 of e24's 7 checkpoints for nothing.
2. **Among the survivors take the smallest mean distance to that same input reference.** If none
   survive, take the one that held on longest and break ties the same way, because a clip with no
   clean checkpoint still has to submit something.

The chosen `checkpoint_sha256` is emitted per episode. The packer stamps provenance per episode
rather than per submission, so different episodes may legitimately come from different checkpoints
-- and with this rule they do.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# The evaluator's own per-frame object position threshold, and the one it reports metrics against.
EVALUATOR_POSITION_THRESHOLD_M = 0.10
# Half that threshold, and 0.2 s at the 50 Hz control rate: see docs/AB4_DESIGN.md.
DROP_Z_M = 0.15
DROP_MARGIN_M = 0.05
DROP_MIN_FRAMES = 10

CHECKPOINT_DIR = re.compile(r"^policy_(\d+)$")
EPISODE_LABEL = re.compile(r"^e?(\d+)$")
REPORTED_METRICS = ("add_auc", "mppe_cm", "chord_sr", "object_orientation_error_deg")


def episode_of(label: str) -> int:
    """`e03`, `e3` and `3` all name episode 3; anything else is a typo worth stopping for."""
    found = EPISODE_LABEL.match(label)
    if not found:
        raise SystemExit(f"--cell label {label!r} is not an episode; write e03, e3 or 3")
    return int(found.group(1))


def first_divergence(distance_m: np.ndarray, limit_m: float) -> int | None:
    """First frame at which the object is further than `limit_m` from its reference."""
    over = np.flatnonzero(distance_m > limit_m)
    return int(over[0]) if over.size else None


def first_z_drop(
    achieved_z: np.ndarray,
    reference_z: np.ndarray,
    drop_z_m: float = DROP_Z_M,
    margin_m: float = DROP_MARGIN_M,
    min_frames: int = DROP_MIN_FRAMES,
) -> int | None:
    """First frame of the first run of `min_frames` frames that is a real loss of the object."""
    under = (achieved_z < drop_z_m) & (reference_z >= drop_z_m)
    under &= (reference_z - achieved_z) >= margin_m
    run = 0
    for frame, is_under in enumerate(under):
        run = run + 1 if is_under else 0
        if run >= min_frames:
            return frame - min_frames + 1
    return None


def read_checkpoint(directory: Path) -> dict[str, Any]:
    """One checkpoint's trajectory and evaluator metrics, with nothing but its own reference."""
    table = pq.read_table(directory / "objects.parquet")
    columns = {name: np.asarray(table[name]) for name in table.schema.names}
    valid = columns.get("environment_valid")
    keep = np.ones(table.num_rows, dtype=bool) if valid is None else valid.astype(bool)
    achieved = np.stack([columns[f"achieved_position_{a}"][keep] for a in "xyz"], axis=1)
    reference = np.stack([columns[f"reference_position_{a}"][keep] for a in "xyz"], axis=1)
    distance = np.linalg.norm(achieved - reference, axis=1)

    metrics = json.loads((directory / "metrics.json").read_text())
    row: dict[str, Any] = {
        "environment_steps": int(metrics["environment_steps"]),
        "frames": int(keep.sum()),
        "frames_invalid": int((~keep).sum()),
        "mean_distance_to_input_reference_cm": round(float(distance.mean()) * 100, 4),
        "max_distance_to_input_reference_cm": round(float(distance.max()) * 100, 4),
        "checkpoint_sha256": metrics.get("checkpoint_sha256"),
        "_distance_m": distance,
        "_achieved_z": achieved[:, 2],
        "_reference_z": reference[:, 2],
    }
    for name in REPORTED_METRICS:
        if name in metrics.get("metrics", {}):
            row[f"evaluator_{name}"] = metrics["metrics"][name]

    # The sha is what the submission will claim about this policy, so check it against the file we
    # actually hold rather than trusting the evaluator's record of a file on a box that is gone.
    policy = directory / "policy.safetensors"
    if policy.is_file():
        digest = hashlib.sha256(policy.read_bytes()).hexdigest()
        row["checkpoint_sha256_of_local_policy"] = digest
        if row["checkpoint_sha256"] and digest != row["checkpoint_sha256"]:
            raise SystemExit(
                f"FATAL: {policy} hashes to {digest}, but its metrics.json claims "
                f"{row['checkpoint_sha256']}. One of the two is from a different run, and a "
                "submission must not claim a provenance it cannot reproduce."
            )
    return row


def read_cell(directory: Path) -> list[dict[str, Any]]:
    """Every evaluated checkpoint in a cell, oldest first."""
    found = []
    for child in sorted(directory.iterdir()):
        match = CHECKPOINT_DIR.match(child.name)
        if match and (child / "objects.parquet").is_file() and (child / "metrics.json").is_file():
            found.append(child)
    if not found:
        raise SystemExit(
            f"no evaluated checkpoints under {directory}: expected policy_<steps>/ holding "
            "objects.parquet and metrics.json"
        )
    rows = [read_checkpoint(child) for child in found]
    return sorted(rows, key=lambda row: row["environment_steps"])


def mark_drops(rows: list[dict[str, Any]], criterion: str, **limits: float) -> None:
    """Annotate each checkpoint with whether, and where, it lost the object."""
    for row in rows:
        if criterion == "divergence":
            frame = first_divergence(row["_distance_m"], limits["divergence_m"])
        else:
            frame = first_z_drop(
                row["_achieved_z"],
                row["_reference_z"],
                drop_z_m=limits["drop_z_m"],
                margin_m=limits["margin_m"],
                min_frames=int(limits["min_frames"]),
            )
        row["dropped"] = frame is not None
        row["first_violation_frame"] = frame
        row["held_frames"] = row["frames"] if frame is None else frame


def select(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    """The rule: no drop first, then closest to the input reference. Deterministic on ties."""
    survivors = [row for row in rows if not row["dropped"]]
    if survivors:
        chosen = min(
            survivors,
            key=lambda row: (row["mean_distance_to_input_reference_cm"], row["environment_steps"]),
        )
        return chosen, "no drop, then the smallest mean distance to the input reference"
    longest = max(row["held_frames"] for row in rows)
    chosen = min(
        (row for row in rows if row["held_frames"] == longest),
        key=lambda row: (row["mean_distance_to_input_reference_cm"], row["environment_steps"]),
    )
    return chosen, (
        "every checkpoint drops; the one that held longest "
        f"({longest} frames), then the smallest mean distance to the input reference"
    )


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _mean(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return round(sum(present) / len(present), 6) if len(present) == len(values) else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cell",
        action="append",
        required=True,
        metavar="LABEL=DIR",
        help="an episode and the directory holding its policy_<steps>/ rollouts, repeatable",
    )
    parser.add_argument(
        "--criterion",
        choices=("divergence", "z-drop"),
        default="divergence",
        help="how a lost object is defined (default: divergence)",
    )
    parser.add_argument(
        "--divergence",
        type=float,
        default=EVALUATOR_POSITION_THRESHOLD_M,
        metavar="M",
        help=f"divergence limit in metres (default {EVALUATOR_POSITION_THRESHOLD_M}, the "
        "evaluator's own object position threshold)",
    )
    parser.add_argument("--drop-z", type=float, default=DROP_Z_M, metavar="M")
    parser.add_argument("--margin", type=float, default=DROP_MARGIN_M, metavar="M")
    parser.add_argument("--min-frames", type=int, default=DROP_MIN_FRAMES, metavar="N")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cells: dict[str, Path] = {}
    for item in args.cell:
        if "=" not in item:
            raise SystemExit(f"--cell wants LABEL=DIR, got {item!r}")
        label, _, directory = item.partition("=")
        episode_of(label)
        cells[label] = Path(directory)

    episodes: dict[str, Any] = {}
    for label, directory in cells.items():
        rows = read_cell(directory)
        mark_drops(
            rows,
            args.criterion,
            divergence_m=args.divergence,
            drop_z_m=args.drop_z,
            margin_m=args.margin,
            min_frames=args.min_frames,
        )
        chosen, reason = select(rows)
        episodes[label] = {
            "episode": episode_of(label),
            "cell": str(directory),
            "checkpoints": [_public(row) for row in rows],
            "drop_free_checkpoints": [
                row["environment_steps"] for row in rows if not row["dropped"]
            ],
            "selected_environment_steps": chosen["environment_steps"],
            "selected_checkpoint_sha256": chosen["checkpoint_sha256"],
            "selection_reason": reason,
        }

    final = [row["checkpoints"][-1] for row in episodes.values()]
    picked = [
        next(
            row
            for row in episode["checkpoints"]
            if row["environment_steps"] == episode["selected_environment_steps"]
        )
        for episode in episodes.values()
    ]
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    record = {
        "what": "one checkpoint per episode, chosen without reading any ground truth",
        "git_sha": sha,
        "rule": {
            "criterion": args.criterion,
            "divergence_limit_m": args.divergence,
            "drop_z_m": args.drop_z,
            "margin_m": args.margin,
            "min_frames": args.min_frames,
            "tie_break": "smallest mean distance to the input reference, then fewest steps",
            "fallback": "if every checkpoint drops, the one that held longest",
            "ground_truth_read": False,
            "reference_used": "the reference_* columns the evaluator wrote, i.e. the noisy input "
            "the policy trained against",
        },
        "episodes": episodes,
        # What the packer needs: provenance is stamped per episode, so each episode may carry its
        # own checkpoint.
        "per_episode_checkpoint_sha256": {
            str(episode["episode"]): episode["selected_checkpoint_sha256"]
            for episode in episodes.values()
        },
        "summary": {
            "episodes": len(episodes),
            "mean_evaluator_add_auc_final": _mean([row.get("evaluator_add_auc") for row in final]),
            "mean_evaluator_add_auc_selected": _mean(
                [row.get("evaluator_add_auc") for row in picked]
            ),
            "mean_evaluator_mppe_cm_final": _mean([row.get("evaluator_mppe_cm") for row in final]),
            "mean_evaluator_mppe_cm_selected": _mean(
                [row.get("evaluator_mppe_cm") for row in picked]
            ),
            "episodes_dropping_at_the_final_checkpoint": sum(row["dropped"] for row in final),
            "episodes_with_no_drop_free_checkpoint": sum(
                not episode["drop_free_checkpoints"] for episode in episodes.values()
            ),
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    for label, episode in episodes.items():
        row = next(
            item
            for item in episode["checkpoints"]
            if item["environment_steps"] == episode["selected_environment_steps"]
        )
        print(
            f"{label}: {len(episode['checkpoints'])} checkpoints, "
            f"{len(episode['drop_free_checkpoints'])} drop-free -> "
            f"{episode['selected_environment_steps'] // 1_000_000}M "
            f"(auc {row.get('evaluator_add_auc')}, mppe {row.get('evaluator_mppe_cm')} cm, "
            f"sha {str(episode['selected_checkpoint_sha256'])[:12]})"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
