"""Say which reference each clip of a retargeted tree actually carries, per clip.

A kit score is computed in two different roots: the rollout is converted against the reference the
policy *tracked*, and scored against the released truth. Get the first one wrong and nothing shouts
-- on B3 it moved MPPE by 0.36 cm and only `rollout-to-npz`'s own frame check caught it.

Until now the staging pipeline keyed that convert root by SOURCE, which was fine while a source was
either "the cleaned dataset" or "the release". interp's finger tree broke that assumption: it is
one tree whose 14 Dex3 finger columns were solved on top of the cleaned retarget for 29 clips and
on top of the released retarget for the clip the cleaner flagged. One tree, two references, so the
convert root is a property of the clip and not of the source.

The handover says as much in prose. Prose is not what a $100 run should be staged on, and the
object trajectory is right there in both trees, so this command measures it instead: for each clip
it compares the candidate tree's object columns against every source whose scoring reference is
known, and the clip inherits the reference of the tree it reproduces. A clip that matches nothing
is fatal -- that is the "fingers were solved against the noisy object" failure, which would put the
fingers where the noisy object was and score against a reference the policy never saw.

Equality is exact by default: a finger solve copies the object columns through untouched, so any
delta at all means the trajectory was re-solved and the match is not safe to assume.

    v2d match-object-reference TREE --map tier2_episode_map.txt \\
        --against clean=/s/t2_clean/tier2 --against raw=/s/t2_raw/tier2 \\
        --ref clean=/home/.../bridge_v1 -o matched.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# The object trajectory, and nothing else: this is what the evaluator's reference_* columns are
# built from and therefore what the convert root has to agree with.
OBJECT_COLUMNS = (
    "object_root_position",
    "object_root_axis_angle",
    "object_body_position",
    "object_body_wxyz",
)


def clip_parquet(root: Path, sequence: str) -> Path:
    return root / f"whole_body/soma/sequence_id={sequence}/robot_name=g1/data.parquet"


def object_trajectory(path: Path) -> dict[str, np.ndarray]:
    """The object columns of one clip, as float arrays."""
    row = pq.read_table(path, columns=list(OBJECT_COLUMNS)).to_pylist()[0]
    return {name: np.asarray(row[name], dtype=np.float64) for name in OBJECT_COLUMNS}


def trajectory_delta(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> float | None:
    """Largest absolute difference across the object columns, or None if the shapes differ."""
    worst = 0.0
    for name in OBJECT_COLUMNS:
        if a[name].shape != b[name].shape:
            return None
        if a[name].size:
            worst = max(worst, float(np.abs(a[name] - b[name]).max()))
    return worst


def read_map(path: Path) -> dict[int, str]:
    episodes: dict[int, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            episodes[int(parts[0])] = parts[1]
    if not episodes:
        raise SystemExit(f"FATAL: {path} lists no episodes")
    return episodes


def parse_pairs(values: list[str], what: str) -> dict[str, str]:
    pairs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"FATAL: --{what} wants LABEL=VALUE, got {value!r}")
        label, _, rest = value.partition("=")
        pairs[label] = rest
    return pairs


def match(
    tree: Path,
    episodes: dict[int, str],
    against: dict[str, str],
    references: dict[str, str],
    tolerance: float,
) -> dict[str, Any]:
    """Which source each clip of `tree` reproduces, in the order `against` was given."""
    per_episode: dict[str, Any] = {}
    unmatched: list[int] = []
    for episode, sequence in sorted(episodes.items()):
        candidate = clip_parquet(tree, sequence)
        if not candidate.is_file():
            # Not this command's business: the stage reports a missing clip, and a clip the
            # candidate tree does not have simply comes from a later source.
            continue
        mine = object_trajectory(candidate)
        deltas: dict[str, float | None] = {}
        matched: str | None = None
        for label, root in against.items():
            other = clip_parquet(Path(root), sequence)
            if not other.is_file():
                continue
            delta = trajectory_delta(mine, object_trajectory(other))
            deltas[label] = delta
            # First match wins, and `against` is in the stage's own priority order, so a clip whose
            # two candidates are byte-identical (a clip the cleaner declined) resolves the same way
            # every time.
            if matched is None and delta is not None and delta <= tolerance:
                matched = label
        entry: dict[str, Any] = {
            "sequence": sequence,
            "matches": matched,
            "max_object_delta_m": deltas,
        }
        if matched is None:
            unmatched.append(episode)
        elif matched in references:
            entry["ref_dataset"] = references[matched]
        per_episode[str(episode)] = entry

    by_label: dict[str, int] = {}
    for entry in per_episode.values():
        label = entry["matches"] or "UNMATCHED"
        by_label[label] = by_label.get(label, 0) + 1
    return {
        "what": "which source's object trajectory each clip of a retargeted tree carries",
        "tree": str(tree),
        "tolerance_m": tolerance,
        "object_columns": list(OBJECT_COLUMNS),
        "against": against,
        "references": references,
        "per_episode": per_episode,
        "by_label": by_label,
        "unmatched": unmatched,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tree", type=Path, help="the retargeted tree to classify")
    parser.add_argument(
        "--map", type=Path, required=True, help="episode<space>sequence_id per line"
    )
    parser.add_argument(
        "--against",
        action="append",
        default=[],
        metavar="LABEL=ROOT",
        help="a source to compare against, in priority order",
    )
    parser.add_argument(
        "--ref",
        action="append",
        default=[],
        metavar="LABEL=DATASET_ROOT",
        help="that source's scoring dataset root; absent means the release",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="largest accepted absolute difference, metres (default 0.0: exact)",
    )
    parser.add_argument("-o", "--out", type=Path, help="write the result here as JSON")
    args = parser.parse_args(argv)

    against = parse_pairs(args.against, "against")
    if not against:
        raise SystemExit("FATAL: nothing to compare against; pass --against LABEL=ROOT")
    references = parse_pairs(args.ref, "ref")
    unknown = set(references) - set(against)
    if unknown:
        raise SystemExit(f"FATAL: --ref names a source that is not --against: {sorted(unknown)}")

    report = match(args.tree, read_map(args.map), against, references, args.tolerance)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1) + "\n")

    for label, count in sorted(report["by_label"].items()):
        print(f"  {count:2d} clips carry the {label} object trajectory")
    if report["unmatched"]:
        raise SystemExit(
            "FATAL: these clips carry an object trajectory that matches none of the sources: "
            f"{report['unmatched']}\n"
            "  Their rollouts could not be converted against the reference the policy tracked, so a"
            " kit score of them would be meaningless.\n"
            "  If the tree was re-solved against a different object trajectory, stage that"
            " trajectory as its own source with its own --ref-dataset."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
