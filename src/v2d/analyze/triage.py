"""`v2d triage-weak-clips`: why a weak clip is weak, from its own rollout rather than its score.

A low AUC says nothing about what to change. These five buckets do, because each one wants a
different fix, and all of them are computed with the registered selection rule's own constants so
a bucket can never disagree with the checkpoint that was picked:

  fell_back        not one checkpoint was drop-free, so `pick-checkpoint` took the longest hold:
                   the policy never kept the object for a whole clip at any point in training.
  dropped          the selected rollout loses the object -- achieved z under `--drop-z` while the
                   reference stays `--margin` above it, for `--min-frames` straight frames.
  lost_mid_clip    never falls, but passes `--lost-limit` from the reference and stays past it to
                   the end: left behind or shoved away. The default is the evaluator's own 10 cm
                   object position gate.
  never_locks_on   no sustained excursion, yet past the gate for more than a quarter of the clip:
                   it keeps losing the object and keeps recovering, never settling.
  held_drifting    inside the gate on every frame. The AUC went to pose error, which is the only
                   bucket an orientation reward can speak to.

Each clip also carries the frame it goes wrong, the position and orientation error over the frames
*before* that (so they describe the policy while it still has the object), the hand count and
object from the released caption, whether the scene has support primitives, the bridge smoother's
anchor count, the reference's own ceiling score, and what the best of that clip's checkpoints would
have scored. The last two are what stop a fix being proposed where none can help: a clip already at
its reference ceiling has no headroom, and a clip whose oracle gain is the whole deficit was a
selection problem, not a policy one.

    v2d triage-weak-clips --results RUN/results --dataset-root DATA \
        --clips e01,e04,e05 -o results/second_wave_triage.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..config import TIER1
from ..select.checkpoints import DROP_MARGIN_M, DROP_MIN_FRAMES, DROP_Z_M, first_z_drop
from ..supports import read_primitives

#: The evaluator's own object position gate; an excursion past it is what "lost" means.
LOST_LIMIT_M = 0.10
#: Past the gate for more than this fraction of the clip, without ever sustaining it.
NEVER_LOCKS_FRACTION = 0.25


def geodesic_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle in degrees between two ``(T,4)`` xyzw quaternion tracks, double cover aware."""
    dot = np.abs(np.sum(first * second, axis=1)).clip(0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def sustained_to_end(flag: np.ndarray, min_frames: int) -> int | None:
    """First frame of the first run of `min_frames` true values that never comes back."""
    run, start = 0, None
    for frame, value in enumerate(flag):
        if not value:
            run = 0
            continue
        run += 1
        if run == 1:
            start = frame
        if run >= min_frames and bool(flag[frame:].all()):
            return start
    return None


def read_rollout(path: Path) -> dict[str, np.ndarray]:
    """The valid frames of one rollout: achieved and reference pose, distance and angle."""
    table = pq.read_table(path)
    column = {name: np.asarray(table[name]) for name in table.schema.names}
    valid = column.get("environment_valid")
    keep = np.ones(table.num_rows, dtype=bool) if valid is None else valid.astype(bool)
    achieved = np.stack([column[f"achieved_position_{a}"][keep] for a in "xyz"], axis=1)
    reference = np.stack([column[f"reference_position_{a}"][keep] for a in "xyz"], axis=1)
    return {
        "achieved": achieved,
        "reference": reference,
        "distance": np.linalg.norm(achieved - reference, axis=1),
        "angle_deg": geodesic_deg(
            np.stack([column[f"achieved_quaternion_{a}"][keep] for a in "xyzw"], axis=1),
            np.stack([column[f"reference_quaternion_{a}"][keep] for a in "xyzw"], axis=1),
        ),
    }


def classify(
    rollout: dict[str, np.ndarray],
    *,
    drop_free: int,
    drop_z: float,
    margin: float,
    min_frames: int,
    lost_limit: float,
) -> tuple[str, int | None, int | None, int | None]:
    """``(bucket, frame it goes wrong, drop frame, lost frame)``."""
    drop = first_z_drop(
        rollout["achieved"][:, 2],
        rollout["reference"][:, 2],
        drop_z_m=drop_z,
        margin_m=margin,
        min_frames=min_frames,
    )
    lost = sustained_to_end(rollout["distance"] > lost_limit, min_frames)
    past = float((rollout["distance"] > lost_limit).mean())
    if not drop_free:
        bucket = "fell_back"
    elif drop is not None:
        bucket = "dropped"
    elif lost is not None:
        bucket = "lost_mid_clip"
    elif past > NEVER_LOCKS_FRACTION:
        bucket = "never_locks_on"
    else:
        bucket = "held_drifting"
    return bucket, (drop if drop is not None else lost), drop, lost


def hand_count(task: str) -> str:
    """One-hand, two-hand or foot, from the released caption."""
    if re.search(r"\bfoot\b|\bfeet\b", task, re.I):
        return "foot"
    return "two-hand" if re.search(r"both hands|two hands", task, re.I) else "one-hand"


def ceiling_auc(record: dict[str, Any]) -> float:
    """The reference's own score: the cleaned one, unless this clip fell back to raw."""
    return record["add_auc_before"] if record["fall_back_to_raw"] else record["add_auc_after"]


def triage(
    results: Path,
    dataset_root: Path,
    *,
    clips: list[str] | None = None,
    anchors_json: Path | None = None,
    ceiling_json: Path | None = None,
    drop_z: float = DROP_Z_M,
    margin: float = DROP_MARGIN_M,
    min_frames: int = DROP_MIN_FRAMES,
    lost_limit: float = LOST_LIMIT_M,
) -> dict[str, Any]:
    tier1 = dataset_root / "track_2" / TIER1
    metadata = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (tier1 / "meta/episodes_metadata.jsonl").read_text().splitlines()
            if line.strip()
        )
    }
    anchors: dict[str, Any] = {}
    if anchors_json and anchors_json.is_file():
        anchors = json.loads(anchors_json.read_text())["per_clip"]["per_episode"]
    ceiling: dict[str, Any] = {}
    if ceiling_json and ceiling_json.is_file():
        ceiling = json.loads(ceiling_json.read_text())["per_clip"]

    rows = []
    for selection_path in sorted(results.glob("e[0-9][0-9]/selection.json")):
        tag = selection_path.parent.name
        if clips is not None and tag not in clips:
            continue
        episode = int(tag[1:])
        selection = json.loads(selection_path.read_text())["episodes"][tag]
        steps = selection["selected_environment_steps"]
        rollout = read_rollout(selection_path.parent / f"policy_{steps}/objects.parquet")
        frames = len(rollout["distance"])

        bucket, goes_wrong, drop, lost = classify(
            rollout,
            drop_free=len(selection["drop_free_checkpoints"]),
            drop_z=drop_z,
            margin=margin,
            min_frames=min_frames,
            lost_limit=lost_limit,
        )
        # Before it goes wrong: after the object is gone these two numbers mean nothing.
        holding = slice(0, max(goes_wrong if goes_wrong is not None else frames, 1))

        scores = {}
        for checkpoint in selection["checkpoints"]:
            score = selection_path.parent / f"policy_{checkpoint['environment_steps']}/score.json"
            if score.is_file():
                scores[checkpoint["environment_steps"]] = json.loads(score.read_text())["overall"]
        if steps not in scores:
            continue
        selected, best = scores[steps], max(scores.values(), key=lambda s: s["add_auc"])

        task = str(metadata[episode]["task"])
        support = tier1 / metadata[episode]["support_surface"]
        primitives = read_primitives(support) if support.is_file() else []
        anchor = anchors.get(str(episode), {})
        reference = rollout["reference"]
        step_cm = np.linalg.norm(np.diff(reference, axis=0), axis=1) * 100
        rows.append(
            {
                "tag": tag,
                "episode": episode,
                "object": metadata[episode]["object"],
                "hands": hand_count(task),
                "scene": "table" if primitives else "floor",
                "support_primitives": len(primitives),
                "frames": frames,
                "bucket": bucket,
                "goes_wrong_frame": goes_wrong,
                "goes_wrong_fraction": round(goes_wrong / frames, 3) if goes_wrong else None,
                "drop_frame": drop,
                "lost_frame": lost,
                "drop_free_checkpoints": len(selection["drop_free_checkpoints"]),
                "selected_environment_steps": steps,
                "kit_add_auc": round(selected["add_auc"], 4),
                "kit_mppe_cm": round(selected["mppe_cm"], 2),
                "kit_spider_sr": selected["spider_sr"],
                "kit_maniptrans_sr": selected["maniptrans_sr"],
                "oracle_add_auc": round(best["add_auc"], 4),
                "oracle_gain": round(best["add_auc"] - selected["add_auc"], 4),
                "mean_distance_cm_while_holding": round(
                    float(rollout["distance"][holding].mean()) * 100, 2
                ),
                "max_distance_cm": round(float(rollout["distance"].max()) * 100, 2),
                "fraction_of_frames_past_gate": round(
                    float((rollout["distance"] > lost_limit).mean()), 3
                ),
                "mean_angle_deg_while_holding": round(
                    float(rollout["angle_deg"][holding].mean()), 1
                ),
                # A reference that jumps at the failure frame is a staging problem, not a policy
                # one, so the two have to be told apart before anything is proposed.
                "reference_step_cm_at_failure": None
                if goes_wrong is None or goes_wrong >= frames - 1
                else round(
                    float(step_cm[max(goes_wrong - 3, 0) : goes_wrong + 3].max()),
                    3,
                ),
                "reference_step_cm_max": round(float(step_cm.max()), 3),
                "bridge_anchor_segments": anchor.get("anchor_segments"),
                "bridge_anchor_frames": anchor.get("anchor_frames"),
                "bridge_rest_segments": anchor.get("rest_segments"),
                "bridge_status": anchor.get("status", ""),
                "reference_ceiling_add_auc": round(ceiling_auc(ceiling[str(episode)]), 4)
                if str(episode) in ceiling
                else None,
                "task": task,
            }
        )

    buckets: dict[str, list[str]] = {}
    for row in rows:
        buckets.setdefault(row["bucket"], []).append(row["tag"])
    return {
        "what": "failure mode per clip, from the selected checkpoint's own rollout",
        "rule_constants": {
            "drop_z_m": drop_z,
            "margin_m": margin,
            "min_frames": min_frames,
            "lost_limit_m": lost_limit,
            "never_locks_fraction": NEVER_LOCKS_FRACTION,
            "note": "the drop constants are the registered selection rule's own",
        },
        "buckets": {name: sorted(tags) for name, tags in sorted(buckets.items())},
        "clips": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="a run's results dir")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--clips", help="comma-separated tags, e.g. e01,e04; default every clip")
    parser.add_argument("--anchors-json", type=Path, help="results/bridge_anchor_counts.json")
    parser.add_argument(
        "--ceiling-json", type=Path, help="the cleaned tree's ceiling_per_clip.json"
    )
    parser.add_argument("--drop-z", type=float, default=DROP_Z_M, metavar="M")
    parser.add_argument("--margin", type=float, default=DROP_MARGIN_M, metavar="M")
    parser.add_argument("--min-frames", type=int, default=DROP_MIN_FRAMES, metavar="N")
    parser.add_argument("--lost-limit", type=float, default=LOST_LIMIT_M, metavar="M")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    if not args.results.is_dir():
        raise SystemExit(f"FATAL: no results dir at {args.results}")
    record = triage(
        args.results,
        args.dataset_root,
        clips=[c.strip() for c in args.clips.split(",")] if args.clips else None,
        anchors_json=args.anchors_json,
        ceiling_json=args.ceiling_json,
        drop_z=args.drop_z,
        margin=args.margin,
        min_frames=args.min_frames,
        lost_limit=args.lost_limit,
    )
    if not record["clips"]:
        raise SystemExit(f"FATAL: {args.results} has no scored clip matching --clips")

    header = f"{'clip':5s} {'bucket':15s} {'wrong@':>7s} {'AUC':>6s} {'orc+':>6s} {'dist':>6s} "
    print(header + f"{'ang':>6s} {'anch':>5s} {'ceil':>6s} {'scene':6s} hands")
    for row in sorted(record["clips"], key=lambda r: r["kit_add_auc"]):
        ceiling = row["reference_ceiling_add_auc"]
        print(
            f"{row['tag']:5s} {row['bucket']:15s} "
            f"{(row['goes_wrong_frame'] if row['goes_wrong_frame'] is not None else '-'):>7} "
            f"{row['kit_add_auc']:6.3f} {row['oracle_gain']:+6.3f} "
            f"{row['mean_distance_cm_while_holding']:6.1f} "
            f"{row['mean_angle_deg_while_holding']:6.1f} "
            f"{row['bridge_anchor_segments']!s:>5} "
            f"{(f'{ceiling:6.3f}' if ceiling is not None else '     -')} "
            f"{row['scene']:6s} {row['hands']}"
        )
    for name, tags in record["buckets"].items():
        print(f"{name:15s} {len(tags):2d}  {' '.join(tags)}")
    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
