"""`v2d deep-dive`: a metrics-level reading of a finished run, per clip and per bucket.

Six questions, all answerable for free from rollouts already paid for, and each one chosen because
it changes where the next GPU dollar goes:

A. **AUC decomposition.** ADD is `mean_v |(R_a - R_r)v + (p_a - p_r)|`, so position and orientation
   enter through the same mesh and a single AUC cannot say which lost it. Substituting the clean
   truth into one channel and rescoring with the kit isolates the other exactly -- the kit's
   `_ALIGN_INITIAL_POSE` is False, so no start-frame warp mixes them back together.
B. **Where the error lives in time.** First frame past the 10 cm gate, the fraction of frames inside
   5/10/20 cm, and the error curve as max, mean before the loss and mean after it. A clip that is
   good for 90% of its frames and a clip that is bad throughout can score the same and want
   opposite fixes.
C. **SP-SR and MP-SR margins.** Both are thresholds on the *time-averaged* error, not per-frame
   success, so each clip either passes or fails and the useful number is how far it is from the
   line: 10 cm / 0.5 rad for SP-SR, 3 cm / 30 deg for MP-SR. A clip 4 mm outside is a different
   proposition from one 40 cm outside.
D. **The AUC-vs-checkpoint curve.** Monotone, peaked or collapsing, and whether the registered
   selection landed on the per-clip best. Collapse means training past the peak hurts; a selection
   that misses the peak is a cheaper problem than one that does not exist.
E. **Hand-object distance.** Palm centre to the nearest point of the object's own AUC vertex cloud,
   for the reference hand against both the reference and the achieved object. This is what turns
   the strip review's "pantomime" into a number: frames where the reference has the object in hand
   and the hand is moving, while the achieved object is somewhere else entirely.
F. **Correlations across the clips** between score and object size, hand count, anchor count,
   reference ceiling and first-loss frame, so a per-clip story is not mistaken for a pattern.

One caveat governs E and is stated in the output: the boxes exported object trajectories only, so
the *achieved* palm was never recorded. E uses the reference palm, which measures "the hand went
where it should and the object did not follow" under the assumption that body tracking is good.
That assumption is the recipe's own -- body weight 5 against object weight 1 -- and the strip
review reports the body performing the motion, but it is an assumption and not a measurement.

    v2d deep-dive --results RUN/results --dataset-root DATA --kit-root KIT \
        --provenance results/t2_combined_provenance.json --work-dir SCRATCH \
        -o results/metrics_deep_dive.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..config import TIER1
from ..dataset import object_pose

#: The kit's own thresholds (metric_code/track_2/AUC.py).
SPIDER_POSITION_M = 0.1
SPIDER_ORIENTATION_RAD = 0.5
MANIPTRANS_POSITION_M = 0.03
MANIPTRANS_ORIENTATION_DEG = 30.0
ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)

#: Gates for part B, in metres. 10 cm is the evaluator's object position threshold.
BANDS_M = (0.05, 0.10, 0.20)
LOST_LIMIT_M = 0.10
LOST_MIN_FRAMES = 10

#: Part E. The achieved object counts as out of the hand when it is this much further from the palm
#: than the reference object is at the same frame, and a palm moving faster than this is treated as
#: still performing the motion.
#:
#: This is a *relative* gate on purpose. `ee_pose_w` is the end-effector link origin, not a
#: fingertip, so even in the reference the palm sits a mean 5.0 cm from the object's surface while
#: the retargeter reports contact (e03: p90 9.4 cm). Any absolute "in hand" distance therefore
#: misclassifies about half the genuinely held frames. Measuring the *excess* over the reference at
#: the same frame cancels the offset, and reuses the evaluator's own 10 cm object gate rather than
#: inventing a constant.
OUT_OF_HAND_EXCESS_M = 0.10
PALM_MOVING_M_PER_S = 0.05


def geodesic_rad(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Angle between two ``(T,4)`` xyzw quaternion tracks, double cover aware, as the kit does."""
    dot = np.abs(np.sum(first * second, axis=1)).clip(0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """``(T,3,3)`` rotation matrices from ``(T,4)`` xyzw quaternions."""
    x, y, z, w = (quaternion[..., i] for i in range(4))
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def add_per_frame(
    achieved: np.ndarray, reference: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    """The kit's ADD, per frame: mean over the object's vertex cloud of the point displacement."""
    delta_rotation = rotation_matrix(achieved[:, 3:]) - rotation_matrix(reference[:, 3:])
    offsets = np.einsum("tij,vj->tvi", delta_rotation, vertices)
    offsets += (achieved[:, :3] - reference[:, :3])[:, None, :]
    return np.linalg.norm(offsets, axis=-1).mean(axis=-1)


def add_auc(add: np.ndarray) -> float:
    """Area under the fraction-below-threshold curve, on the kit's normalised axis."""
    axis = np.linspace(0.0, 1.0, ADD_THRESHOLDS_M.size)
    fractions = np.array([float(np.mean(add < limit)) for limit in ADD_THRESHOLDS_M])
    return float(np.trapezoid(fractions, x=axis))


def first_past(flag: np.ndarray, min_frames: int) -> int | None:
    """First frame of the first run of `min_frames` consecutive true values."""
    run = 0
    for frame, value in enumerate(flag):
        run = run + 1 if value else 0
        if run >= min_frames:
            return frame - min_frames + 1
    return None


#: A curve whose whole range is inside this is flat, not peaked: argmax of noise is not a peak.
FLAT_RANGE = 0.02


def curve_shape(steps: list[int], aucs: list[float]) -> str:
    """Flat, monotone, peaked or collapsing, from where the best checkpoint sits."""
    if len(aucs) < 3:
        return "too few checkpoints"
    if max(aucs) - min(aucs) <= FLAT_RANGE:
        # Training moved the score by less than the noise, so there is no shape to read.
        return "flat"
    best = int(np.argmax(aucs))
    end_drop = aucs[best] - aucs[-1]
    if best == len(aucs) - 1:
        return "monotone"
    # A peak that the end gives back more than a tenth of is a collapse, not a plateau.
    return "collapsing" if end_drop > 0.1 * max(aucs[best], 1e-9) + 0.02 else "peaked"


def read_rollout(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """``(achieved, reference)`` ``(T,7)`` xyzw poses over the valid frames of one rollout."""
    table = pq.read_table(path)
    column = {name: np.asarray(table[name]) for name in table.schema.names}
    valid = column.get("environment_valid")
    keep = np.ones(table.num_rows, dtype=bool) if valid is None else valid.astype(bool)
    pose = []
    for role in ("achieved", "reference"):
        pose.append(
            np.concatenate(
                [
                    np.stack([column[f"{role}_position_{a}"][keep] for a in "xyz"], axis=1),
                    np.stack([column[f"{role}_quaternion_{a}"][keep] for a in "xyzw"], axis=1),
                ],
                axis=1,
            )
        )
    return pose[0], pose[1]


def staged_hands(parquet: Path) -> dict[str, np.ndarray] | None:
    """Reference palm poses, per-hand contact flags and the capture rate, or None if absent."""
    if not parquet.is_file():
        return None
    table = pq.read_table(parquet, columns=["ee_pose_w", "hand_contact_active", "fps"])
    row = {name: table[name][0].as_py() for name in table.schema.names}
    palms = np.asarray(row["ee_pose_w"], dtype=float)
    if palms.ndim != 3 or palms.shape[1] == 0:
        return None
    return {
        "palms": palms,  # (T, hands, 7)
        "contact": np.asarray(row["hand_contact_active"], dtype=float),  # (hands, T)
        "fps": float(row["fps"]),
    }


def surface_distance(palm: np.ndarray, pose: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """``(T,)`` distance from a palm centre to the nearest object vertex, per frame.

    The object's own AUC vertex cloud, so "surface" means the same points the score is computed
    over rather than a bounding sphere, which on a bat or a cane would be wrong by its half-length.
    """
    world = np.einsum("tij,vj->tvi", rotation_matrix(pose[:, 3:]), vertices) + pose[:, None, :3]
    return np.linalg.norm(world - palm[:, None, :], axis=-1).min(axis=-1)


def write_npz(path: Path, episode: int, frames: np.ndarray, pose: np.ndarray) -> None:
    np.savez(
        path,
        episode_index=episode,
        coordinate_frame="released_track2_world",
        frame_index=frames.astype(np.int64),
        pose_xyzw=pose.reshape(len(pose), 1, 7),
    )


def hand_object_summary(
    staged: dict[str, np.ndarray],
    achieved_sim: np.ndarray,
    reference_sim: np.ndarray,
    *,
    vertices: np.ndarray,
) -> dict[str, Any]:
    """Part E: pantomime as a number, in the SIMULATOR frame.

    The frame matters and is the one thing here that can silently go wrong. The palms come from
    the staged motion, whose object track is bit-identical to the rollout's `reference_*` columns;
    the released-frame poses have been through rollout-to-npz's rigid conversion. Measuring a palm
    against a converted object mixes two frames and reports ~200 cm on every clip, including ones
    whose hand is demonstrably on the object, so both object tracks here are the rollout's own.

    "In hand" is relative, not absolute: `ee_pose_w` is an end-effector link origin rather than a
    fingertip, so even the reference sits a few centimetres off the surface while holding. The
    excess over the reference at the same frame cancels that offset.
    """
    usable = min(len(achieved_sim), staged["palms"].shape[0])
    achieved_sim, reference_sim = achieved_sim[:usable], reference_sim[:usable]
    palms = staged["palms"][:usable]
    contact = staged["contact"][:, :usable]

    nearest = {}
    for label, pose in (("achieved", achieved_sim), ("reference", reference_sim)):
        nearest[label] = np.stack(
            [surface_distance(palms[:, hand, :3], pose, vertices)
             for hand in range(palms.shape[1])],
            axis=0,
        ).min(axis=0)

    speed = np.zeros(usable)
    speed[1:] = np.linalg.norm(np.diff(palms[:, :, :3], axis=0), axis=-1).max(axis=-1) * staged[
        "fps"
    ]
    should_hold = contact.max(axis=0) > 0.5
    excess = nearest["achieved"] - nearest["reference"]
    achieved_in_hand = excess <= OUT_OF_HAND_EXCESS_M
    pantomime = should_hold & ~achieved_in_hand & (speed > PALM_MOVING_M_PER_S)
    held = should_hold if should_hold.any() else np.ones(usable, dtype=bool)
    return {
        "frame": "simulator; palms and both object tracks come from the same staged motion",
        "frames_reference_says_held": int(should_hold.sum()),
        "fraction_reference_says_held": round(float(should_hold.mean()), 3),
        "fraction_achieved_in_hand_while_reference_holds": round(
            float(achieved_in_hand[held].mean()), 3
        ),
        "pantomime_frames": int(pantomime.sum()),
        "pantomime_fraction": round(float(pantomime.mean()), 3),
        "pantomime_fraction_of_held": round(float(pantomime[held].mean()), 3),
        "palm_to_object_excess_cm_while_held": {
            "mean": round(float(excess[held].mean()) * 100, 1),
            "max": round(float(excess[held].max()) * 100, 1),
        },
        # The calibration the relative gate rests on: how far the palm sits from the object it IS
        # holding. If this were ~0 an absolute gate would have done, and it is not.
        "reference_palm_to_object_cm_while_held": {
            "mean": round(float(nearest["reference"][held].mean()) * 100, 1),
            "p90": round(float(np.percentile(nearest["reference"][held], 90)) * 100, 1),
        },
        "palm_to_achieved_object_cm_while_held": round(
            float(nearest["achieved"][held].mean()) * 100, 1
        ),
    }


def released_series(
    trajectory: Path,
    *,
    episode: int,
    convert_root: Path,
    dataset_root: Path,
    work: Path,
    vertices: np.ndarray,
) -> dict[str, np.ndarray]:
    """One rollout's per-frame error against the clean truth, in the released world frame.

    Shared by the metrics table and the figures so a plotted curve and a tabulated number can
    never come from two different computations.
    """
    work.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(
        [
            sys.executable, "-I", "-m", "v2d.cli", "rollout-to-npz", str(trajectory),
            "--dataset-root", str(convert_root),
            "--episode", str(episode),
            "--tier", "tier2",
            "--npz-out", str(work),
        ],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    if done.returncode != 0:
        raise RuntimeError(f"rollout-to-npz failed on episode {episode}: {done.stderr[-400:]}")
    with np.load(work / f"episode_{episode:06d}.npz") as archive:
        frames = np.asarray(archive["frame_index"], dtype=np.int64)
        released = np.asarray(archive["pose_xyzw"], dtype=np.float64)[:, 0]
    truth = object_pose(dataset_root, "tier1", episode)[0][frames]
    return {
        "frames": frames,
        "released": released,
        "truth": truth,
        "position_cm": np.linalg.norm(released[:, :3] - truth[:, :3], axis=1) * 100,
        "orientation_deg": np.degrees(geodesic_rad(released[:, 3:], truth[:, 3:])),
        "add_cm": add_per_frame(released, truth, vertices) * 100,
    }


def kit_score(kit_root: Path, dataset_root: Path, npz_dir: Path, episode: int) -> dict[str, float]:
    """`v2d score` on one episode of one npz directory. The organisers' metric code, not a copy."""
    out = npz_dir / "score.json"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "v2d.cli",
            "score",
            "--kit-root",
            str(kit_root),
            "--dataset-root",
            str(dataset_root),
            "--tier",
            "tier2",
            "--pred",
            f"npz:{npz_dir}",
            "--episodes",
            str(episode),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    if done.returncode != 0 or not out.is_file():
        raise RuntimeError(f"kit score failed on episode {episode}: {done.stderr[-400:]}")
    return json.loads(out.read_text())["overall"]


def spearman(first: list[float], second: list[float]) -> tuple[float, int]:
    """Rank correlation and n, over the pairs where both values exist. No SciPy on this box."""
    pairs = [(a, b) for a, b in zip(first, second, strict=True) if a is not None and b is not None]
    if len(pairs) < 3:
        return float("nan"), len(pairs)
    left = np.argsort(np.argsort([p[0] for p in pairs])).astype(float)
    right = np.argsort(np.argsort([p[1] for p in pairs])).astype(float)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt((left**2).sum() * (right**2).sum()))
    return (float((left * right).sum() / denominator) if denominator else float("nan")), len(pairs)


def decompose(
    *,
    episode: int,
    achieved: np.ndarray,
    truth: np.ndarray,
    frames: np.ndarray,
    kit_root: Path,
    dataset_root: Path,
    work: Path,
) -> dict[str, Any]:
    """Part A: rescore with one channel replaced by truth, so the other one is isolated.

    `achieved` is already in the released world frame, which is the frame `truth` is in, so the
    substitution is a column swap and nothing is re-fitted.
    """
    out: dict[str, Any] = {}
    for label, pose in (
        # Truth orientation in: whatever AUC is still lost is lost to position.
        ("position_only", np.concatenate([achieved[:, :3], truth[:, 3:]], axis=1)),
        # Truth position in: whatever is still lost is lost to orientation.
        ("orientation_only", np.concatenate([truth[:, :3], achieved[:, 3:]], axis=1)),
    ):
        directory = work / f"e{episode:02d}_{label}"
        directory.mkdir(parents=True, exist_ok=True)
        write_npz(directory / f"episode_{episode:06d}.npz", episode, frames, pose)
        scored = kit_score(kit_root, dataset_root, directory, episode)
        out[label] = {
            "kit_add_auc": round(scored["add_auc"], 6),
            "kit_mppe_cm": round(scored["mppe_cm"], 4),
            "kit_spider_sr": scored["spider_sr"],
            "kit_maniptrans_sr": scored["maniptrans_sr"],
        }
    return out


def clip_metrics(
    tag: str,
    *,
    results: Path,
    dataset_root: Path,
    convert_root: Path,
    kit_root: Path,
    vertices: np.ndarray,
    staged: dict[str, np.ndarray] | None,
    work: Path,
) -> dict[str, Any]:
    """Everything parts A-E say about one clip."""
    episode = int(tag[1:])
    selection = json.loads((results / tag / "selection.json").read_text())["episodes"][tag]
    steps = selection["selected_environment_steps"]

    # Part D first: it needs only the scores that already exist.
    curve = []
    for checkpoint in sorted(selection["checkpoints"], key=lambda c: c["environment_steps"]):
        score = results / tag / f"policy_{checkpoint['environment_steps']}/score.json"
        if score.is_file():
            curve.append(
                (
                    checkpoint["environment_steps"],
                    float(json.loads(score.read_text())["overall"]["add_auc"]),
                )
            )
    if not curve:
        raise RuntimeError(f"{tag}: no checkpoint is scored")
    curve_steps = [s for s, _ in curve]
    curve_aucs = [a for _, a in curve]
    best_index = int(np.argmax(curve_aucs))

    # The selected rollout, converted into the released frame exactly as the scoring pass did.
    npz_dir = work / f"e{episode:02d}_base"
    npz_dir.mkdir(parents=True, exist_ok=True)
    converted = subprocess.run(
        [
            sys.executable, "-I", "-m", "v2d.cli", "rollout-to-npz",
            str(results / tag / f"policy_{steps}/objects.parquet"),
            "--dataset-root", str(convert_root),
            "--episode", str(episode),
            "--tier", "tier2",
            "--npz-out", str(npz_dir),
        ],
        capture_output=True, text=True, check=False,
        cwd=str(Path(__file__).resolve().parents[2]),
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
    )
    if converted.returncode != 0:
        raise RuntimeError(f"{tag}: rollout-to-npz failed: {converted.stderr[-400:]}")
    with np.load(npz_dir / f"episode_{episode:06d}.npz") as archive:
        frames = np.asarray(archive["frame_index"], dtype=np.int64)
        released = np.asarray(archive["pose_xyzw"], dtype=np.float64)[:, 0]

    truth_full, _visible = object_pose(dataset_root, "tier1", episode)
    truth = truth_full[frames]

    # Part B, against the truth the score is computed over, not against the noisy input.
    add = add_per_frame(released, truth, vertices)
    position = np.linalg.norm(released[:, :3] - truth[:, :3], axis=1)
    orientation = geodesic_rad(released[:, 3:], truth[:, 3:])
    lost = first_past(position > LOST_LIMIT_M, LOST_MIN_FRAMES)
    before = slice(0, lost) if lost else slice(0, len(position))
    after = slice(lost, len(position)) if lost else slice(0, 0)

    record: dict[str, Any] = {
        "tag": tag,
        "episode": episode,
        "frames": len(frames),
        "selected_environment_steps": steps,
        "kit_add_auc": round(curve_aucs[curve_steps.index(steps)], 6),
        # B
        "first_loss_frame": lost,
        "first_loss_fraction": round(lost / len(frames), 3) if lost else None,
        "fraction_inside": {
            f"{int(band * 100)}cm": round(float((position <= band).mean()), 4) for band in BANDS_M
        },
        "position_error_cm": {
            "max": round(float(position.max()) * 100, 2),
            "mean_before_loss": round(float(position[before].mean()) * 100, 2),
            "mean_after_loss": (
                round(float(position[after].mean()) * 100, 2) if lost else None
            ),
            "mean_all": round(float(position.mean()) * 100, 2),
        },
        "orientation_error_deg": {
            "max": round(float(np.degrees(orientation.max())), 1),
            "mean_before_loss": round(float(np.degrees(orientation[before].mean())), 1),
            "mean_after_loss": (
                round(float(np.degrees(orientation[after].mean())), 1) if lost else None
            ),
            "mean_all": round(float(np.degrees(orientation.mean())), 1),
        },
        "add_cm": {
            "max": round(float(add.max()) * 100, 2),
            "mean_before_loss": round(float(add[before].mean()) * 100, 2),
            "recomputed_auc": round(add_auc(add), 6),
        },
        # C: both thresholds are on the time-averaged error, so each clip passes or fails once.
        "success": {
            "spider_pass": bool(
                position.mean() <= SPIDER_POSITION_M
                and orientation.mean() <= SPIDER_ORIENTATION_RAD
            ),
            "spider_position_margin_cm": round(
                (SPIDER_POSITION_M - float(position.mean())) * 100, 2
            ),
            "spider_orientation_margin_deg": round(
                float(np.degrees(SPIDER_ORIENTATION_RAD - orientation.mean())), 1
            ),
            "maniptrans_pass": bool(
                position.mean() < MANIPTRANS_POSITION_M
                and np.degrees(orientation.mean()) < MANIPTRANS_ORIENTATION_DEG
            ),
            "maniptrans_position_margin_cm": round(
                (MANIPTRANS_POSITION_M - float(position.mean())) * 100, 2
            ),
            "maniptrans_orientation_margin_deg": round(
                MANIPTRANS_ORIENTATION_DEG - float(np.degrees(orientation.mean())), 1
            ),
        },
        # D
        "checkpoint_curve": {
            "steps_millions": [s // 1_000_000 for s in curve_steps],
            "kit_add_auc": [round(a, 4) for a in curve_aucs],
            "shape": curve_shape(curve_steps, curve_aucs),
            "best_steps_millions": curve_steps[best_index] // 1_000_000,
            "selection_is_per_clip_best": curve_steps[best_index] == steps,
            "cost_of_selection": round(
                curve_aucs[best_index] - curve_aucs[curve_steps.index(steps)], 4
            ),
            "spread": round(float(np.std(curve_aucs)), 4),
        },
        "object_radius_cm": round(float(np.linalg.norm(vertices, axis=1).max()) * 100, 1),
    }

    record["hand_object"] = (
        None
        if staged is None
        else hand_object_summary(
            staged,
            *read_rollout(results / tag / f"policy_{steps}/objects.parquet"),
            vertices=vertices,
        )
    )

    record["_frames"] = frames
    record["_released"] = released
    record["_truth"] = truth
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="a run's results dir")
    parser.add_argument("--dataset-root", type=Path, required=True, help="the released dataset")
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True, help="the run's provenance.json")
    parser.add_argument("--triage", type=Path, help="results/second_wave_triage.json, for buckets")
    parser.add_argument("--anchors", type=Path, help="results/bridge_anchor_counts.json")
    parser.add_argument("--ceiling", type=Path, help="the cleaned tree's ceiling_per_clip.json")
    parser.add_argument("--work-dir", type=Path, required=True, help="scratch for the npz variants")
    parser.add_argument("--workers", type=int, default=2, help="at most 2 on this box")
    parser.add_argument("--clips", help="comma-separated tags; default every scored clip")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    provenance = json.loads(args.provenance.read_text())
    by_clip = provenance["ref_dataset_by_clip"]
    clips = provenance["clips"]
    wanted = [c.strip() for c in args.clips.split(",")] if args.clips else sorted(clips)

    metadata = {
        int(row["episode_index"]): row
        for row in (
            json.loads(line)
            for line in (args.dataset_root / "track_2" / TIER1 / "meta/episodes_metadata.jsonl")
            .read_text()
            .splitlines()
            if line.strip()
        )
    }
    # The kit's own sampler, so "the object's points" means the points the score is computed over.
    if str(args.kit_root) not in sys.path:
        sys.path.insert(0, str(args.kit_root))
    from v2dlb.vertices import build_blob

    names = sorted({str(metadata[int(t[1:])]["object"]) for t in wanted})
    _blob, vertex_cloud = build_blob(args.dataset_root / "track_2" / TIER1 / "mesh", names)

    triage = {}
    if args.triage and args.triage.is_file():
        triage = {c["tag"]: c for c in json.loads(args.triage.read_text())["clips"]}
    anchors = {}
    if args.anchors and args.anchors.is_file():
        anchors = json.loads(args.anchors.read_text())["per_clip"]["per_episode"]
    ceiling = {}
    if args.ceiling and args.ceiling.is_file():
        ceiling = json.loads(args.ceiling.read_text())["per_clip"]

    def one(tag: str) -> dict[str, Any]:
        episode = int(tag[1:])
        record = clip_metrics(
            tag,
            results=args.results,
            dataset_root=args.dataset_root,
            convert_root=Path(by_clip[tag]) if tag in by_clip else args.dataset_root,
            kit_root=args.kit_root,
            vertices=np.asarray(vertex_cloud[str(metadata[episode]["object"])], dtype=np.float64),
            staged=staged_hands(Path(clips[tag]["source_parquet"])),
            work=args.work_dir,
        )
        record.update(
            decompose(
                episode=episode,
                achieved=record.pop("_released"),
                truth=record.pop("_truth"),
                frames=record.pop("_frames"),
                kit_root=args.kit_root,
                dataset_root=args.dataset_root,
                work=args.work_dir,
            )
        )
        record["object"] = metadata[episode]["object"]
        record["bucket"] = triage.get(tag, {}).get("bucket", "strong")
        record["hands"] = triage.get(tag, {}).get("hands")
        record["scene"] = triage.get(tag, {}).get("scene")
        record["bridge_anchor_segments"] = anchors.get(str(episode), {}).get("anchor_segments")
        record["reference_ceiling_add_auc"] = (
            round(
                ceiling[str(episode)]["add_auc_before"]
                if ceiling[str(episode)]["fall_back_to_raw"]
                else ceiling[str(episode)]["add_auc_after"],
                4,
            )
            if str(episode) in ceiling
            else None
        )
        return record

    args.work_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 2))) as pool:
        rows = list(pool.map(one, wanted))

    # The clips the triage never classified are the ones that worked; label them once, here.
    for row in rows:
        if row["bucket"] == "strong" and row["hands"] is None:
            task = str(metadata[row["episode"]]["task"])
            row["hands"] = (
                "foot"
                if "foot" in task.lower() or "feet" in task.lower()
                else "two-hand"
                if "both hands" in task.lower() or "two hands" in task.lower()
                else "one-hand"
            )

    def column(key: str) -> list[Any]:
        return [row.get(key) for row in rows]

    correlations = {
        "what": "Spearman rank correlation of kit AUC against each clip property, over the clips "
        "where both are defined. Rank, not linear, because none of these is normal.",
        "pairs": {},
    }
    hands_rank = {"one-hand": 0, "two-hand": 1, "foot": 2}
    for label, values in (
        ("object_radius_cm", column("object_radius_cm")),
        ("bridge_anchor_segments", column("bridge_anchor_segments")),
        ("reference_ceiling_add_auc", column("reference_ceiling_add_auc")),
        ("first_loss_frame", [r["first_loss_frame"] for r in rows]),
        ("first_loss_fraction", [r["first_loss_fraction"] for r in rows]),
        ("hands_one_two_foot", [hands_rank.get(r["hands"]) for r in rows]),
        ("scene_floor0_table1", [{"floor": 0, "table": 1}.get(r["scene"]) for r in rows]),
        ("pantomime_fraction", [(r["hand_object"] or {}).get("pantomime_fraction") for r in rows]),
        ("frames", column("frames")),
    ):
        rho, n = spearman(column("kit_add_auc"), values)
        correlations["pairs"][label] = {"spearman_rho": None if np.isnan(rho) else round(rho, 3),
                                        "n": n}

    buckets: dict[str, dict[str, Any]] = {}
    for row in rows:
        entry = buckets.setdefault(row["bucket"], {"clips": [], "_auc": [], "_pos": [], "_ori": []})
        entry["clips"].append(row["tag"])
        entry["_auc"].append(row["kit_add_auc"])
        entry["_pos"].append(row["position_only"]["kit_add_auc"])
        entry["_ori"].append(row["orientation_only"]["kit_add_auc"])
    for entry in buckets.values():
        entry["n"] = len(entry["clips"])
        entry["mean_kit_add_auc"] = round(float(np.mean(entry.pop("_auc"))), 4)
        entry["mean_position_only_auc"] = round(float(np.mean(entry.pop("_pos"))), 4)
        entry["mean_orientation_only_auc"] = round(float(np.mean(entry.pop("_ori"))), 4)
        entry["clips"] = sorted(entry["clips"])

    record = {
        "what": "metrics-level reading of a finished run: AUC decomposition, error in time, "
        "success margins, checkpoint curves, hand-object distance and correlations",
        "thresholds": {
            "spider_position_m": SPIDER_POSITION_M,
            "spider_orientation_rad": SPIDER_ORIENTATION_RAD,
            "maniptrans_position_m": MANIPTRANS_POSITION_M,
            "maniptrans_orientation_deg": MANIPTRANS_ORIENTATION_DEG,
            "lost_limit_m": LOST_LIMIT_M,
            "out_of_hand_excess_m": OUT_OF_HAND_EXCESS_M,
            "palm_moving_m_per_s": PALM_MOVING_M_PER_S,
            "note": "the first four are the kit's own; both success rates threshold the "
            "TIME-AVERAGED error, so a clip passes or fails once",
        },
        "caveat_part_e": "the boxes exported object trajectories only, so the achieved palm was "
        "never recorded. Hand-object distances use the REFERENCE palm and so measure 'the hand "
        "went where it should and the object did not follow', assuming body tracking is good.",
        "mean_kit_add_auc": round(float(np.mean(column("kit_add_auc"))), 6),
        "mean_position_only_auc": round(
            float(np.mean([r["position_only"]["kit_add_auc"] for r in rows])), 6
        ),
        "mean_orientation_only_auc": round(
            float(np.mean([r["orientation_only"]["kit_add_auc"] for r in rows])), 6
        ),
        "buckets": dict(sorted(buckets.items())),
        "correlations": correlations,
        "clips": rows,
    }

    print(f"{'clip':5s} {'bucket':14s} {'AUC':>6s} {'posonly':>8s} {'orionly':>8s} {'lost@':>6s} "
          f"{'<10cm':>6s} {'SP':>3s} {'MP':>3s} {'shape':11s} {'panto':>6s}")
    for row in sorted(rows, key=lambda r: r["kit_add_auc"]):
        hand = row["hand_object"] or {}
        print(
            f"{row['tag']:5s} {row['bucket']:14s} {row['kit_add_auc']:6.3f} "
            f"{row['position_only']['kit_add_auc']:8.3f} "
            f"{row['orientation_only']['kit_add_auc']:8.3f} "
            f"{(row['first_loss_frame'] if row['first_loss_frame'] is not None else '-'):>6} "
            f"{row['fraction_inside']['10cm']:6.2f} "
            f"{'Y' if row['success']['spider_pass'] else 'n':>3} "
            f"{'Y' if row['success']['maniptrans_pass'] else 'n':>3} "
            f"{row['checkpoint_curve']['shape']:11s} "
            f"{hand.get('pantomime_fraction', float('nan')):6.2f}"
        )
    print(f"\nmean AUC {record['mean_kit_add_auc']:.4f}  "
          f"position-only {record['mean_position_only_auc']:.4f}  "
          f"orientation-only {record['mean_orientation_only_auc']:.4f}")
    for label, pair in record["correlations"]["pairs"].items():
        print(f"  rho(AUC, {label:26s}) = {pair['spearman_rho']}  n={pair['n']}")
    if args.out:
        args.out.write_text(json.dumps(record, indent=2) + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
