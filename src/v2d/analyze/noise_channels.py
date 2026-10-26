"""`v2d noise-channels`: which channels Tier 2 corrupts, how hard, and which ones the metric reads.

Four measurements, all free, all on the released data and the rollouts we have already paid for:

A. Per-channel Tier 2 noise against Tier 1, over all 30 clips, split into frames where a hand is
   in contact with the object and frames where none is. Each channel gets magnitude (mean, std,
   max), spread across clips, and temporal character -- frame-to-frame jitter against slow drift,
   plus lag-1 autocorrelation, which is what separates a channel a filter can fix from one it
   cannot.
B. Whether the clean object pose is predictable from the clean hands: the variance of the object
   pose expressed in the palm frame during held frames, on Tier 1 only. A clip whose grasp is rigid
   gives a policy an absolute anchor; one whose grasp slips does not.
C. What each noise channel costs the metric, by scoring the noisy input against the clean truth
   with the kit scorer and then removing one channel at a time.
D. What our own policies did across the checkpoints we have: checkpoint-to-checkpoint spread, how
   often the object was dropped, and whether orientation error tracks AUC once object size is
   accounted for.

The held/resting split is the retargeter's own per-frame `hand_contact_active` at its 2 cm
threshold, taken from the TIER 1 clip and applied to both tiers, so the split itself is never
contaminated by the noise being measured.

    v2d noise-channels --dataset-root DATA --tier1-root T1 --tier2-root T2 \
        --part-c-dir PARTC --checkpoints-json CKPT -o results/noise_channels.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from ..clean.tilt_yaw import align_rotation, geodesic_deg, quat_from_matrix, rotation_matrix
from ..geometry import wxyz_to_xyzw

FPS = 30.0
DRIFT_WINDOW = 15  # frames, 0.5 s at 30 fps: anything slower than this is drift, not jitter
RIGID_POSITION_CM = 2.0
RIGID_ORIENTATION_DEG = 10.0
TWO_HANDED_FRACTION = 0.20  # of held frames with BOTH hands in contact

# The 43-joint G1+Dex3 layout, by name order, as the retargeter writes it.
JOINT_GROUPS: dict[str, tuple[int, ...]] = {
    "legs": tuple(range(0, 12)),
    "waist": tuple(range(12, 15)),
    "arms": (15, 16, 17, 18, 29, 30, 31, 32),
    "wrists": (19, 20, 21, 33, 34, 35),
    "fingers": tuple(range(22, 29)) + tuple(range(36, 43)),
}

CHANNEL_UNITS: dict[str, str] = {
    "object_position": "cm",
    "object_tilt": "deg",
    "object_yaw": "deg",
    "object_orientation_total": "deg",
    "root_position": "cm",
    "root_orientation": "deg",
    "palm_position": "cm",
    "palm_orientation": "deg",
    **{f"joint_{group}": "deg" for group in JOINT_GROUPS},
}


# --------------------------------------------------------------------------- loading
def episode_map(dataset_root: Path, tier_dir: str) -> dict[int, str]:
    path = dataset_root / "track_2" / tier_dir / "meta" / "episodes_metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {int(r["episode_index"]): str(r["sequence_id"]) for r in rows}


def clip_path(root: Path, sequence_id: str) -> Path:
    return root / f"whole_body/soma/sequence_id={sequence_id}/robot_name=g1/data.parquet"


def load_clip(path: Path) -> dict[str, Any]:
    """One retargeted clip as arrays. The schema stores a single row of nested lists."""
    table = pq.read_table(path)
    row = {name: table[name][0].as_py() for name in table.column_names}
    pose = np.asarray(row["ee_pose_w"], dtype=np.float64)  # (T, 2, 7): xyz then wxyz
    return {
        "root_position": np.asarray(row["robot_root_position"], dtype=np.float64),
        "root_quat_xyzw": wxyz_to_xyzw(np.asarray(row["robot_root_wxyz"], dtype=np.float64)),
        "joints": np.asarray(row["robot_joint_positions"], dtype=np.float64),
        "palm_position": pose[:, :, :3],
        "palm_quat_xyzw": wxyz_to_xyzw(pose[:, :, 3:]),
        "object_position": np.asarray(row["object_body_position"], dtype=np.float64)[:, 0, :],
        "object_quat_xyzw": wxyz_to_xyzw(
            np.asarray(row["object_body_wxyz"], dtype=np.float64)[:, 0, :]
        ),
        "contact_active": np.asarray(row["hand_contact_active"], dtype=np.float64),  # (2, T)
        "object_mesh_radius": float(np.asarray(row["object_mesh_radius"], dtype=np.float64)[0]),
        "object_name": str(row["object_name"]),
        "frames": len(row["robot_root_position"]),
    }


# --------------------------------------------------------------------------- statistics
def temporal(signal: np.ndarray) -> dict[str, float | None]:
    """Jitter against drift, and lag-1 autocorrelation, for one scalar error series."""
    if signal.size < 3:
        return {"jitter_std": None, "drift_std": None, "lag1_autocorr": None}
    jitter = float(np.std(np.diff(signal)))
    kernel = np.ones(min(DRIFT_WINDOW, signal.size)) / min(DRIFT_WINDOW, signal.size)
    drift = float(np.std(np.convolve(signal, kernel, mode="valid")))
    centred = signal - signal.mean()
    denominator = float(np.dot(centred, centred))
    lag1 = float(np.dot(centred[:-1], centred[1:]) / denominator) if denominator > 0 else None
    return {"jitter_std": jitter, "drift_std": drift, "lag1_autocorr": lag1}


def describe(signal: np.ndarray) -> dict[str, float | None]:
    if signal.size == 0:
        return {"frames": 0, "mean": None, "std": None, "max": None,
                "jitter_std": None, "drift_std": None, "lag1_autocorr": None}
    if not np.any(signal):
        # Identically zero is not a quiet channel, it is an absent one: the official retargeter
        # leaves all 14 Dex3 finger joints at exactly 0 in BOTH tiers, so there is nothing to
        # corrupt and nothing to track. Reporting 0.000 noise would read as "clean".
        return {"frames": int(signal.size), "mean": 0.0, "std": 0.0, "max": 0.0,
                "jitter_std": 0.0, "drift_std": 0.0, "lag1_autocorr": None,
                "degenerate": "identically zero in both tiers"}
    return {
        "frames": int(signal.size),
        "mean": float(signal.mean()),
        "std": float(signal.std()),
        "max": float(signal.max()),
        **temporal(signal),
    }


def pool(per_clip: dict[int, dict[str, Any]], key: str) -> dict[str, Any]:
    """Aggregate a per-clip statistic into one channel-level summary."""
    means = [v[key]["mean"] for v in per_clip.values() if v[key]["mean"] is not None]
    maxes = [v[key]["max"] for v in per_clip.values() if v[key]["max"] is not None]
    if not means:
        return {"clips": 0}
    weights = [v[key]["frames"] for v in per_clip.values() if v[key]["mean"] is not None]
    return {
        "clips": len(means),
        "mean": float(np.average(means, weights=weights)),
        "std_of_per_clip_mean": float(np.std(means)),
        "per_clip_mean_min": float(np.min(means)),
        "per_clip_mean_max": float(np.max(means)),
        "max": float(np.max(maxes)),
        "jitter_std": float(np.mean([v[key]["jitter_std"] for v in per_clip.values()
                                     if v[key]["jitter_std"] is not None])),
        "drift_std": float(np.mean([v[key]["drift_std"] for v in per_clip.values()
                                    if v[key]["drift_std"] is not None])),
        "lag1_autocorr": _mean_or_none([v[key]["lag1_autocorr"] for v in per_clip.values()]),
        "degenerate": next((v[key]["degenerate"] for v in per_clip.values()
                            if v[key].get("degenerate")), None),
    }


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else None


# --------------------------------------------------------------------------- part A
def tilt_yaw_split(clean_xyzw: np.ndarray, noisy_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tilt is the mis-orientation of the object's own z axis; yaw is what is left about it.

    Same construction as `v2d tilt-yaw`, with the object frame's z axis standing in for that
    stage's support normal, so the two measurements are comparable.
    """
    axis = np.array([0.0, 0.0, 1.0])
    r_clean, r_noisy = rotation_matrix(clean_xyzw), rotation_matrix(noisy_xyzw)
    m_clean = np.einsum("tji,j->ti", r_clean, axis)
    m_noisy = np.einsum("tji,j->ti", r_noisy, axis)
    tilt = np.degrees(np.arccos(np.clip(np.sum(m_clean * m_noisy, axis=-1), -1.0, 1.0)))
    correction = align_rotation(m_noisy, m_clean)
    yaw = geodesic_deg(clean_xyzw, quat_from_matrix(r_noisy @ np.swapaxes(correction, -1, -2)))
    return tilt, yaw


def channel_errors(clean: dict[str, Any], noisy: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-frame Tier 2 minus Tier 1, one scalar series per channel."""
    frames = min(clean["frames"], noisy["frames"])
    cut = slice(0, frames)
    tilt, yaw = tilt_yaw_split(clean["object_quat_xyzw"][cut], noisy["object_quat_xyzw"][cut])
    out = {
        "object_position": np.linalg.norm(
            noisy["object_position"][cut] - clean["object_position"][cut], axis=-1) * 100.0,
        "object_tilt": tilt,
        "object_yaw": yaw,
        "object_orientation_total": geodesic_deg(
            clean["object_quat_xyzw"][cut], noisy["object_quat_xyzw"][cut]),
        "root_position": np.linalg.norm(
            noisy["root_position"][cut] - clean["root_position"][cut], axis=-1) * 100.0,
        "root_orientation": geodesic_deg(
            clean["root_quat_xyzw"][cut], noisy["root_quat_xyzw"][cut]
        ),
        # Both palms pooled: the mean over the two hands, per frame.
        "palm_position": np.linalg.norm(
            noisy["palm_position"][cut] - clean["palm_position"][cut], axis=-1
        ).mean(axis=-1) * 100.0,
        "palm_orientation": geodesic_deg(
            clean["palm_quat_xyzw"][cut], noisy["palm_quat_xyzw"][cut]).mean(axis=-1),
    }
    delta = np.degrees(noisy["joints"][cut] - clean["joints"][cut])
    for group, index in JOINT_GROUPS.items():
        out[f"joint_{group}"] = np.sqrt((delta[:, list(index)] ** 2).mean(axis=1))
    return out


def part_a(dataset_root: Path, tier1_root: Path, tier2_root: Path) -> dict[str, Any]:
    maps = episode_map(dataset_root, "tier_1_multiview_caption")
    per_clip: dict[str, dict[int, dict[str, Any]]] = {name: {} for name in CHANNEL_UNITS}
    held_fraction: dict[int, float] = {}
    missing: list[int] = []

    for episode, sequence in sorted(maps.items()):
        clean_path, noisy_path = clip_path(tier1_root, sequence), clip_path(tier2_root, sequence)
        if not (clean_path.is_file() and noisy_path.is_file()):
            missing.append(episode)
            continue
        clean, noisy = load_clip(clean_path), load_clip(noisy_path)
        errors = channel_errors(clean, noisy)
        frames = len(errors["object_position"])
        # The Tier 1 mask, so the split is not contaminated by the noise being measured.
        held = clean["contact_active"][:, :frames].max(axis=0) > 0.0
        held_fraction[episode] = float(held.mean())
        for name, signal in errors.items():
            per_clip[name][episode] = {
                "all": describe(signal),
                "held": describe(signal[held]),
                "resting": describe(signal[~held]),
            }

    channels = {
        name: {
            "unit": CHANNEL_UNITS[name],
            "all": pool(clips, "all"),
            "held": pool(clips, "held"),
            "resting": pool(clips, "resting"),
            "per_clip_mean": {str(e): v["all"]["mean"] for e, v in sorted(clips.items())},
        }
        for name, clips in per_clip.items()
        if clips
    }
    return {
        "definition": {
            "noise": "Tier 2 retarget minus Tier 1 retarget, per frame, same retargeter",
            "held": "the Tier 1 clip's own hand_contact_active > 0 on either hand (2 cm threshold)",
            "tilt": "mis-orientation of the object's own z axis",
            "yaw": "the residual rotation about that axis after the tilt is aligned away",
            "jitter_std": "std of the frame-to-frame difference of the error series",
            "drift_std": f"std of the error series after a {DRIFT_WINDOW}-frame moving average",
            "joint_groups": {k: list(v) for k, v in JOINT_GROUPS.items()},
        },
        "clips_measured": len(held_fraction),
        "clips_missing_a_retarget": missing,
        "held_fraction_per_clip": {str(k): v for k, v in sorted(held_fraction.items())},
        "held_fraction_mean": (
            float(np.mean(list(held_fraction.values()))) if held_fraction else None
        ),
        "channels": channels,
    }


# --------------------------------------------------------------------------- part B
def quat_conjugate(q_xyzw: np.ndarray) -> np.ndarray:
    out = q_xyzw.copy()
    out[..., :3] *= -1.0
    return out


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) spans where `mask` is true."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _segment_mask(mask: np.ndarray, start: int, end: int) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    out[start:end] = mask[start:end]
    return out


def _spread(clip: dict[str, Any], frames: int, mask: np.ndarray, index: int) -> tuple[float, float]:
    """Positional and angular spread of the object in one palm's frame, over `mask`."""
    palm_p = clip["palm_position"][:frames][mask, index, :]
    palm_q = clip["palm_quat_xyzw"][:frames][mask, index, :]
    obj_p = clip["object_position"][:frames][mask]
    obj_q = clip["object_quat_xyzw"][:frames][mask]
    rel_p = np.einsum("tji,tj->ti", rotation_matrix(palm_q), obj_p - palm_p)
    rel_q = quat_multiply(quat_conjugate(palm_q), obj_q)
    mean_q = rel_q[0] if len(rel_q) == 1 else _mean_quat(rel_q)
    return (
        float(np.sqrt(rel_p.var(axis=0).sum()) * 100.0),
        float(geodesic_deg(np.broadcast_to(mean_q, rel_q.shape), rel_q).mean()),
    )


def part_b(dataset_root: Path, tier1_root: Path) -> dict[str, Any]:
    """Is the clean object pose rigid in the palm frame while the object is held?"""
    maps = episode_map(dataset_root, "tier_1_multiview_caption")
    clips: dict[str, Any] = {}
    for episode, sequence in sorted(maps.items()):
        path = clip_path(tier1_root, sequence)
        if not path.is_file():
            continue
        clip = load_clip(path)
        frames = clip["frames"]
        contact = clip["contact_active"][:, :frames] > 0.0
        held = contact.max(axis=0)
        if held.sum() < 10:
            clips[str(episode)] = {"object": clip["object_name"], "held_frames": int(held.sum()),
                                   "verdict": "too few held frames to measure"}
            continue
        both = float((contact[0] & contact[1])[held].mean())
        # Each hand that is ever in contact is measured on its own contact frames.
        sides: dict[str, Any] = {}
        for index, side in enumerate(("left", "right")):
            mask = contact[index]
            if mask.sum() < 10:
                continue
            palm_p = clip["palm_position"][:frames][mask, index, :]
            palm_q = clip["palm_quat_xyzw"][:frames][mask, index, :]
            obj_p = clip["object_position"][:frames][mask]
            obj_q = clip["object_quat_xyzw"][:frames][mask]
            # Object pose in the palm frame: R_palm^T (p_obj - p_palm), q_palm^-1 (x) q_obj.
            rel_p = np.einsum("tji,tj->ti", rotation_matrix(palm_q), obj_p - palm_p)
            rel_q = quat_multiply(quat_conjugate(palm_q), obj_q)
            # Spread about the mean: total positional std, and mean angle from the mean rotation.
            spread_cm = float(np.sqrt((rel_p.var(axis=0)).sum()) * 100.0)
            mean_q = rel_q[0] if len(rel_q) == 1 else _mean_quat(rel_q)
            spread_deg = float(geodesic_deg(np.broadcast_to(mean_q, rel_q.shape), rel_q).mean())
            # Spread over ALL contact frames conflates a loose grasp with several different
            # grasps, so measure each contiguous contact segment on its own too. If the
            # within-segment spread is small and the overall one is not, the hand regrasped; if
            # both are large, the object genuinely moves in the hand.
            segments = [(a, b) for a, b in _runs(mask) if b - a >= 10]
            within = [
                _spread(clip, frames, mask_slice, index)
                for mask_slice in (_segment_mask(mask, a, b) for a, b in segments)
            ]
            sides[side] = {
                "frames": int(mask.sum()),
                "position_spread_cm": spread_cm,
                "orientation_spread_deg": spread_deg,
                "contact_segments": len(segments),
                "within_segment_position_spread_cm": float(np.median([w[0] for w in within]))
                if within else None,
                "within_segment_orientation_spread_deg": float(np.median([w[1] for w in within]))
                if within else None,
                "rigid": bool(spread_cm < RIGID_POSITION_CM and spread_deg < RIGID_ORIENTATION_DEG),
                "rigid_within_segment": bool(
                    within
                    and np.median([w[0] for w in within]) < RIGID_POSITION_CM
                    and np.median([w[1] for w in within]) < RIGID_ORIENTATION_DEG
                ),
            }
        rigid = bool(sides) and all(s["rigid"] for s in sides.values())
        clips[str(episode)] = {
            "object": clip["object_name"],
            "object_mesh_radius_cm": clip["object_mesh_radius"] * 100.0,
            "held_frames": int(held.sum()),
            "both_hands_fraction_of_held": both,
            "two_handed": bool(both >= TWO_HANDED_FRACTION),
            "sides": sides,
            "rigid_grasp": rigid,
        }

    measured = [c for c in clips.values() if "rigid_grasp" in c]
    two_handed = [c for c in measured if c["two_handed"]]
    one_handed = [c for c in measured if not c["two_handed"]]
    return {
        "definition": {
            "rigid": f"position spread < {RIGID_POSITION_CM} cm AND orientation spread "
                     f"< {RIGID_ORIENTATION_DEG} deg, every hand in contact",
            "spread": "position: sqrt of the summed per-axis variance; orientation: mean geodesic "
                      "angle from the mean rotation, both over that hand's contact frames",
            "two_handed": f"both hands in contact for >= {TWO_HANDED_FRACTION:.0%} of held frames",
        },
        "clips_measured": len(measured),
        "rigid_clips": sum(c["rigid_grasp"] for c in measured),
        "not_rigid_clips": sum(not c["rigid_grasp"] for c in measured),
        "rigid_clips_within_contact_segment": sum(
            bool(c["sides"]) and all(s["rigid_within_segment"] for s in c["sides"].values())
            for c in measured
        ),
        "one_handed": {
            "clips": len(one_handed),
            "rigid": sum(c["rigid_grasp"] for c in one_handed),
            "median_position_spread_cm": _median_over(one_handed, "position_spread_cm"),
            "median_orientation_spread_deg": _median_over(one_handed, "orientation_spread_deg"),
        },
        "two_handed": {
            "clips": len(two_handed),
            "rigid": sum(c["rigid_grasp"] for c in two_handed),
            "median_position_spread_cm": _median_over(two_handed, "position_spread_cm"),
            "median_orientation_spread_deg": _median_over(two_handed, "orientation_spread_deg"),
        },
        "per_clip": clips,
    }


def _mean_quat(q_xyzw: np.ndarray) -> np.ndarray:
    """Mean rotation by the dominant eigenvector of the outer-product sum, sign-aligned first."""
    aligned = q_xyzw * np.sign(q_xyzw @ q_xyzw[0])[:, None]
    _, vectors = np.linalg.eigh(aligned.T @ aligned)
    return vectors[:, -1]


def _median_over(clips: list[dict[str, Any]], key: str) -> float | None:
    values = [s[key] for c in clips for s in c.get("sides", {}).values()]
    return float(np.median(values)) if values else None


# --------------------------------------------------------------------------- parts C and D
def part_c(part_c_dir: Path) -> dict[str, Any]:
    """What each noise channel costs the kit scorer. The specs are the kit scorer's own."""
    def read(name: str) -> dict[str, float]:
        return json.loads((part_c_dir / f"{name}.json").read_text())["overall"]

    arms = {
        "clean_truth": {"spec": "reference", **read("reference")},
        "tier2_input": {"spec": "tier2-input", **read("tier2-input")},
        "orientation_noise_removed": {"spec": "tier2-pos-only", **read("tier2-pos-only")},
        "position_noise_removed": {"spec": "tier2-rot-only", **read("tier2-rot-only")},
    }
    base = arms["tier2_input"]
    for name in ("orientation_noise_removed", "position_noise_removed"):
        arms[name]["auc_gain_vs_tier2_input"] = arms[name]["add_auc"] - base["add_auc"]
        arms[name]["spider_sr_gain_vs_tier2_input"] = arms[name]["spider_sr"] - base["spider_sr"]
    return {
        "scorer": "the submission kit's own track_2 metric code, all 30 clips, scored grid",
        "arms": arms,
        "root_and_joint_noise_removed": {
            "auc_gain_vs_tier2_input": 0.0,
            "why": "zero by construction, not by measurement: a Track 2 submission carries the "
                   "object pose per frame and nothing else, so the metric cannot read root or "
                   "joint channels at all. They matter only through what they do to the policy.",
        },
        "ratio_orientation_to_position_auc_cost": (
            arms["orientation_noise_removed"]["auc_gain_vs_tier2_input"]
            / arms["position_noise_removed"]["auc_gain_vs_tier2_input"]
        ),
    }


def part_d(checkpoints_json: Path, dataset_root: Path, tier1_root: Path) -> dict[str, Any]:
    """What our own policies did across the 4 clips x 7 checkpoints already paid for."""
    data = json.loads(checkpoints_json.read_text())
    maps = episode_map(dataset_root, "tier_1_multiview_caption")
    clips: dict[str, Any] = {}
    flat: list[tuple[float, float, float]] = []  # (orientation_deg, auc, radius_cm)
    for clip, record in data.items():
        if clip.startswith("_"):
            continue
        rows = record["checkpoints"]
        aucs = np.array([r["evaluator_add_auc"] for r in rows])
        mppes = np.array([r["evaluator_mppe_cm"] for r in rows])
        oris = np.array([r["evaluator_object_orientation_error_deg"] for r in rows])
        radius = None
        path = clip_path(tier1_root, maps[record["episode"]])
        if path.is_file():
            radius = load_clip(path)["object_mesh_radius"] * 100.0
        clips[clip] = {
            "episode": record["episode"],
            "object_mesh_radius_cm": radius,
            "checkpoints": len(rows),
            "auc_mean": float(aucs.mean()),
            "auc_std_across_checkpoints": float(aucs.std()),
            "auc_min": float(aucs.min()),
            "auc_max": float(aucs.max()),
            "mppe_mean_cm": float(mppes.mean()),
            "mppe_std_across_checkpoints_cm": float(mppes.std()),
            "dropped_checkpoints": int(sum(r["dropped"] for r in rows)),
            "orientation_error_deg_mean": float(oris.mean()),
            "spearman_orientation_vs_auc": _spearman(oris, aucs),
        }
        if radius is not None:
            flat.extend((o, a, radius) for o, a in zip(oris, aucs, strict=True))

    oris = np.array([f[0] for f in flat])
    aucs = np.array([f[1] for f in flat])
    radii = np.array([f[2] for f in flat])
    small = radii <= float(np.median(radii))
    return {
        "note": "evaluator metrics, which is what every checkpoint has; the kit scorer agrees on "
                "ranking but not on absolute AUC",
        "per_clip": clips,
        "pooled": {
            "checkpoints": len(flat),
            "spearman_orientation_vs_auc": _spearman(oris, aucs),
            "spearman_orientation_vs_auc_small_objects": _spearman(oris[small], aucs[small]),
            "spearman_orientation_vs_auc_large_objects": _spearman(oris[~small], aucs[~small]),
            "object_radius_median_cm": float(np.median(radii)),
            "dropped_checkpoints": int(sum(
                c["dropped_checkpoints"] for c in clips.values())),
            "total_checkpoints": sum(c["checkpoints"] for c in clips.values()),
        },
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3:
        return None
    rank_a = np.argsort(np.argsort(a)).astype(float)
    rank_b = np.argsort(np.argsort(b)).astype(float)
    rank_a -= rank_a.mean()
    rank_b -= rank_b.mean()
    denominator = float(np.sqrt(np.dot(rank_a, rank_a) * np.dot(rank_b, rank_b)))
    return float(np.dot(rank_a, rank_b) / denominator) if denominator > 0 else None


# --------------------------------------------------------------------------- entry point
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tier1-root", type=Path, required=True,
                        help="retargeted Tier 1 tree holding whole_body/soma/...")
    parser.add_argument("--tier2-root", type=Path, required=True)
    parser.add_argument("--part-c-dir", type=Path, required=True,
                        help="directory of `v2d score` records: reference.json, tier2-input.json, "
                             "tier2-pos-only.json, tier2-rot-only.json")
    parser.add_argument("--checkpoints-json", type=Path, required=True,
                        help="the per-checkpoint record from the no-drop selection")
    parser.add_argument("--parts", default="abcd", help="subset of parts to run")
    parser.add_argument("-o", "--out", type=Path, required=True)
    args = parser.parse_args(argv)

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         check=False).stdout.strip()
    record: dict[str, Any] = {
        "what": "which channels the policy should track and how hard (docs/TRACK_WHAT.md)",
        "git_sha": sha,
        "inputs": {
            "dataset_root": str(args.dataset_root),
            "tier1_root": str(args.tier1_root),
            "tier2_root": str(args.tier2_root),
        },
    }
    if "a" in args.parts:
        record["part_a_noise_per_channel"] = part_a(args.dataset_root, args.tier1_root,
                                                    args.tier2_root)
    if "b" in args.parts:
        record["part_b_grasp_rigidity"] = part_b(args.dataset_root, args.tier1_root)
    if "c" in args.parts:
        record["part_c_metric_sensitivity"] = part_c(args.part_c_dir)
    if "d" in args.parts:
        record["part_d_our_policies"] = part_d(args.checkpoints_json, args.dataset_root,
                                               args.tier1_root)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
