"""Score the released Tier 2 object input against the clean Tier 1 reference.

A policy that tracked the Tier 2 input perfectly would score exactly this, so it is
the ceiling for an entry that trusts its input. Smoothed variants show how much of
the corruption a filter recovers without touching the clean data.

SP-SR, MP-SR and MPPE follow metric_code/track_2 in the submission kit. AUC needs mesh
vertices that are not downloaded here, so `auc_kp` is ADD-AUC over the six MPPE keypoints
(a proxy, not the leaderboard number).

Usage: python scripts/tier2_input_ceiling.py DATA_DIR [-o results/tier2_input_ceiling.json]
DATA_DIR holds tier_1_multiview_caption/ and tier_2_synthetic_noise/ episode parquets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.spatial.transform import Rotation as R

ADD_THRESHOLDS_M = np.arange(0.01, 0.10, 0.01)
KEYPOINTS = 0.05 * np.concatenate([np.eye(3), -np.eye(3)])
TIER1, TIER2 = "tier_1_multiview_caption", "tier_2_synthetic_noise"


def load(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t = pq.read_table(path, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.stack(t.column(0).to_numpy(zero_copy_only=False)).astype(np.float64)
    vis = t.column(1).to_numpy(zero_copy_only=False).astype(bool)
    return pose[:, :3], pose[:, [4, 5, 6, 3]]  # xyz, wxyz -> xyzw


def fill_invisible(pos, quat, vis):
    """Hold the last visible pose (zero-filled rows are not rotations)."""
    pos, quat = pos.copy(), quat.copy()
    idx = np.where(vis)[0]
    src = idx[np.clip(np.searchsorted(idx, np.arange(len(vis)), side="right") - 1, 0, None)]
    return pos[src], quat[src]


def smooth(pos, quat, kind: str, width: float):
    if kind == "raw":
        return pos, quat
    q = quat.copy()
    for i in range(1, len(q)):  # keep consecutive quaternions in one hemisphere before filtering
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    if kind == "gauss":
        f = lambda x: gaussian_filter1d(x, width, axis=0, mode="nearest")
    elif kind == "median":
        f = lambda x: median_filter(x, size=(int(width), 1), mode="nearest")
    elif kind == "median+gauss":
        f = lambda x: gaussian_filter1d(median_filter(x, size=(9, 1), mode="nearest"), width, axis=0, mode="nearest")
    else:
        raise ValueError(kind)
    q = f(q)
    return f(pos), q / np.linalg.norm(q, axis=1, keepdims=True)


def episode_metrics(pa, qa, pr, qr) -> dict:
    pos_err = np.linalg.norm(pa - pr, axis=1)
    dot = np.clip(np.abs(np.sum(qa * qr, axis=1)), 0.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    ka = pa[:, None] + np.einsum("tij,kj->tki", R.from_quat(qa).as_matrix(), KEYPOINTS)
    kr = pr[:, None] + np.einsum("tij,kj->tki", R.from_quat(qr).as_matrix(), KEYPOINTS)
    kp_err = np.linalg.norm(ka - kr, axis=-1)
    add = kp_err.mean(axis=1)
    auc = np.trapezoid([np.mean(add < t) for t in ADD_THRESHOLDS_M], x=np.linspace(0, 1, len(ADD_THRESHOLDS_M)))
    return {
        "pos_cm": float(pos_err.mean() * 100),
        "rot_deg": float(np.degrees(ang.mean())),
        "sp_sr": bool(pos_err.mean() <= 0.1 and ang.mean() <= 0.5),
        "mp_sr": bool(pos_err.mean() < 0.03 and np.degrees(ang.mean()) < 30.0),
        "mppe_cm": float(kp_err.mean() * 100),
        "auc_kp": float(auc),
    }


def noise_profile(p1, q1, p2, q2) -> dict:
    """Split the Tier 2 position error into a slow (>1 s) part and a fast (jitter) part."""
    d = p2 - p1
    slow = gaussian_filter1d(d, 15, axis=0, mode="nearest")
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=1)), 0, 1)
    return {
        "pos_err_rms_cm": float(np.sqrt((np.linalg.norm(d, axis=1) ** 2).mean()) * 100),
        "slow_rms_cm": float(np.sqrt((np.linalg.norm(slow, axis=1) ** 2).mean()) * 100),
        "fast_rms_cm": float(np.sqrt((np.linalg.norm(d - slow, axis=1) ** 2).mean()) * 100),
        "frames_err_over_10cm": float(np.mean(np.linalg.norm(d, axis=1) > 0.10)),
        "rot_err_p95_deg": float(np.degrees(np.percentile(2 * np.arccos(dot), 95))),
        "frozen_frames": float(np.mean(np.all(np.abs(np.diff(p2, axis=0)) < 1e-7, axis=1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("-o", "--out", type=Path, default=Path("results/tier2_input_ceiling.json"))
    args = ap.parse_args()

    variants = [("raw", 0)] + [("gauss", s) for s in (2, 5, 10, 20)] + [("median", 15), ("median+gauss", 5)]
    per_variant: dict[str, list[dict]] = {f"{k}{w or ''}": [] for k, w in variants}
    profiles = []
    for ep in range(30):
        name = f"episode_{ep:06d}.parquet"
        t1 = pq.read_table(args.data_dir / TIER1 / name, columns=["observation.object.visible"])
        v1 = t1.column(0).to_numpy(zero_copy_only=False).astype(bool)
        t2 = pq.read_table(args.data_dir / TIER2 / name, columns=["observation.object.visible"])
        v2 = t2.column(0).to_numpy(zero_copy_only=False).astype(bool)
        p1, q1 = fill_invisible(*load(args.data_dir / TIER1 / name), v1)
        p2, q2 = fill_invisible(*load(args.data_dir / TIER2 / name), v2)
        profiles.append(noise_profile(p1, q1, p2, q2))
        for kind, w in variants:
            pa, qa = smooth(p2, q2, kind, w)
            m = episode_metrics(pa[v1], qa[v1], p1[v1], q1[v1])
            m["episode"] = ep
            per_variant[f"{kind}{w or ''}"].append(m)

    summary = {}
    for key, rows in per_variant.items():
        summary[key] = {
            "sp_sr": float(np.mean([r["sp_sr"] for r in rows])),
            "mp_sr": float(np.mean([r["mp_sr"] for r in rows])),
            "mppe_cm": float(np.mean([r["mppe_cm"] for r in rows])),
            "auc_kp": float(np.mean([r["auc_kp"] for r in rows])),
            "pos_cm_mean": float(np.mean([r["pos_cm"] for r in rows])),
            "rot_deg_mean": float(np.mean([r["rot_deg"] for r in rows])),
        }
    prof = {k: float(np.mean([p[k] for p in profiles])) for k in profiles[0]}
    out = {
        "what": "Tier 2 object input scored as if tracked perfectly, vs clean Tier 1 reference",
        "auc_note": "auc_kp is ADD-AUC over 6 keypoints at 5 cm, not mesh vertices; proxy only",
        "noise_profile_mean_over_episodes": prof,
        "summary": summary,
        "per_episode": per_variant,
        "per_episode_noise": profiles,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"noise": prof, "summary": summary}, indent=1))


if __name__ == "__main__":
    main()
