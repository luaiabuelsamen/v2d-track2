"""Probe: does snapping Tier 2 object poses to one pose per detected rest segment reduce error?

Rest segments are found from the Tier 2 input alone (speed below a threshold for at least
MIN_LEN frames); the clean Tier 1 reference is used only for scoring. Prints per-variant
SP-SR / MP-SR / MPPE / mean rotation error and writes results/tier2_rest_snap.json.

Usage: python scripts/tier2_rest_snap.py DATA_DIR
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
import tier2_input_ceiling as T  # noqa: E402

MIN_LEN = 15


def rest_segments(pos: np.ndarray, thresh: float) -> list[tuple[int, int]]:
    speed = np.linalg.norm(np.gradient(gaussian_filter1d(pos, 3, axis=0), axis=0), axis=1) * 30.0
    still = speed < thresh
    segs, start = [], None
    for i, s in enumerate(np.r_[still, False]):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if i - start >= MIN_LEN:
                segs.append((start, i))
            start = None
    return segs


def snap(pos, quat, segs):
    pos, quat = pos.copy(), quat.copy()
    for a, b in segs:
        pos[a:b] = np.median(pos[a:b], axis=0)
        quat[a:b] = R.from_quat(quat[a:b]).mean().as_quat()
    return pos, quat


def main() -> None:
    d = Path(sys.argv[1])
    out: dict = {}
    for thresh in (None, 0.02, 0.05, 0.10):
        rows, rest_frac = [], []
        for ep in range(30):
            name = f"episode_{ep:06d}.parquet"
            vis = lambda tier: pq.read_table(d / tier / name, columns=["observation.object.visible"]).column(0).to_numpy(zero_copy_only=False).astype(bool)
            v1, v2 = vis(T.TIER1), vis(T.TIER2)
            p1, q1 = T.fill_invisible(*T.load(d / T.TIER1 / name), v1)
            p2, q2 = T.fill_invisible(*T.load(d / T.TIER2 / name), v2)
            segs = [] if thresh is None else rest_segments(p2, thresh)
            rest_frac.append(sum(b - a for a, b in segs) / len(p2))
            pa, qa = snap(p2, q2, segs)
            rows.append(T.episode_metrics(pa[v1], qa[v1], p1[v1], q1[v1]))
        key = "raw" if thresh is None else f"rest<{thresh}m/s"
        out[key] = {
            "rest_frame_frac": float(np.mean(rest_frac)),
            "sp_sr": float(np.mean([r["sp_sr"] for r in rows])),
            "mp_sr": float(np.mean([r["mp_sr"] for r in rows])),
            "mppe_cm": float(np.mean([r["mppe_cm"] for r in rows])),
            "pos_cm_mean": float(np.mean([r["pos_cm"] for r in rows])),
            "rot_deg_mean": float(np.mean([r["rot_deg"] for r in rows])),
        }
    Path("results").mkdir(exist_ok=True)
    Path("results/tier2_rest_snap.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
