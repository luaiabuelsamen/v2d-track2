"""Per-clip SP-SR and MP-SR margins of a packed Tier 1 entry.

A success rate is a gate, not a metric, and the two behave differently. SP-SR passes when the
time-mean position error is <= 0.1 m *and* the time-mean geodesic orientation error is <= 0.5 rad
(kit `SP-SR.py:132-141`); MP-SR uses 0.03 m and 30 deg. A clip can therefore sit a thirtieth of a
radian from a board point while its AUC says nothing is wrong, which is how e16 -- 1.3 degrees from
flipping SP-SR -- survived every arm this project judged on AUC.

Reading the clean reference is legal on Tier 1 and only there: `config/tracks.json` gives the
released multiview trajectory as both the permitted input and the configured scoring reference. Do
not point this at a Tier 2 or Tier 3 entry to decide anything.

    python tools/t1/gate_margins.py --dataset-root DATA --kit-root KIT --csv ENTRY.csv [-o OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v2d.score.track2 import build_frames

POSITION = ["pos_x", "pos_y", "pos_z"]
QUATERNION = ["quat_x", "quat_y", "quat_z", "quat_w"]
SP_POSITION_M = 0.1
SP_ORIENTATION_RAD = 0.5
MP_POSITION_M = 0.03
MP_ORIENTATION_DEG = 30.0


def geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The kit's own quaternion distance, double-cover folded away."""
    dot = np.clip(np.abs((a * b).sum(axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def margins(solution: pd.DataFrame, submission: pd.DataFrame) -> list[dict[str, object]]:
    frame = solution.merge(submission, on="row_id", suffixes=("", "_sub"))
    episode = frame["row_id"].str.extract(r"e(\d{6})")[0].astype(int)
    rows: list[dict[str, object]] = []
    for index in sorted(episode.unique()):
        mask = episode == index
        position = np.linalg.norm(
            frame.loc[mask, POSITION].to_numpy(float)
            - frame.loc[mask, ["ref_" + c for c in POSITION]].to_numpy(float),
            axis=-1,
        ).mean()
        orientation = geodesic_angle(
            frame.loc[mask, QUATERNION].to_numpy(float),
            frame.loc[mask, ["ref_" + c for c in QUATERNION]].to_numpy(float),
        ).mean()
        rows.append(
            {
                "clip": f"e{index:02d}",
                "pos_m": float(position),
                "ori_rad": float(orientation),
                "spider_sr": bool(position <= SP_POSITION_M and orientation <= SP_ORIENTATION_RAD),
                "maniptrans_sr": bool(
                    position < MP_POSITION_M and np.degrees(orientation) < MP_ORIENTATION_DEG
                ),
            }
        )
    return rows


def describe(row: dict[str, object]) -> str:
    position, orientation = float(row["pos_m"]), float(row["ori_rad"])
    missed = []
    if not row["spider_sr"]:
        if position > SP_POSITION_M:
            missed.append(f"SP pos by {100 * (position - SP_POSITION_M):.2f} cm")
        if orientation > SP_ORIENTATION_RAD:
            missed.append(f"SP ori by {orientation - SP_ORIENTATION_RAD:.3f} rad")
    if not row["maniptrans_sr"]:
        if position >= MP_POSITION_M:
            missed.append(f"MP pos by {100 * (position - MP_POSITION_M):.2f} cm")
        if np.degrees(orientation) >= MP_ORIENTATION_DEG:
            missed.append(f"MP ori by {np.degrees(orientation) - MP_ORIENTATION_DEG:.1f} deg")
    return "; ".join(missed) if missed else "passes both"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--kit-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True, help="a packed Tier 1 submission")
    parser.add_argument("--grid", default="sample")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    solution, submission, _ = build_frames(
        args.dataset_root, args.kit_root, f"csv:{args.csv}", "tier1", args.grid, None
    )
    rows = margins(solution, submission)
    rows.sort(key=lambda r: (r["spider_sr"], r["ori_rad"]))

    print(f"{'clip':5s} {'pos cm':>7s} {'ori rad':>8s} {'SP':>5s} {'MP':>5s}   what it misses")
    for row in rows:
        print(
            f"{row['clip']:5s} {100 * float(row['pos_m']):7.2f} {float(row['ori_rad']):8.3f} "
            f"{row['spider_sr']!s:>5s} {row['maniptrans_sr']!s:>5s}   {describe(row)}"
        )
    passes = sum(bool(r["spider_sr"]) for r in rows)
    manip = sum(bool(r["maniptrans_sr"]) for r in rows)
    print(f"\nSP-SR {passes}/{len(rows)}   MP-SR {manip}/{len(rows)}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
