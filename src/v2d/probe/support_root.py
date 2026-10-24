"""The measurements that ground docs/EXP_SUPPORT_AND_ROOT.md, taken before predicting.

Three questions, none of which needs a GPU or a clean reference:

1. Are the released `support_surface/` primitives usable despite the missing Z-up -> Y-up
   transform? A height above the floor is a scalar, invariant to any transform preserving up and
   sliding within the floor plane, so a primitive's top-face height is directly comparable to the
   lowest-vertex height `clean_support_contact.py` already measures from noisy data.
2. How many rest segments would a primitive-gated cleaner newly accept, and in which episodes?
3. Is Tier 2's root perturbation a constant per-clip offset (one frame identifies it) or
   time-varying (a per-frame estimator is needed)?

Usage:
  python scripts/probe_support_root.py --dataset-root DATA --cleaner-results FILE \
      [-o results/probe_support_root.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation

TIER1, TIER2 = "tier_1_multiview_caption", "tier_2_synthetic_noise"
AXIS = {"X": 0, "Y": 1, "Z": 2}
# A rest segment matches a candidate support height when it sits no more than ABOVE_CM above it
# and no more than BELOW_CM below. Penetration is allowed because a tilt error is what digs a
# corner through a surface; the bound is safe because candidate heights are >= 46 cm apart.
ABOVE_CM, BELOW_CM = 3.0, 10.0


def primitives(path: Path) -> list[dict]:
    """Every support primitive in a `_support.usda`, with its top-face height."""
    stage = Usd.Stage.Open(str(path))
    out = []
    for prim in stage.Traverse():
        kind = prim.GetTypeName()
        if kind not in ("Cylinder", "Cube", "Sphere"):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centre = np.array(transform.ExtractTranslation(), dtype=np.float64)
        if kind == "Cylinder":
            cylinder = UsdGeom.Cylinder(prim)
            half = float(cylinder.GetHeightAttr().Get()) / 2.0
            radius = float(cylinder.GetRadiusAttr().Get())
            local = np.full(3, radius)
            local[AXIS[str(cylinder.GetAxisAttr().Get())]] = half
            shape = (
                f"Cylinder(axis={cylinder.GetAxisAttr().Get()}, h={half * 2:.3f}, r={radius:.3f})"
            )
        elif kind == "Cube":
            size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
            local = np.full(3, size / 2.0)
            shape = f"Cube(size={size:.3f})"
        else:
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get())
            local = np.full(3, radius)
            shape = f"Sphere(r={radius:.3f})"
        # rows of the 3x3 carry the per-local-axis scale
        scale = np.linalg.norm(np.array(transform, dtype=np.float64)[:3, :3], axis=1)
        half_extent = local * scale
        out.append(
            {
                "prim": prim.GetName(),
                "type": kind,
                "shape": shape,
                "centre": [round(float(v), 4) for v in centre],
                "half_extent": [round(float(v), 4) for v in half_extent],
                "top_height_m": round(float(centre[2] + half_extent[2]), 4),
                # a surface you rest ON is thin in z; a panel you lean against is thin horizontally
                "horizontal": bool(np.argmin(half_extent) == 2),
                "footprint_m2": round(float(4 * half_extent[0] * half_extent[1]), 4),
            }
        )
    return out


def support_probe(dataset_root: Path, cleaner: dict) -> dict:
    root = dataset_root / "track_2" / TIER1
    meta = [
        json.loads(line)
        for line in (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    meta.sort(key=lambda r: r["episode_index"])
    rows, gaps = [], []
    accepted_today = newly_accepted = still_refused = 0
    for row in meta:
        episode = row["episode_index"]
        prims = primitives(root / row["support_surface"])
        record = cleaner[str(episode)] if str(episode) in cleaner else cleaner[episode]
        heights = record["support_heights_cm"]
        tops = [round(p["top_height_m"] * 100, 2) for p in prims if p["horizontal"]]
        candidates = [0.0, *tops]
        floor, new, refused = 0, 0, 0
        for height in heights:
            nearest = min(candidates, key=lambda c: abs(height - c))
            matched = -BELOW_CM <= height - nearest <= ABOVE_CM
            if matched and nearest == 0.0:
                floor += 1
            elif matched:
                new += 1
            else:
                refused += 1
        nonfloor = [h for h in heights if h > ABOVE_CM]
        gaps += [round(min(abs(h - t) for t in tops), 2) for h in nonfloor] if tops else []
        accepted_today += floor
        newly_accepted += new
        still_refused += refused
        rows.append(
            {
                "episode": episode,
                "sequence": row["sequence_id"],
                "object": row["object"],
                "primitives": prims,
                "horizontal_tops_cm": tops,
                "rest_heights_cm": heights,
                "rest_segments": len(heights),
                "segments_floor": floor,
                "segments_newly_accepted": new,
                "segments_refused": refused,
                "unlocked_by_extension": bool(floor == 0 and new > 0),
            }
        )
    within2 = sum(1 for g in gaps if g <= 2.0)
    return {
        "question": "are the released support_surface primitives usable without the "
        "missing transform",
        "nonfloor_segments": len(gaps),
        "nonfloor_within_2cm_of_a_horizontal_top": within2,
        "nonfloor_within_5cm": sum(1 for g in gaps if g <= 5.0),
        "gap_median_cm": round(float(np.median(gaps)), 2) if gaps else None,
        "gap_max_cm": round(float(np.max(gaps)), 2) if gaps else None,
        "episodes_with_a_primitive": sum(1 for r in rows if r["primitives"]),
        "segments_accepted_today": accepted_today,
        "segments_newly_accepted": newly_accepted,
        "segments_still_refused": still_refused,
        "episodes_unlocked": [r["episode"] for r in rows if r["unlocked_by_extension"]],
        # The safety claim for the BELOW_CM window is that a floor segment can never be
        # mistaken for a table segment, so the statistic that matters is the floor-to-lowest-top
        # gap. Two tops in one episode can sit arbitrarily close (e25 has two 0.36 cm apart) and
        # confusing those two is harmless: they are the same plane.
        "minimum_floor_to_top_gap_cm": round(
            min(
                (min(r["horizontal_tops_cm"]) for r in rows if r["horizontal_tops_cm"]), default=0.0
            ),
            2,
        ),
        "minimum_gap_between_two_tops_cm": round(
            min(
                (
                    min(
                        abs(a - b)
                        for i, a in enumerate(r["horizontal_tops_cm"])
                        for b in r["horizontal_tops_cm"][i + 1 :]
                    )
                    for r in rows
                    if len(r["horizontal_tops_cm"]) > 1
                ),
                default=None,
            ),
            2,
        ),
        "per_episode": rows,
    }


def root_probe(dataset_root: Path) -> dict:
    data = dataset_root / "track_2"
    meta = [
        json.loads(line)
        for line in (data / TIER1 / "meta" / "episodes_metadata.jsonl").read_text().splitlines()
        if line.strip()
    ]
    meta.sort(key=lambda r: r["episode_index"])
    columns = ["observation.human.pose", "observation.human.translation"]
    rows = []
    for row in meta:
        episode = row["episode_index"]
        clean = pq.read_table(
            data / TIER1 / f"data/chunk-000/episode_{episode:06d}.parquet", columns=columns
        )
        noisy = pq.read_table(
            data / TIER2 / f"data/chunk-000/episode_{episode:06d}.parquet", columns=columns
        )
        pose_c = np.asarray(clean[columns[0]].to_pylist(), dtype=np.float64)
        pose_n = np.asarray(noisy[columns[0]].to_pylist(), dtype=np.float64)
        pose_c = pose_c.reshape(len(pose_c), -1, 3)
        pose_n = pose_n.reshape(len(pose_n), -1, 3)
        delta = Rotation.from_rotvec(pose_n[:, 0]) * Rotation.from_rotvec(pose_c[:, 0]).inv()
        degrees = np.degrees(delta.magnitude())
        constant = delta.mean()
        residual = np.degrees((delta * constant.inv()).magnitude())
        shift = np.asarray(noisy[columns[1]].to_pylist(), dtype=np.float64) - np.asarray(
            clean[columns[1]].to_pylist(), dtype=np.float64
        )
        rows.append(
            {
                "episode": episode,
                "frames": len(pose_c),
                "joints": int(pose_c.shape[1]),
                "root_err_deg_mean": round(float(degrees.mean()), 4),
                "root_err_deg_max": round(float(degrees.max()), 4),
                "root_constant_deg": round(float(np.degrees(constant.magnitude())), 4),
                "root_residual_deg_mean": round(float(residual.mean()), 4),
                "constant_fraction": round(1.0 - float(residual.mean() / degrees.mean()), 4),
                "transl_err_cm_mean": round(float(np.linalg.norm(shift, axis=-1).mean() * 100), 4),
                "transl_residual_cm_mean": round(
                    float(np.linalg.norm(shift - shift.mean(axis=0), axis=-1).mean() * 100), 4
                ),
                "nonroot_joints_bit_identical": bool(np.array_equal(pose_c[:, 1:], pose_n[:, 1:])),
            }
        )
    mean = lambda key: float(np.mean([r[key] for r in rows]))  # noqa: E731
    return {
        "question": "is Tier 2's root perturbation a constant per-clip offset or time-varying",
        "root_err_deg_mean": round(mean("root_err_deg_mean"), 4),
        "root_err_deg_max": round(max(r["root_err_deg_max"] for r in rows), 4),
        "root_residual_deg_mean": round(mean("root_residual_deg_mean"), 4),
        "root_constant_share": round(
            1 - mean("root_residual_deg_mean") / mean("root_err_deg_mean"), 4
        ),
        "transl_err_cm_mean": round(mean("transl_err_cm_mean"), 4),
        "transl_constant_share": round(
            1 - mean("transl_residual_cm_mean") / mean("transl_err_cm_mean"), 4
        ),
        "all_nonroot_joints_bit_identical": all(r["nonroot_joints_bit_identical"] for r in rows),
        "per_episode": rows,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--cleaner-results",
        type=Path,
        default=Path("results/clean_support_contact.json"),
        help="the floor-only cleaner's record, for its measured rest heights",
    )
    parser.add_argument("-o", "--out", type=Path, default=Path("results/probe_support_root.json"))
    args = parser.parse_args(argv)

    cleaner = json.loads(args.cleaner_results.read_text())["per_episode"]
    support = support_probe(args.dataset_root, cleaner)
    root = root_probe(args.dataset_root)

    print("=== support primitives ===")
    print(f"episodes with a primitive            : {support['episodes_with_a_primitive']}/30")
    print(f"non-floor rest segments              : {support['nonfloor_segments']}")
    print(
        "  within 2 cm of a horizontal top    : "
        f"{support['nonfloor_within_2cm_of_a_horizontal_top']}"
        f" (median gap {support['gap_median_cm']} cm)"
    )
    print(f"rest segments accepted today         : {support['segments_accepted_today']}")
    print(f"  newly accepted by the extension    : {support['segments_newly_accepted']}")
    print(f"  still refused                      : {support['segments_still_refused']}")
    print(f"episodes unlocked outright           : {support['episodes_unlocked']}")
    print(f"lowest table top above the floor     : {support['minimum_floor_to_top_gap_cm']} cm")
    print(f"closest two tops within one episode  : {support['minimum_gap_between_two_tops_cm']} cm")
    print("\n=== root perturbation ===")
    print(
        f"root rotation error  : {root['root_err_deg_mean']:.3f} deg mean, "
        f"{root['root_err_deg_max']:.3f} max"
    )
    print(f"after removing the best constant: {root['root_residual_deg_mean']:.3f} deg")
    print(
        f"=> constant share    : {root['root_constant_share']:.1%} rotation, "
        f"{root['transl_constant_share']:.1%} translation"
    )
    print(f"translation error    : {root['transl_err_cm_mean']:.3f} cm")
    print(f"all non-root joints bit-identical: {root['all_nonroot_joints_bit_identical']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "what": "measurements taken before predicting, for docs/EXP_SUPPORT_AND_ROOT.md",
                "support_surfaces": support,
                "root_perturbation": root,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
