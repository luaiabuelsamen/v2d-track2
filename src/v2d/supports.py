"""The released `support_surface/` primitives, and matching a rest segment to one.

The primitives live in a Z-up `g1_retarget_world`; the object poses are in a Y-up frame; no
transform between the two ships. But a **height above the floor is a scalar**, invariant to any
transform that preserves up and slides within the floor plane, so a primitive's top-face height
is directly comparable to a height measured from the released floor plane. That is what makes
them usable at all, and it checks out: of the 96 non-floor rest segments the floor-only gate
refuses, 73 land within 2 cm of a horizontal primitive top at a median gap of 0.61 cm
(`v2d probe supports`). Sub-centimetre agreement over 73 segments in 13 episodes is also the
evidence that the two frames share an up axis and that the Z-up frame's floor is at z = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

AXIS_INDEX: Final[dict[str, int]] = {"X": 0, "Y": 1, "Z": 2}

#: How far above its support a resting object's lowest vertex may sit. An object at rest touches.
ABOVE_TOLERANCE_CM: Final[float] = 3.0
#: How far below a *raised* support a segment may sit and still match it. Penetration is real --
#: a tilt error is what digs a corner through a surface, and `traffic_cone` sits 2.3 cm under its
#: table -- but below a raised surface the floor is down there, so it must be bounded. Unbounded
#: below the floor, where nothing is further down. Safe because the lowest table top in any
#: episode is 46.9 cm up, far outside this window.
BELOW_TOLERANCE_CM: Final[float] = 10.0


@dataclass(frozen=True)
class Primitive:
    """One support primitive, with the height of its top face above the Z-up frame's z = 0."""

    name: str
    kind: str
    centre: tuple[float, float, float]
    half_extent: tuple[float, float, float]
    top_height_m: float
    horizontal: bool
    footprint_m2: float


def read_primitives(path: Path) -> list[Primitive]:
    """Every Cube, Cylinder or Sphere in a `_support.usda`."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    out: list[Primitive] = []
    for prim in stage.Traverse():
        kind = str(prim.GetTypeName())
        if kind not in ("Cylinder", "Cube", "Sphere"):
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        centre = np.array(transform.ExtractTranslation(), dtype=np.float64)
        if kind == "Cylinder":
            cylinder = UsdGeom.Cylinder(prim)
            local = np.full(3, float(cylinder.GetRadiusAttr().Get()))
            local[AXIS_INDEX[str(cylinder.GetAxisAttr().Get())]] = (
                float(cylinder.GetHeightAttr().Get()) / 2.0
            )
        elif kind == "Cube":
            local = np.full(3, float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2.0)
        else:
            local = np.full(3, float(UsdGeom.Sphere(prim).GetRadiusAttr().Get()))
        scale = np.linalg.norm(np.array(transform, dtype=np.float64)[:3, :3], axis=1)
        half = local * scale
        out.append(
            Primitive(
                name=str(prim.GetName()),
                kind=kind,
                centre=tuple(float(v) for v in centre),
                half_extent=tuple(float(v) for v in half),
                top_height_m=float(centre[2] + half[2]),
                # A surface you rest ON is thin in z. A primitive thin in x or y is a panel you
                # lean against, which pins no tilt -- `vacuum`'s only primitive is one of those.
                horizontal=bool(np.argmin(half) == 2),
                footprint_m2=float(4 * half[0] * half[1]),
            )
        )
    return out


def candidate_heights(path: Path | None) -> list[float]:
    """Support heights above the floor: the floor itself, plus every horizontal primitive top."""
    if path is None:
        return [0.0]
    return [0.0, *sorted(p.top_height_m for p in read_primitives(path) if p.horizontal)]


def match_support(
    height: float,
    candidates: list[float],
    above: float = ABOVE_TOLERANCE_CM / 100.0,
    below: float = BELOW_TOLERANCE_CM / 100.0,
) -> float | None:
    """The candidate support height a rest segment at `height` sits on, or None if none fits.

    Asymmetric by design; see the module constants. Unbounded penetration below the floor,
    bounded below a raised surface.
    """
    best = min(candidates, key=lambda candidate: abs(height - candidate))
    gap = height - best
    if gap > above:
        return None
    if best > 0.0 and gap < -below:
        return None
    return best
