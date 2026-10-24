"""Hand-object contact coverage: the training signal the scored metrics do not measure.

The retargeter logs, per clip, how many frames each hand is within 2 cm of the object. That
fraction is what a policy actually has to learn from, and it moves independently of AUC: over the
12 episodes the support-surface extension changed, per-clip AUC gain and contact change correlate
at r = +0.481, yet the mean AUC gain is +0.0237 against a mean contact change of -0.0060.

Coverage per clip is ``max(left, right)``, not the sum or the mean: a one-handed task has one
hand on the object throughout and the other doing nothing, and averaging the two would score that
as half.

This extraction was duplicated in three separate scripts before the package existed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Final

import numpy as np

CONTACT: Final[re.Pattern[str]] = re.compile(
    r"(left|right)_hand_contact_active: (\d+)/(\d+) frames"
)
COMPLETE: Final[str] = "Retargeting complete"


def coverage(log_dir: Path, prefix: str = "tier*") -> dict[int, float]:
    """episode -> contact coverage, for every clip that finished under `log_dir`."""
    out: dict[int, float] = {}
    for log in sorted(Path(log_dir).glob(f"{prefix}_e*.log")):
        text = log.read_text(errors="replace")
        if COMPLETE not in text:
            continue
        found = CONTACT.findall(text)
        if not found:
            continue
        sides = {side: (int(active), int(total)) for side, active, total in found}
        frames = max(total for _, total in sides.values())
        if frames:
            out[int(log.stem.split("_e")[-1])] = (
                max(active for active, _ in sides.values()) / frames
            )
    return out


def paired(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    """``a - b`` on the episodes both contain."""
    return {episode: a[episode] - b[episode] for episode in sorted(set(a) & set(b))}


def sign_test(differences: dict[int, float], tolerance: float = 1e-9) -> dict[str, float | int]:
    """Exact two-sided sign test on paired differences, ignoring ties.

    Contact's per-clip sign is unreliable -- the floor-only cleaner moved 15 clips up and 10 down
    for a +0.0141 mean -- so a paired mean without this is an overclaim.
    """
    up = sum(1 for value in differences.values() if value > tolerance)
    down = sum(1 for value in differences.values() if value < -tolerance)
    total = up + down
    if total == 0:
        return {"up": 0, "down": 0, "tied": len(differences), "p_value": 1.0}

    def mass(k: int) -> float:
        return math.comb(total, k) * 0.5**total

    threshold = mass(up) + 1e-15
    p_value = sum(mass(k) for k in range(total + 1) if mass(k) <= threshold)
    return {
        "up": up,
        "down": down,
        "tied": len(differences) - total,
        "p_value": round(min(p_value, 1.0), 4),
    }


def mean_over(values: dict[int, float], episodes: list[int]) -> float:
    return float(np.mean([values[episode] for episode in episodes]))


def ladder(series: dict[str, dict[int, float]]) -> dict[str, float]:
    """Mean coverage per series over the episodes every series shares."""
    populated = [value for value in series.values() if value]
    if not populated:
        return {}
    shared = sorted(set.intersection(*(set(value) for value in populated)))
    return {name: round(mean_over(value, shared), 4) for name, value in series.items() if value}


def recovery(achieved: float, baseline: float, ceiling: float) -> float | None:
    """What fraction of the gap from `baseline` to `ceiling` an arm closed."""
    span = ceiling - baseline
    if abs(span) < 1e-9:
        return None
    return round((achieved - baseline) / span, 4)
