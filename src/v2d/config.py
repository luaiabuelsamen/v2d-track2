"""Shared constants: dataset layout, and the resource floors this machine runs under."""

from __future__ import annotations

from typing import Final

#: Released tier directory names, keyed by the short name used on the command line.
TIER_DIRS: Final[dict[str, str]] = {
    "tier1": "tier_1_multiview_caption",
    "tier2": "tier_2_synthetic_noise",
}
TIER1: Final[str] = TIER_DIRS["tier1"]
TIER2: Final[str] = TIER_DIRS["tier2"]

#: The released SOMA rig has 78 joints including "Root"; exports carry the other 77.
EXPECTED_JOINTS: Final[int] = 77

#: Capture rate of every released episode.
FPS: Final[float] = 30.0

#: Resource floors for batch work on this 15.6 GB Jetson. A batch aborts rather than thrash.
DISK_FLOOR_GB: Final[float] = 20.0
MEM_FLOOR_MB: Final[float] = 2500.0

#: Two workers is the measured ceiling for concurrent retargeting here: three lost 28 clips of
#: 60 to the OOM killer, and even two lose clips when other work competes for memory.
MAX_RETARGET_WORKERS: Final[int] = 2

#: Episode 29 manipulates its object with a foot, so it has ~0 hand contact in every arm by
#: construction. It stays in every mean and carries no conclusion on its own.
FOOT_EPISODE: Final[int] = 29
