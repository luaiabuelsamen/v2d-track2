"""The Tier 2 production path must not read the tier_1 tree at all.

The organisers' final checklist (2026-11-01) says "Submissions for Tiers 2 and 3 must use the
appropriate reference trajectory for each tier", and that submissions which do not adhere
"will be invalidated". Our Tier 2 entry is defensible on the trajectory -- the cleaner reads
`object_pose(..., "tier2", ...)` and the per-clip gate is our own synthetic recovery gate, with
no truth anywhere in it -- but it used to take the static support-surface geometry from the
tier_1 directory, which is an argument nobody needs to have. Both trees ship byte-identical
support_surface USDAs for all 30 clips, so reading the Tier 2 copy changes no number.

This pins the separation by source, because it is the kind of thing a later edit restores
silently: the scoring and gate-construction modules may read tier_1 (that is how a cleaner is
measured), the production cleaner may not.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "v2d" / "clean"
TIER1 = re.compile(r"""tier_root\(\s*dataset_root\s*,\s*["']tier1["']\s*\)|["']tier_1_""")

# gates.py builds the known-answer test and measures recovery against the clean trajectory;
# its own docstring records that Tier 1 "is never an input or a target for anything that runs
# on Tier 2". tilt_yaw.py and root_planting.py are the paired-comparison experiments.
MAY_READ_TIER1 = {"gates.py", "tilt_yaw.py", "root_planting.py"}


def test_the_production_cleaner_reads_no_tier1_path() -> None:
    offenders = {}
    for path in sorted(SRC.glob("*.py")):
        if path.name in MAY_READ_TIER1:
            continue
        hits = [
            f"{path.name}:{n}"
            for n, line in enumerate(path.read_text().splitlines(), 1)
            if TIER1.search(line) and not line.lstrip().startswith("#")
        ]
        if hits:
            offenders[path.name] = hits
    assert not offenders, f"Tier 2 production path reads tier_1: {offenders}"


def test_support_contact_reads_tier2_support_geometry() -> None:
    source = (SRC / "support_contact.py").read_text()
    assert 'tier_root(dataset_root, "tier2") / episode.support_surface' in source
    assert 'object_pose(dataset_root, "tier2"' in source, "trajectory source must be Tier 2"
