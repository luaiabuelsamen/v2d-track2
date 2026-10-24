"""Aggregate the retargeter's own diagnostics across every retargeted clip.

`validate_retarget.py` proves the object trajectory survived the retarget. It says nothing
about whether the *robot* motion is usable, and that is what FlashCHORD actually trains on. The
retargeter prints the numbers that matter to its log, so this reads them back:

- **hand contact coverage.** The G1 recipe rewards tracking a grasped object. A manipulation
  clip whose retarget found contact on almost no frames gives the policy nothing to hold, so a
  near-zero fraction is a red flag, not a curiosity.
- **object frame-0 ground correction** and **plane-leveling tilt**: how far the released scene
  had to move to sit on z=0. A large correction means the reconstructed plane and the object
  disagree.
- **sole Z range** and **anti-penetration lift**: whether the robot ends up standing on the
  floor rather than through it or above it.

When both tiers are present it also pairs each episode across them. The same clip retargeted
from Tier 1's clean object trajectory and from Tier 2's noisy one differs only by the injected
noise, so the drop in contact coverage measures how much of the grasp signal that noise costs.

A clip whose task manipulates the object with something other than the hands (the release has
one: `rolling_luggage_left_foot_push`) is expected to show zero hand contact, so the flag is a
prompt to read the task name, not a verdict.

Usage:
  python scripts/retarget_qa.py --log-dir DIR [-o results/retarget_qa.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PATTERNS = {
    "ground_correction_min_m": re.compile(
        r"object frame-0 ground correction: minimum=([-+0-9.eE]+) m"
    ),
    "ground_correction_applied_m": re.compile(r"ground correction:.*applied=([-+0-9.eE]+) m"),
    "plane_tilt_deg": re.compile(r"plane leveling: tilt=([-+0-9.eE]+) deg"),
    "plane_translation_z_m": re.compile(r"plane leveling:.*translation_z=([-+0-9.eE]+) m"),
    "sole_z_min_m": re.compile(r"Saved robot sole Z range: \[([-+0-9.eE]+),"),
    "sole_z_max_m": re.compile(r"Saved robot sole Z range: \[[-+0-9.eE]+, *([-+0-9.eE]+)\]"),
    "foot_offset_min_m": re.compile(r"foot-target Z offset range: \[([-+0-9.eE]+),"),
    "foot_offset_max_m": re.compile(r"foot-target Z offset range: \[[-+0-9.eE]+, *([-+0-9.eE]+)\]"),
}
CONTACT = re.compile(r"(left|right)_hand_contact_active: (\d+)/(\d+) frames")
ANTIPEN = re.compile(
    r"anti-penetration lift: \[([-+0-9.eE]+), *([-+0-9.eE]+)\] m; (\d+)/(\d+) frames"
)
GROUND_SOURCE = re.compile(r"grounding plane: (\w+)")
SEQUENCE_ID = re.compile(r"sequence_id=([^ /\n]+)")


def parse(log: Path) -> dict | None:
    text = log.read_text(errors="replace")
    if "Retargeting complete" not in text:
        return None
    record: dict = {}
    for key, pattern in PATTERNS.items():
        found = pattern.findall(text)
        record[key] = float(found[-1]) if found else None
    source = GROUND_SOURCE.findall(text)
    record["grounding_plane"] = source[-1] if source else None
    sequence = SEQUENCE_ID.findall(text)
    record["sequence_id"] = sequence[0] if sequence else None

    contacts = {}
    for side, active, total in CONTACT.findall(text):
        contacts[side] = {
            "frames": int(total),
            "active": int(active),
            "fraction": round(int(active) / int(total), 4) if int(total) else None,
        }
    record["contact"] = contacts
    fractions = [v["fraction"] for v in contacts.values() if v["fraction"] is not None]
    record["contact_fraction_either"] = round(max(fractions), 4) if fractions else None

    lift = ANTIPEN.findall(text)
    if lift:
        _low, high, frames, total = lift[-1]
        record["antipenetration_lift_max_m"] = float(high)
        record["antipenetration_frames"] = f"{frames}/{total}"
    return record


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument(
        "--contact-floor",
        type=float,
        default=0.02,
        help="flag a clip whose best hand-contact fraction is below this",
    )
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    clips = {}
    for log in sorted(args.log_dir.glob("tier*_e*.log")):
        parsed = parse(log)
        if parsed:
            clips[log.stem] = parsed
    if not clips:
        raise SystemExit(f"no completed retarget logs under {args.log_dir}")

    def column(key: str) -> list[float]:
        return [v[key] for v in clips.values() if v.get(key) is not None]

    flagged = sorted(
        name
        for name, v in clips.items()
        if v.get("contact_fraction_either") is not None
        and v["contact_fraction_either"] < args.contact_floor
    )
    paired = {}
    for name, value in clips.items():
        tier, _, episode = name.partition("_")
        paired.setdefault(episode, {})[tier] = value.get("contact_fraction_either")
    both = {
        e: v for e, v in paired.items() if v.get("tier1") is not None and v.get("tier2") is not None
    }
    tier_noise = None
    if both:
        deltas = {e: round(v["tier2"] - v["tier1"], 4) for e, v in sorted(both.items())}
        lost_half = sorted(
            e for e, v in both.items() if v["tier1"] > 0.05 and v["tier2"] < 0.5 * v["tier1"]
        )
        tier_noise = {
            "paired_episodes": len(both),
            "contact_mean_tier1": round(float(np.mean([v["tier1"] for v in both.values()])), 4),
            "contact_mean_tier2": round(float(np.mean([v["tier2"] for v in both.values()])), 4),
            "contact_mean_delta": round(float(np.mean(list(deltas.values()))), 4),
            "episodes_degraded": sum(1 for d in deltas.values() if d < -0.01),
            "episodes_lost_over_half_of_contact": lost_half,
            "delta_per_episode": deltas,
        }

    summary = {
        "clips": len(clips),
        "grounding_plane_sources": {
            s: sum(1 for v in clips.values() if v.get("grounding_plane") == s)
            for s in {v.get("grounding_plane") for v in clips.values()}
        },
        "contact_fraction_either_mean": round(float(np.mean(column("contact_fraction_either"))), 4),
        "contact_fraction_either_min": round(float(np.min(column("contact_fraction_either"))), 4),
        "clips_below_contact_floor": flagged,
        "ground_correction_applied_max_m": round(
            float(np.max(np.abs(column("ground_correction_applied_m")))), 5
        ),
        "plane_tilt_deg_max": round(float(np.max(column("plane_tilt_deg"))), 3),
        "sole_z_min_m_worst": round(float(np.min(column("sole_z_min_m"))), 5),
        "sole_z_max_m_worst": round(float(np.max(column("sole_z_max_m"))), 5),
        "tier2_noise_contact_cost": tier_noise,
    }
    print(json.dumps(summary, indent=2))
    print("\nper clip: contact fraction (left/right), plane tilt, sole z range")
    for name, value in sorted(clips.items()):
        left = value["contact"].get("left", {}).get("fraction")
        right = value["contact"].get("right", {}).get("fraction")
        task = (value.get("sequence_id") or "")[20:]
        print(
            f"  {name:14s} L={left if left is None else f'{left:.3f}'} "
            f"R={right if right is None else f'{right:.3f}'}"
            f"  tilt={value.get('plane_tilt_deg')}deg"
            f"  sole=[{value.get('sole_z_min_m')}, {value.get('sole_z_max_m')}]  {task}"
        )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"summary": summary, "clips": clips}, indent=2, sort_keys=True) + "\n"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
