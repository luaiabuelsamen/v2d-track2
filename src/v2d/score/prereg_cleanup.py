"""Score the pre-registered predictions in docs/EXP_CLEANUP_CONTACT.md.

Written and committed before any arm finished, so the scoring could not be tuned to the
numbers it would see. It reads each arm's retarget logs, extracts the same hand-object contact
coverage `retarget_qa.py` already parses, and checks P1-P6 against the thresholds fixed in the
pre-registration. Nothing here decides anything a human would not; the decision rule is applied
mechanically and its verdict printed.

Contact coverage per clip = max(left, right) of `{side}_hand_contact_active: n/T`, the
retargeter's own 2 cm threshold. Arms are paired by episode index.

Usage:
  python scripts/score_cleanup_prediction.py --arms-root DIR --tier1-logs DIR --tier2-logs DIR \
      [-o results/cleanup_contact.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CONTACT = re.compile(r"(left|right)_hand_contact_active: (\d+)/(\d+) frames")
COMPLETE = "Retargeting complete"
# Episode 29 manipulates the object with a foot, so it has ~0 hand contact in every arm by
# construction. Kept in the mean, reported separately, never allowed to carry a conclusion.
FOOT_EPISODE = 29
# The 8 clips that lost over half their contact to Tier 2 noise (results/retarget_qa.json).
WORST = [0, 1, 14, 15, 18, 19, 22, 28]
ARMS = ["tilt_fixed", "rot_clean", "yaw_fixed", "object_clean"]


def coverage(log_dir: Path) -> dict[int, float]:
    """episode index -> contact coverage, for every completed clip under `log_dir`."""
    out: dict[int, float] = {}
    for log in sorted(log_dir.glob("tier*_e*.log")):
        text = log.read_text(errors="replace")
        if COMPLETE not in text:
            continue
        found = CONTACT.findall(text)
        if not found:
            continue
        sides: dict[str, tuple[int, int]] = {}
        for side, active, total in found:
            sides[side] = (int(active), int(total))
        frames = max(v[1] for v in sides.values())
        if not frames:
            continue
        episode = int(log.stem.split("_e")[-1])
        out[episode] = max(v[0] for v in sides.values()) / frames
    return out


def mean(values: dict[int, float], episodes: list[int]) -> float:
    return float(np.mean([values[e] for e in episodes]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms-root", type=Path, required=True, help="holds <arm>/logs per arm")
    parser.add_argument("--tier1-logs", type=Path, required=True)
    parser.add_argument("--tier2-logs", type=Path, required=True)
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    # The baselines live in the main run's log dir, where tier1 and tier2 share one directory,
    # so they are split by filename prefix below rather than by directory.
    # tier1_*.log and tier2_*.log sit together; split by filename prefix.
    def by_prefix(log_dir: Path, prefix: str) -> dict[int, float]:
        out = {}
        for log in sorted(log_dir.glob(f"{prefix}_e*.log")):
            text = log.read_text(errors="replace")
            if COMPLETE not in text:
                continue
            found = CONTACT.findall(text)
            if not found:
                continue
            sides = {side: (int(a), int(n)) for side, a, n in found}
            frames = max(v[1] for v in sides.values())
            if frames:
                out[int(log.stem.split("_e")[-1])] = max(v[0] for v in sides.values()) / frames
        return out

    tier1 = by_prefix(args.tier1_logs, "tier1")
    tier2 = by_prefix(args.tier2_logs, "tier2")

    arms = {a: coverage(args.arms_root / a / "logs") for a in ARMS}
    series = {"tier1": tier1, "tier2": tier2, **arms}

    shared = sorted(set.intersection(*(set(v) for v in series.values() if v)))
    missing = {k: sorted(set(tier2) - set(v)) for k, v in series.items() if set(tier2) - set(v)}
    if not shared:
        raise SystemExit("no episode is present in every arm yet")

    means = {k: mean(v, shared) for k, v in series.items()}
    no_foot = [e for e in shared if e != FOOT_EPISODE]
    means_no_foot = {k: mean(v, no_foot) for k, v in series.items()}

    c_t1, c_t2 = means["tier1"], means["tier2"]
    c_rot, c_obj = means["rot_clean"], means["object_clean"]

    def recovery(arm: str, ceiling: float) -> float | None:
        span = ceiling - c_t2
        if abs(span) < 1e-9:
            return None
        return (means[arm] - c_t2) / span

    rec_rot = {a: recovery(a, c_rot) for a in ARMS}
    rec_obj = {a: recovery(a, c_obj) for a in ARMS}

    # --- the pre-registered predictions, checked mechanically ---
    tilt_vs_yaw = [arms["tilt_fixed"][e] - arms["yaw_fixed"][e] for e in shared]
    tilt_gain = {e: arms["tilt_fixed"][e] - tier2[e] for e in shared}
    mean_gain = float(np.mean(list(tilt_gain.values())))
    worst_present = [e for e in WORST if e in shared]
    worst_above_mean = [e for e in worst_present if tilt_gain[e] > mean_gain]

    checks = {
        "P1_ceiling_below_tier1": {
            "claim": "C_object_clean < C_tier1 = 0.5744, point prediction 0.42-0.52",
            "c_object_clean": round(c_obj, 4),
            "c_tier1": round(c_t1, 4),
            "below_tier1": bool(c_obj < c_t1),
            "in_predicted_range": bool(0.42 <= c_obj <= 0.52),
            "pass": bool(c_obj < c_t1),
        },
        "P2_tilt_beats_yaw": {
            "claim": "mean paired (tilt - yaw) >= +0.03 and sign consistent in >= 20/30",
            "mean_paired_delta": round(float(np.mean(tilt_vs_yaw)), 4),
            "episodes_tilt_higher": int(sum(1 for d in tilt_vs_yaw if d > 0)),
            "episodes_total": len(shared),
            "pass": bool(
                np.mean(tilt_vs_yaw) >= 0.03 and sum(1 for d in tilt_vs_yaw if d > 0) >= 20
            ),
        },
        "P3_tilt_recovers_half": {
            "claim": "recovery_rot(tilt_fixed) >= 0.5",
            "recovery_rot": None
            if rec_rot["tilt_fixed"] is None
            else round(rec_rot["tilt_fixed"], 4),
            "pass": bool(rec_rot["tilt_fixed"] is not None and rec_rot["tilt_fixed"] >= 0.5),
        },
        "P4_yaw_recovers_little": {
            "claim": "recovery_rot(yaw_fixed) <= 0.25",
            "recovery_rot": None
            if rec_rot["yaw_fixed"] is None
            else round(rec_rot["yaw_fixed"], 4),
            "pass": bool(rec_rot["yaw_fixed"] is not None and rec_rot["yaw_fixed"] <= 0.25),
        },
        "P5_worst_clips_move_most": {
            "claim": "at least 5 of the 8 worst-hit clips gain more than the 30-clip mean gain",
            "mean_gain": round(mean_gain, 4),
            "worst_clips_above_mean": worst_above_mean,
            "count": len(worst_above_mean),
            "pass": bool(len(worst_above_mean) >= 5),
        },
        "P6_arms_bracket_ceiling": {
            "claim": "tilt and yaw both lie between tier2 and rot_clean",
            "c_tier2": round(c_t2, 4),
            "c_tilt_fixed": round(means["tilt_fixed"], 4),
            "c_yaw_fixed": round(means["yaw_fixed"], 4),
            "c_rot_clean": round(c_rot, 4),
            "pass": bool(
                min(c_t2, c_rot) - 1e-9 <= means["tilt_fixed"] <= max(c_t2, c_rot) + 1e-9
                and min(c_t2, c_rot) - 1e-9 <= means["yaw_fixed"] <= max(c_t2, c_rot) + 1e-9
            ),
        },
    }

    # --- the decision rule, fixed in advance ---
    tilt_recovery = rec_rot["tilt_fixed"]
    if checks["P2_tilt_beats_yaw"]["pass"] and checks["P3_tilt_recovers_half"]["pass"]:
        verdict = "HOLDS"
        action = (
            "Build the tilt cleanup into the Tier 2 pipeline; the Brev gate trains on "
            "cleaned references."
        )
    elif tilt_recovery is not None and tilt_recovery < 0.25:
        verdict = "FAILS"
        action = (
            "The orientation cleanup does not help the grasp signal. Keep it for its "
            "scored-metric gain only (AUC 0.425 -> 0.654, measured separately and "
            "unaffected by this result), and do not claim a training-signal benefit."
        )
    else:
        verdict = "PARTIAL"
        action = (
            "Build the cleanup in for the scoring gain; do not claim a training-signal "
            "benefit. Report the recovery figure as measured."
        )

    result = {
        "what": "pre-registered test of whether cleaning Tier 2 object orientation restores "
        "hand-object contact",
        "preregistration": "docs/EXP_CLEANUP_CONTACT.md",
        "episodes_scored": shared,
        "episodes_missing_per_series": missing,
        "contact_mean": {k: round(v, 4) for k, v in means.items()},
        "contact_mean_excluding_foot_episode": {k: round(v, 4) for k, v in means_no_foot.items()},
        "recovery_vs_rot_clean": {
            k: None if v is None else round(v, 4) for k, v in rec_rot.items()
        },
        "recovery_vs_object_clean": {
            k: None if v is None else round(v, 4) for k, v in rec_obj.items()
        },
        "checks": checks,
        "verdict": verdict,
        "action": action,
        "per_episode_contact": {
            k: {str(e): round(v[e], 4) for e in shared} for k, v in series.items()
        },
    }

    print(json.dumps({k: v for k, v in result.items() if k != "per_episode_contact"}, indent=2))
    print(f"\n{'ep':>3} " + " ".join(f"{k:>12s}" for k in series))
    for e in shared:
        flag = "  <- foot" if e == FOOT_EPISODE else ("  <- worst-hit" if e in WORST else "")
        print(f"{e:3d} " + " ".join(f"{series[k][e]:12.3f}" for k in series) + flag)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    print(f"\nVERDICT: {verdict}\n{action}")


if __name__ == "__main__":
    main()
