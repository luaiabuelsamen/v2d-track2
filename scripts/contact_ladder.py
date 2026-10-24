"""Contact coverage for the deployable cleaner, paired against the oracle arms.

Same extraction as scripts/score_cleanup_prediction.py: coverage per clip is
max(left, right) of `{side}_hand_contact_active: n/T` from the retargeter's own log,
at its 2 cm threshold. Duplicated rather than imported because these tools run under
`python -I`, which implies -P, so scripts/ cannot import from itself.

Expects the arm trees scripts/run_cleanup_arms.sh produced plus arms/support_cleaned, and the
main run's logs at <scratch>/logs where tier1 and tier2 share one directory.

Usage:
  python scripts/contact_ladder.py <scratch-dir> [-o results/arm_support_cleaned_contact.json]
"""
import argparse, json, re
from pathlib import Path
import numpy as np

CONTACT = re.compile(r"(left|right)_hand_contact_active: (\d+)/(\d+) frames")
COMPLETE = "Retargeting complete"
FOOT_EPISODE = 29
WORST = [0, 1, 14, 15, 18, 19, 22, 28]

def coverage(log_dir, prefix="tier*"):
    out = {}
    for log in sorted(Path(log_dir).glob(f"{prefix}_e*.log")):
        text = log.read_text(errors="replace")
        if COMPLETE not in text:
            continue
        found = CONTACT.findall(text)
        if not found:
            continue
        sides = {s: (int(a), int(n)) for s, a, n in found}
        frames = max(v[1] for v in sides.values())
        if frames:
            out[int(log.stem.split("_e")[-1])] = max(v[0] for v in sides.values()) / frames
    return out

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("scratch", type=Path, help="the dir scripts/setup_jetson_env.sh used")
parser.add_argument("-o", "--out", type=Path,
                    default=Path("results/arm_support_cleaned_contact.json"))
args = parser.parse_args()

S = args.scratch
main_logs = S / "logs"
series = {
    "tier1": coverage(main_logs, "tier1"),
    "tier2": coverage(main_logs, "tier2"),
    "support_cleaned": coverage(S / "arms/support_cleaned/logs"),
    "tilt_fixed": coverage(S / "arms/tilt_fixed/logs"),
    "rot_clean": coverage(S / "arms/rot_clean/logs"),
    "yaw_fixed": coverage(S / "arms/yaw_fixed/logs"),
    "object_clean": coverage(S / "arms/object_clean/logs"),
}
shared = sorted(set.intersection(*(set(v) for v in series.values())))
means = {k: float(np.mean([v[e] for e in shared])) for k, v in series.items()}
t1, t2 = means["tier1"], means["tier2"]

def rec(arm, ceiling):
    span = ceiling - t2
    return None if abs(span) < 1e-9 else (means[arm] - t2) / span

dep = series["support_cleaned"]
gain = {e: dep[e] - series["tier2"][e] for e in shared}
higher = sum(1 for g in gain.values() if g > 1e-9)
lower = sum(1 for g in gain.values() if g < -1e-9)
worst_gain = float(np.mean([gain[e] for e in WORST if e in shared]))
tilt_gain = {e: series["tilt_fixed"][e] - series["tier2"][e] for e in shared}
# fraction of the tilt oracle's contact gain the deployable cleaner captures
tilt_span = means["tilt_fixed"] - t2
result = {
    "what": "hand-object contact coverage for the deployable support-contact cleaner",
    "episodes_scored": shared,
    "contact_mean": {k: round(v, 4) for k, v in means.items()},
    "support_cleaned": {
        "contact_mean": round(means["support_cleaned"], 4),
        "delta_vs_tier2": round(means["support_cleaned"] - t2, 4),
        "episodes_improved": higher,
        "episodes_degraded": lower,
        "episodes_unchanged": len(shared) - higher - lower,
        "recovery_vs_rot_clean": round(rec("support_cleaned", means["rot_clean"]), 4),
        "recovery_vs_tier1": round(rec("support_cleaned", t1), 4),
        "fraction_of_tilt_oracle_contact_gain": None if abs(tilt_span) < 1e-9
            else round((means["support_cleaned"] - t2) / tilt_span, 4),
        "mean_gain_on_8_worst_hit": round(worst_gain, 4),
        "worst_single_episode_loss": round(min(gain.values()), 4),
    },
    "per_episode": {k: {str(e): round(v[e], 4) for e in shared} for k, v in series.items()},
    "per_episode_gain_vs_tier2": {str(e): round(gain[e], 4) for e in shared},
}
cols = ["tier2", "support_cleaned", "tilt_fixed", "rot_clean", "object_clean", "tier1"]
print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_episode")}, indent=2))
print("\n ep " + " ".join(f"{c:>16s}" for c in cols) + "      gain")
for e in shared:
    flag = " <- foot" if e == FOOT_EPISODE else (" <- worst-hit" if e in WORST else "")
    print(f"{e:3d} " + " ".join(f"{series[c][e]:16.3f}" for c in cols) + f" {gain[e]:+9.3f}{flag}")
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(f"\nwrote {args.out}")
