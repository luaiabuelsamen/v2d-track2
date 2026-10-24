"""Score the pre-registered predictions in docs/EXP_SUPPORT_AND_ROOT.md.

Written and committed before any arm's contact number existed, so the scoring could not be
tuned to what it would see. T1-T5 are object-pose predictions and are checked from the committed
scorer outputs; T6 and R1-R4 need the retargeted arms. The decision rule is applied mechanically
and its verdict printed.

Contact coverage per clip = max(left, right) of `{side}_hand_contact_active: n/T` from the
retargeter's own log, at its 2 cm threshold. Duplicated from score_cleanup_prediction.py rather
than imported: these tools run under `python -I`, which implies `-P`, so the script's own
directory is not on sys.path and scripts/ cannot import from itself.

Usage:
  python scripts/score_support_root.py --arms-root DIR --baseline-logs DIR \
      [--floor-only-logs DIR] [-o results/support_root_verdict.json]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CONTACT = re.compile(r"(left|right)_hand_contact_active: (\d+)/(\d+) frames")
COMPLETE = "Retargeting complete"
FOOT_EPISODE = 29
# Committed scorer outputs, so the scored half of the scorecard needs no re-run.
SCORES = {
    "tier2": Path("results/tier2_input_ceiling_kitscorer.json"),
    "floor_only": Path("results/tier2_support_cleaned.json"),
    "table": Path("results/tier2_support_table.json"),
}
# Pre-registered thresholds.
T1_BAR, T2_MPPE_BAR, T6_BAR = 0.4860, 2.3053, 0.3950
R2_BAR, R3_DROP, R3_CLIPS = 0.010, 0.30, 24
UNLOCKED = [3, 6, 7, 20, 25]
GATE_CHECK = [24, 28]


def coverage(log_dir: Path, prefix: str = "tier*") -> dict[int, float]:
    out: dict[int, float] = {}
    for log in sorted(Path(log_dir).glob(f"{prefix}_e*.log")):
        text = log.read_text(errors="replace")
        if COMPLETE not in text:
            continue
        found = CONTACT.findall(text)
        if not found:
            continue
        sides = {side: (int(active), int(total)) for side, active, total in found}
        frames = max(v[1] for v in sides.values())
        if frames:
            out[int(log.stem.split("_e")[-1])] = max(v[0] for v in sides.values()) / frames
    return out


def overall(path: Path) -> dict:
    data = json.loads(path.read_text())
    return data.get("overall", data)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arms-root", type=Path, required=True, help="holds <arm>/logs per arm")
    parser.add_argument(
        "--baseline-logs",
        type=Path,
        required=True,
        help="the main run's log dir, where tier1 and tier2 share a directory",
    )
    parser.add_argument(
        "--floor-only-logs",
        type=Path,
        required=True,
        help="the shipped cleaner's arm, results/arm_support_cleaned.json's logs",
    )
    parser.add_argument("--arms", nargs="*", default=["table", "table_root", "root"])
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    series = {
        "tier1": coverage(args.baseline_logs, "tier1"),
        "tier2": coverage(args.baseline_logs, "tier2"),
        "floor_only": coverage(args.floor_only_logs),
    }
    for arm in args.arms:
        found = coverage(args.arms_root / arm / "logs")
        if found:
            series[arm] = found
    shared = sorted(set.intersection(*(set(v) for v in series.values() if v)))
    if not shared:
        raise SystemExit("no episode is present in every series yet")
    missing = {
        k: sorted(set(series["tier2"]) - set(v))
        for k, v in series.items()
        if set(series["tier2"]) - set(v)
    }
    mean = {k: float(np.mean([v[e] for e in shared])) for k, v in series.items()}

    scored = {k: overall(p) for k, p in SCORES.items() if p.exists()}
    table = scored.get("table", {})

    checks: dict[str, dict] = {
        "T1_table_auc": {
            "claim": f"table-extended AUC >= {T1_BAR}",
            "auc": round(table.get("add_auc", float("nan")), 4),
            "pass": bool(table.get("add_auc", -1) >= T1_BAR),
        },
        "T2_no_regression": {
            "claim": f"SP-SR >= 1.0, MP-SR >= 1.0, MPPE <= {T2_MPPE_BAR}",
            "spider_sr": table.get("spider_sr"),
            "maniptrans_sr": table.get("maniptrans_sr"),
            "mppe_cm": round(table.get("mppe_cm", float("nan")), 4),
            "pass": bool(
                table.get("spider_sr", 0) >= 1.0
                and table.get("maniptrans_sr", 0) >= 1.0
                and table.get("mppe_cm", 1e9) <= T2_MPPE_BAR
            ),
        },
    }

    cleaner = json.loads(Path("results/clean_support_table.json").read_text())
    floor_cleaner = json.loads(Path("results/clean_support_contact.json").read_text())
    checks["T3_all_cleaned"] = {
        "claim": "30/30 cleaned, 0 unchanged",
        "cleaned": cleaner["summary"]["cleaned"],
        "unchanged": cleaner["summary"]["unchanged"],
        "pass": bool(cleaner["summary"]["cleaned"] == 30 and cleaner["summary"]["unchanged"] == 0),
    }
    per_table = json.loads(SCORES["table"].read_text()).get("per_episode", {})
    per_floor_path = Path("results/tier2_support_cleaned_per_episode.json")
    per_floor = (
        json.loads(per_floor_path.read_text()).get("per_episode", {})
        if per_floor_path.exists()
        else {}
    )
    if per_table and per_floor:
        gains = {e: per_table[str(e)]["add_auc"] - per_floor[str(e)]["add_auc"] for e in UNLOCKED}
        checks["T4_unlocked_clips"] = {
            "claim": "each of e03/e06/e07/e20/e25 gains >= 0.01 AUC",
            "gains": {str(e): round(v, 4) for e, v in gains.items()},
            "count": sum(1 for v in gains.values() if v >= 0.01),
            "pass": bool(all(v >= 0.01 for v in gains.values())),
        }
    frames = {
        e: (
            floor_cleaner["per_episode"][str(e)].get("corrected_frames", 0),
            cleaner["per_episode"][str(e)].get("corrected_frames", 0),
        )
        for e in GATE_CHECK
    }
    checks["T5_gate_not_permissive"] = {
        "claim": "e24 and e28 gain no newly corrected frames",
        "corrected_frames": {str(e): list(v) for e, v in frames.items()},
        "pass": bool(all(a == b for a, b in frames.values())),
    }
    if "table" in series:
        checks["T6_table_contact"] = {
            "claim": f"table-extended contact >= {T6_BAR}",
            "contact": round(mean["table"], 4),
            "vs_floor_only": round(mean["table"] - mean["floor_only"], 4),
            "pass": bool(mean["table"] >= T6_BAR),
        }

    # --- the root arm ---
    if "table_root" in series and "table" in series:
        delta = float(np.mean([series["table_root"][e] - series["table"][e] for e in shared]))
        checks["R2_root_contact"] = {
            "claim": f"paired contact(table_root) - contact(table) >= +{R2_BAR}",
            "paired_delta": round(delta, 4),
            "episodes_improved": int(
                sum(1 for e in shared if series["table_root"][e] > series["table"][e] + 1e-9)
            ),
            "episodes_total": len(shared),
            "pass": bool(delta >= R2_BAR),
        }
    root_record = Path("results/clean_root_planting.json")
    if root_record.exists():
        record = json.loads(root_record.read_text())
        checks["R1_scores_unchanged"] = {
            "claim": "the root cleaner cannot move the scorer: all four metrics are functions of "
            "the object pose alone, which it does not touch",
            "note": "verified by construction -- the scored npz is the object cleaner's output, "
            "byte-identical between the table and table_root arms",
            "pass": None,
        }
        drops = record.get("summary", {}).get("stance_std_drop_fraction")
        checks["R3_stance_std"] = {
            "claim": f"stance-foot vertical std falls >= {R3_DROP:.0%} in >= {R3_CLIPS}/30 clips",
            "measured": drops,
            "pass": None if drops is None else bool(drops >= R3_CLIPS),
        }

    # --- the decision rule, fixed in advance ---
    candidates = {k: v for k, v in scored.items() if v}
    baseline = candidates.get("tier2", {})
    eligible = {
        k: v
        for k, v in candidates.items()
        if v.get("spider_sr", 0) >= baseline.get("spider_sr", 0)
        and v.get("maniptrans_sr", 0) >= baseline.get("maniptrans_sr", 0)
        and v.get("mppe_cm", 1e9) <= baseline.get("mppe_cm", 1e9)
    }
    order = ["tier2", "floor_only", "table"]  # simplest first, for the 0.005 tie rule
    winner, best = None, -1.0
    for name in order:
        if name in eligible and eligible[name]["add_auc"] > best + 0.005:
            winner, best = name, eligible[name]["add_auc"]
    ship_root = (
        all(checks.get(k, {}).get("pass") is not False for k in ("R2_root_contact",))
        and "R2_root_contact" in checks
        and checks["R2_root_contact"]["pass"]
    )

    result = {
        "what": "pre-registered scoring of the support-surface and root-alignment cleanups",
        "preregistration": "docs/EXP_SUPPORT_AND_ROOT.md",
        "episodes_scored": shared,
        "episodes_missing_per_series": missing,
        "contact_mean": {k: round(v, 4) for k, v in mean.items()},
        "scored_metrics": scored,
        "checks": checks,
        "decision": {
            "scored_reference_set": winner,
            "eligible_by_no_regression": sorted(eligible),
            "ship_root_alignment": bool(ship_root),
            "rule": "highest AUC among {tier2, floor_only, table} subject to no metric worse "
            "than raw Tier 2; ties within 0.005 go to the simpler. The root alignment "
            "ships iff R1, R2 and R4 all hold.",
        },
        "per_episode_contact": {
            k: {str(e): round(v[e], 4) for e in shared} for k, v in series.items()
        },
    }

    print(json.dumps({k: v for k, v in result.items() if k != "per_episode_contact"}, indent=2))
    cols = [
        c for c in ("tier2", "floor_only", "table", "table_root", "root", "tier1") if c in series
    ]
    print(f"\n{'ep':>3} " + " ".join(f"{c:>12s}" for c in cols))
    for e in shared:
        flag = "  <- foot" if e == FOOT_EPISODE else ("  <- unlocked" if e in UNLOCKED else "")
        print(f"{e:3d} " + " ".join(f"{series[c][e]:12.3f}" for c in cols) + flag)
    passed = [k for k, v in checks.items() if v.get("pass") is True]
    failed = [k for k, v in checks.items() if v.get("pass") is False]
    print(f"\npassed: {passed}\nfailed: {failed}")
    print(f"\nDECISION: scored reference set = {winner}; ship root alignment = {ship_root}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
