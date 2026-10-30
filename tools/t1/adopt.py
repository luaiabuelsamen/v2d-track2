"""Tier 1 adoption: which fine-tuned clips replace their shipped rollout.

The rule registered in `docs/T1_WARMSTART.md`: a clip's new rollout is adopted only if its kit AUC
improves AND spider_sr does not fall AND maniptrans_sr does not fall AND mppe_cm does not rise.

Reading the kit score is legal here, and only here. `config/tracks.json` gives Tier 1's
`input_note` as "the original SOMA-X human and object reconstruction in the released Tier 1
multiview-caption dataset -- this is the clean trajectory used as the configured scoring
reference", so on Tier 1 the permitted input *is* the reference being scored against. On Tier 2 and
Tier 3 it is not, and there the input-only z-drop rule stays.

All four metrics, not AUC alone, because each tier is four boards: taking the per-clip best AUC
across the arms we hold raised AUC 0.46523 -> 0.46858 and lost SP-SR, MP-SR and MPPE
(`results/t1_perclip_best.json`). A clip that buys AUC with a success gate is not an improvement.

    python tools/t1/adopt.py SHIPPED_PERCLIP.json NEW_SCORES_DIR [-o OUT.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GATES = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")


def adopts(old: dict, new: dict) -> bool:
    return (
        new["add_auc"] > old["add_auc"]
        and new["spider_sr"] >= old["spider_sr"]
        and new["maniptrans_sr"] >= old["maniptrans_sr"]
        and new["mppe_cm"] <= old["mppe_cm"]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("shipped", type=Path, help="per-clip metrics of the entry on the boards")
    parser.add_argument("new_dir", type=Path, help="holds eXX.score.json for the new rollouts")
    parser.add_argument("--entry-record", type=Path, default=Path("results/tier2_entry_wave2.json"))
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args(argv)

    shipped = json.loads(args.shipped.read_text())
    entry = json.loads(args.entry_record.read_text())
    selected = {e["tag"]: e["selected_environment_steps"] for e in entry["episodes"]}
    final = max(selected.values())

    clips, adopted = {}, []
    for path in sorted(args.new_dir.glob("e*.score.json")):
        tag = path.name.split(".")[0]
        new = json.loads(path.read_text())["overall"]
        old = shipped[tag]
        verdict = adopts(old, new)
        if verdict:
            adopted.append(tag)
        clips[tag] = {
            "shipped": {k: round(old[k], 5) for k in GATES},
            "fine_tuned": {k: round(new[k], 5) for k in GATES},
            "delta_auc": round(new["add_auc"] - old["add_auc"], 5),
            "ships_final_checkpoint": selected[tag] == final,
            "adopted": verdict,
        }

    # The entry mean moves only by what is adopted, over all 30 clips.
    gain = sum(clips[t]["delta_auc"] for t in adopted)
    base = sum(v["add_auc"] for v in shipped.values()) / len(shipped)
    record = {
        "what": "Tier 1 adoption of the warm-start fine-tune, under the registered"
        " no-regression rule",
        "rule": "AUC improves and no other board regresses; legal because Tier 1's permitted input"
        " is the configured scoring reference",
        "clips_scored": len(clips),
        "adopted": adopted,
        "entry_mean_auc": {
            "shipped": round(base, 5),
            "after_adoption": round(base + gain / len(shipped), 5),
            "gain": round(gain / len(shipped), 5),
        },
        "chord_tier1_baseline": 0.51038,
        "clips": clips,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "clips"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
