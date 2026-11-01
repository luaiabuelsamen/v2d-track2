"""Input-only adoption for the Tier 3 fine-tune, as registered in docs/T3_FINETUNE.md.

Reads nothing but each rollout's own `reference_*` columns -- the reconstruction the policy was
driven by -- so no ground truth enters the verdict. A fine-tuned rollout replaces its pass-1
rollout only if it has no drop and sits closer to that input reference.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DROP_M = 0.10  # the registered z-drop gate: achieved beyond 10 cm of the input reference

# A bare "closer to the input" test adopts noise. e17 improved its mean input error by 0.55% on a
# 99 cm error, the rule adopted, and against truth its AUC fell 0.2858 -> 0.2732
# (results/t3_e17_negative_headroom.json). So a batch may demand a relative margin. This stays
# input-only, which is what keeps it legal on Tier 3, where the input is not the scoring reference.
MIN_GAIN_PCT = 0.0


def measure(path):
    d = pd.read_parquet(path)
    a = d[["achieved_position_x", "achieved_position_y", "achieved_position_z"]].to_numpy()
    r = d[["reference_position_x", "reference_position_y", "reference_position_z"]].to_numpy()
    err = np.linalg.norm(a - r, axis=1)
    lost = np.flatnonzero(err > DROP_M)
    return {
        "mean_input_err_cm": round(float(err.mean()) * 100, 3),
        "max_input_err_cm": round(float(err.max()) * 100, 3),
        "first_loss_frame": int(lost[0]) if len(lost) else None,
        "drop": bool(len(lost)),
        "frames": len(d),
    }


def load(dirs):
    out = {}
    for base in dirs:
        for p in Path(base).glob("episode_*.parquet"):
            tag = f"e{int(p.name[8:14]):02d}"
            out[tag] = str(p)
    return out


# -o <path> is required rather than derived: an earlier version wrote its JSON to
# `sys.argv[0].replace("t3_adopt.py", "t3ws_adoption.json")`, and this file is not named
# t3_adopt.py, so the replace was a no-op and the tool overwrote ITSELF with its own output.
argv = sys.argv[1:]
out_path = None
if "-o" in argv:
    i = argv.index("-o")
    out_path = argv[i + 1]
    del argv[i : i + 2]
if "--min-input-gain-pct" in argv:
    i = argv.index("--min-input-gain-pct")
    MIN_GAIN_PCT = float(argv[i + 1])
    del argv[i : i + 2]
if out_path is None or len(argv) < 2:
    sys.exit("usage: adopt.py <incumbent-rollout-dir> <new-rollout-dir>... -o <out.json>")

old = load([argv[0]])
new = load(argv[1:])
rows, adopt = [], []
for tag in sorted(new):
    if tag not in old:
        continue
    o, n = measure(old[tag]), measure(new[tag])
    # No drop is the first condition; among no-drop rollouts, closer to the input wins.
    gain_pct = 0.0
    if o["mean_input_err_cm"] > 0:
        before, after = o["mean_input_err_cm"], n["mean_input_err_cm"]
        gain_pct = 100.0 * (before - after) / before
    better = (not n["drop"] and o["drop"]) or (n["drop"] == o["drop"] and gain_pct > MIN_GAIN_PCT)
    if better:
        adopt.append(tag)
    rows.append(
        {"clip": tag, "old": o, "new": n, "input_gain_pct": round(gain_pct, 2), "adopt": better}
    )
    print(
        f"{tag}  old {o['mean_input_err_cm']:7.2f} cm drop={int(o['drop'])}"
        f"   new {n['mean_input_err_cm']:7.2f} cm drop={int(n['drop'])}"
        f"   gain {gain_pct:+6.2f}%"
        f"   -> {'ADOPT' if better else 'keep pass 1'}"
    )
print(f"\nadopted {len(adopt)}/{len(rows)}: {' '.join(adopt)}")
Path(out_path).write_text(
    json.dumps(
        {
            "what": "Tier 3 fine-tune adoption, decided on the reconstruction input alone",
            "verdict_reads_ground_truth": False,
            "drop_gate_m": DROP_M,
            "min_input_gain_pct": MIN_GAIN_PCT,
            "adopted": adopt,
            "clips": rows,
        },
        indent=1,
    )
)
print(f"wrote {out_path}")
