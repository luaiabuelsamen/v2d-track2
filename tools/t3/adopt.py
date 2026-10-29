"""Input-only adoption for the Tier 3 fine-tune, as registered in docs/T3_FINETUNE.md.

Reads nothing but each rollout's own `reference_*` columns -- the reconstruction the policy was
driven by -- so no ground truth enters the verdict. A fine-tuned rollout replaces its pass-1
rollout only if it has no drop and sits closer to that input reference.
"""
import glob, json, os, sys
import numpy as np, pandas as pd

DROP_M = 0.10  # the registered z-drop gate: achieved beyond 10 cm of the input reference


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
        for p in glob.glob(os.path.join(base, "episode_*.parquet")):
            tag = "e%02d" % int(os.path.basename(p)[8:14])
            out[tag] = p
    return out


old = load([sys.argv[1]])
new = load(sys.argv[2:])
rows, adopt = [], []
for tag in sorted(new):
    if tag not in old:
        continue
    o, n = measure(old[tag]), measure(new[tag])
    # No drop is the first condition; among no-drop rollouts, closer to the input wins.
    better = (not n["drop"] and o["drop"]) or (
        n["drop"] == o["drop"] and n["mean_input_err_cm"] < o["mean_input_err_cm"]
    )
    if better:
        adopt.append(tag)
    rows.append({"clip": tag, "old": o, "new": n, "adopt": better})
    print(
        f"{tag}  old {o['mean_input_err_cm']:7.2f} cm drop={int(o['drop'])}"
        f"   new {n['mean_input_err_cm']:7.2f} cm drop={int(n['drop'])}"
        f"   -> {'ADOPT' if better else 'keep pass 1'}"
    )
print(f"\nadopted {len(adopt)}/{len(rows)}: {' '.join(adopt)}")
json.dump(
    {
        "what": "Tier 3 fine-tune adoption, decided on the reconstruction input alone",
        "verdict_reads_ground_truth": False,
        "drop_gate_m": DROP_M,
        "adopted": adopt,
        "clips": rows,
    },
    open(sys.argv[0].replace("t3_adopt.py", "t3ws_adoption.json"), "w"),
    indent=1,
)
