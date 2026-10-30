"""Why did a transfer clip score what it scored? Reports the mechanism, not just the number.

A donor actor can fail three distinguishable ways, and the number alone cannot tell them apart:
  * it never engages the object at all (transfer did not happen in any useful sense)
  * it holds briefly then drops (transfer partially worked; more steps or a closer donor might)
  * it tracks throughout but imprecisely (transfer worked; the clip is simply hard)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DROP_M = 0.10

path, tag = sys.argv[1], sys.argv[2]
d = pd.read_parquet(path)
ach = d[["achieved_position_x", "achieved_position_y", "achieved_position_z"]].to_numpy()
ref = d[["reference_position_x", "reference_position_y", "reference_position_z"]].to_numpy()
err = np.linalg.norm(ach - ref, axis=1)
lost = np.flatnonzero(err > DROP_M)
first = int(lost[0]) if len(lost) else None
frames = len(d)

rec = json.loads(Path("results/t1_shipped_perclip.json").read_text())[tag]

held = float((err <= DROP_M).mean())
if first is None:
    mech = "tracked the whole clip inside 10 cm -- transfer worked"
elif first < 0.05 * frames:
    mech = f"never engaged: lost the object by frame {first} of {frames}"
elif held < 0.5:
    mech = f"held then dropped at frame {first} of {frames} ({held:.0%} of frames inside 10 cm)"
else:
    mech = f"mostly tracked ({held:.0%} inside 10 cm), first excursion at frame {first}"

print(f"{tag}: {frames} frames | mean err {err.mean() * 100:6.2f} cm | "
      f"max {err.max() * 100:6.2f} cm | inside 10 cm {held:.0%}")
print(f"       first loss: {first if first is not None else 'never'} | {mech}")
print(f"       shipped record: AUC {rec['add_auc']:.4f}, MPPE {rec['mppe_cm']:.2f} cm")
