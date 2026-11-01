"""Last logged value of every objective term, read from an offline W&B run on the box.

Runs INSIDE the training container, where wandb is installed:

    python read_wandb_terms.py "/workspace/outputs/t2_e05/wandb/offline-run-*/run-*.wandb"

W&B is offline on the box, and `files/wandb-summary.json` is only written when the run closes, so
a term cannot be checked from it while the run is alive. The `.wandb` file itself is appended as
training goes, and this walks its history records for the reward components -- which is how a
non-zero, finite contact_role_palm was confirmed 13 minutes into arm CR's first cells rather than
at the first checkpoint 40 minutes later.
"""

import json
import sys
from pathlib import Path

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

pattern = Path(sys.argv[1])
paths = sorted(pattern.parent.glob(pattern.name))
if not paths:
    print(json.dumps({"error": "no .wandb file yet"}))
    raise SystemExit(0)
store = datastore.DataStore()
store.open_for_scan(str(paths[-1]))
last, steps = {}, 0
while True:
    data = store.scan_data()
    if data is None:
        break
    record = pb.Record()
    record.ParseFromString(data)
    if record.HasField("history"):
        steps += 1
        for item in record.history.item:
            key = "/".join(item.nested_key) if item.nested_key else item.key
            if "contact" in key or "body_joint" in key or "objective" in key or "reward" in key:
                last[key] = item.value_json
print(json.dumps({"history_records": steps, "terms": last}, indent=1))
