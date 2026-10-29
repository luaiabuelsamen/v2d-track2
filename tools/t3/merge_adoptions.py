"""Merge every Tier 3 adoption batch into one record that names each clip's source tree.

The packer must never pick a rollout by directory order. The Tier 1 packer did exactly that --
it searched a list of trees and kept the last match, so appending one silently re-chose two clips
and shipped rollouts that had lost their own comparison. So the merged record carries
`source[clip]`, an absolute rollout directory, and the packer copies from there or fails.

Later batches win over earlier ones for the same clip, which is the intent: a clip re-run under a
newer arm was judged against the incumbent at that moment.

    python tools/t3/merge_adoptions.py <batch.json>:<rollout-dir> [...] -o <merged.json>
"""

import json
import os
import sys

args = sys.argv[1:]
if "-o" not in args:
    sys.exit("usage: merge_adoptions.py <batch.json>:<rollout-dir> [...] -o <merged.json>")
i = args.index("-o")
out_path = args[i + 1]
pairs = args[:i] + args[i + 2:]
if not pairs:
    sys.exit("no batches given")

adopted: list[str] = []
source: dict[str, str] = {}
batches = []
for spec in pairs:
    path, _, roll = spec.rpartition(":")
    if not path or not roll:
        sys.exit(f"expected <batch.json>:<rollout-dir>, got {spec!r}")
    with open(path) as fh:
        rec = json.load(fh)
    got = list(rec.get("adopted", []))
    # Claim only the clips whose rollout is really in THIS tree, both files. One batch can be
    # spread over several boxes (gen 1 ran on ws_t3a and ws_t3b), and the aggregate directory
    # someone made later held the parquets without their .json sidecars -- which the packer needs.
    # So presence is checked here, where it can be reported, not at copy time where cp just dies.
    claimed = []
    for tag in got:
        ep = int(tag[1:])
        base = f"{roll}/episode_{ep:06d}"
        if os.path.exists(f"{base}.parquet") and os.path.exists(f"{base}.json"):
            if tag not in adopted:
                adopted.append(tag)
            source[tag] = roll      # a later spec overrides an earlier one
            claimed.append(tag)
    batches.append({"batch": path, "rollouts": roll, "adopted": got, "claimed_here": claimed})
    print(f"{path}: {len(got)} adopted, {len(claimed)} present here -> {roll}")
    for tag in got:
        if tag not in source:
            print(f"    {tag}: not in this tree yet")

missing = [t for t in adopted if t not in source]
if missing:
    sys.exit(f"FATAL: adopted with no source tree: {' '.join(missing)}")

merged = {
    "what": "every Tier 3 clip that ships a fine-tuned rollout, with the tree it comes from",
    "verdict_reads_ground_truth": False,
    "adopted": sorted(adopted),
    "source": source,
    "batches": batches,
}
with open(out_path, "w") as fh:
    json.dump(merged, fh, indent=1)
print(f"\n{len(adopted)} adopted in total -> {out_path}")
