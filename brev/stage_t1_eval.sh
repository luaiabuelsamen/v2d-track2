#!/usr/bin/env bash
# Stage the clean Tier 1 references so the EXISTING wave-2 checkpoints evaluate against them.
#
# Why this is a staging problem and not a flag: `evaluate_policy.py` rebuilds the experiment from
# the checkpoint's own saved config (`evaluate_checkpoint` -> `checkpoint_config`), and its only
# path lever, `evaluation.source_root`, rebases a saved parquet path only when that path is
# relative or under `/workspace`. Ours are absolute and under `/v2d/assets`, so source_root does
# nothing for us. What does work is putting the clean parquet at the exact path the checkpoint
# already records:
#
#   /v2d/assets/<stage>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet
#
# Each of the 30 checkpoints names its own `<stage>` -- sw_A, t2_combined, sw_B1, sw_B2, sw_B4,
# sw_B4x, a29_refs -- read here from the checkpoint rather than assumed, so a clip whose arm I
# misremember fails loudly instead of silently evaluating against the wrong tree.
#
# Everything else about the run is held fixed by that same saved config: world count, reset mode,
# scene.support as trained, the object, the mesh. The one variable is the trajectory in the
# parquet: Tier 2's noisy object track becomes Tier 1's clean one. That is the comparison.
#
# Three shape rules, inherited from stage_t2_combined.sh where each was paid for:
#   - the support USDA goes in `<stage>/whole_body/reconstructed_stage/<seq>_support.usda`, where
#     `_discover_support_surface` looks; without it a supported scene builds with no table and the
#     object falls at reset;
#   - `object_mesh_paths` / `object_urdf_paths` are rebased from Jetson-local absolute paths to the
#     in-container ones, or Newton parses the path as inline URDF XML and dies on "not well-formed";
#   - the md5 manifest is written LAST and its line count is checked against the file count, so a
#     file written after the manifest cannot pass the box's gate.
#
#   brev/stage_t1_eval.sh [--motion DIR] [--out DIR] [--map FILE]
set -uo pipefail

REPO=/home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
MOTION="$S/t1_clean/motion/tier1"
OUT="$S/t1_eval/assets"
CKPTS="$REPO/ship_ckpts_wave2"
MAP="$S/tier2_episode_map.txt"
DATA="$S/data"
TIER1="$DATA/track_2/tier_1_multiview_caption"

while [ $# -gt 0 ]; do
  case "$1" in
    --motion) MOTION="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --map) MAP="$2"; shift 2 ;;
    --checkpoints) CKPTS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"
PY="env -u PYTHONPATH $REPO/.venv311/bin/python -I"
run_v2d() {
  $PY -c "import sys; sys.path.insert(0,'src'); sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; raise SystemExit(main())" "$@"
}

[ -f "$MAP" ] || { echo "FATAL: no episode map at $MAP" >&2; exit 2; }
[ -d "$MOTION" ] || { echo "FATAL: no clean Tier 1 motion tree at $MOTION" >&2; exit 2; }
[ -d "$TIER1" ] || { echo "FATAL: no released Tier 1 tree at $TIER1" >&2; exit 2; }

rm -rf "$OUT"
mkdir -p "$OUT"
: > "$S/t1_eval/sources.txt"
FAIL=0
STAGED=0

# What each checkpoint expects, from the checkpoint. Printed as "ep stage seq steps".
PLAN="$S/t1_eval/plan.txt"
$PY - "$CKPTS" > "$PLAN" <<'PYEOF' || exit 3
import json, struct, sys
from pathlib import Path

for tag_dir in sorted(Path(sys.argv[1]).iterdir()):
    if not tag_dir.is_dir() or not tag_dir.name.startswith("e"):
        continue
    policy = next(tag_dir.glob("policy_*/policy.safetensors"), None)
    if policy is None:
        raise SystemExit(f"FATAL: {tag_dir} has no policy.safetensors")
    with policy.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(length))
    config = None
    for key, value in header.get("__metadata__", {}).items():
        if "config" in key.lower():
            try:
                config = json.loads(value)
            except json.JSONDecodeError:
                continue
    if config is None:
        raise SystemExit(f"FATAL: {policy} carries no resolved config in its metadata")
    parquet = Path(config["task"]["parquet"])
    stage = parquet.parts[3]
    sequence = next(p.split("=", 1)[1] for p in parquet.parts if p.startswith("sequence_id="))
    support = bool(config.get("scene", {}).get("support", False))
    print(f"{int(tag_dir.name[1:])} {stage} {sequence} {policy.parent.name} {support}")
PYEOF

while read -r EP STAGE SEQ POLICY_DIR SUPPORT; do
  [ -n "${EP:-}" ] || continue
  TAG="$(printf 'e%02d' "$EP")"
  SRC="$MOTION/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
  if [ ! -s "$SRC" ]; then
    echo "$TAG NO CLEAN RETARGET at $SRC"; FAIL=1; continue
  fi
  SEQD="sequence_id=${SEQ}"
  DST="$OUT/$STAGE/whole_body/soma"
  mkdir -p "$DST/$SEQD/robot_name=g1" "$OUT/$STAGE/whole_body/reconstructed_stage"
  cp -r "$MOTION/whole_body/soma/$SEQD/object" "$DST/$SEQD/object"

  # Tier 1's own support surface, because this is a Tier 1 evaluation. Whether the clip HAS
  # primitives is a property of the asset, and the checkpoint's saved scene.support says whether
  # the trained scene expected one -- a clip that trained supported and has no usable primitive
  # here would build a table-less scene, so it is named.
  SUP="$TIER1/support_surface/${SEQ}_support.usda"
  if [ -f "$SUP" ]; then
    cp "$SUP" "$OUT/$STAGE/whole_body/reconstructed_stage/${SEQ}_support.usda"
  elif [ "$SUPPORT" = True ]; then
    echo "$TAG trained with scene.support=true but Tier 1 ships no $SUP"; FAIL=1
  fi

  run_v2d rebase-assets "$SRC" \
    --new-root "/v2d/assets/${STAGE}/whole_body/soma/$SEQD/object" \
    -o "$DST/$SEQD/robot_name=g1/data.parquet" > /dev/null \
    || { echo "$TAG rebase FAILED"; FAIL=1; }

  # The checkpoint travels with its clip, under the name the box's runner expects.
  mkdir -p "$OUT/ckpt/$TAG"
  cp "$CKPTS/$TAG/$POLICY_DIR/policy.safetensors" "$OUT/ckpt/$TAG/policy.safetensors"
  printf '%s %s %s %s %s\n' "$EP" "$TAG" "$STAGE" "$SEQ" "$SUPPORT" >> "$S/t1_eval/sources.txt"
  STAGED=$((STAGED + 1))
done < "$PLAN"

echo "staged $STAGED of 30 clips"
[ "$STAGED" = 30 ] || { echo "FATAL: $STAGED of 30 staged" >&2; FAIL=1; }

# Prove the tree by the trainer's own discovery rule, not by having been careful above.
OUT="$OUT" $PY - <<'PYEOF' || FAIL=1
import glob, os, pathlib
import pyarrow.parquet as pq

out = pathlib.Path(os.environ["OUT"])
bad = 0
supported = 0
files = sorted(glob.glob(f"{out}/*/whole_body/soma/*/robot_name=g1/data.parquet"))
for path in files:
    path = pathlib.Path(path)
    seq_dir = path.parent.parent.name
    seq = seq_dir.split("=", 1)[1]
    stage = path.parents[4].name
    table = pq.read_table(path)
    if table.num_rows != 1:
        print(f"BAD {seq}: {table.num_rows} rows, expected the single nested row")
        bad += 1
    for column in ("object_mesh_paths", "object_urdf_paths"):
        for row in table.column(column).to_pylist():
            for asset in row:
                want = f"/v2d/assets/{stage}/whole_body/soma/{seq_dir}/object/"
                if not asset.startswith(want):
                    print(f"BAD {seq} {column}: {asset}")
                    bad += 1
                    continue
                local = out / asset[len("/v2d/assets/"):]
                if not local.exists():
                    print(f"MISSING LOCALLY {seq}: {local}")
                    bad += 1
    usda = path.parents[2] / "reconstructed_stage" / f"{seq}_support.usda"
    supported += usda.exists()
print(f"verified {len(files)} staged parquets, {supported} with a support USDA, {bad} problem(s)")
raise SystemExit(1 if bad or len(files) != 30 else 0)
PYEOF

# The manifest LAST, and its line count checked, because the box gates on it.
( cd "$OUT" && find . -type f ! -name MD5SUMS -print0 | sort -z | xargs -0 md5sum > MD5SUMS )
LINES="$(wc -l < "$OUT/MD5SUMS")"
FILES="$(find "$OUT" -type f ! -name MD5SUMS | wc -l)"
echo "manifest $LINES lines, $FILES files"
[ "$LINES" = "$FILES" ] || { echo "FATAL: manifest/file count mismatch" >&2; FAIL=1; }

[ "$FAIL" = 0 ] || { echo "STAGING FAILED" >&2; exit 3; }
# The CONTENTS go into ~/v2d_assets on the box, not a t1_eval/ subdirectory: each checkpoint
# records /v2d/assets/<stage>/... with no prefix, and the evaluator reads that path verbatim.
echo "OK: $OUT ($(du -sh "$OUT" | cut -f1)) -- its contents mount at /v2d/assets/"
