#!/usr/bin/env bash
# Stage the six A/B references into the layout the box expects, and rebase their asset paths.
#
# Two shape changes from what the retargeter emits:
#   - drop the tier2/ level, so the tree matches gated_stage (whole_body/soma/sequence_id=.../)
#     and run_ab350.sh's PARQUET path is the same shape it was for e24;
#   - rebase object_mesh_paths / object_urdf_paths, which the retargeter writes as absolute LOCAL
#     paths. Newton cannot stat a missing mesh, falls back to parsing the string as inline URDF
#     XML and dies on "not well-formed", naming neither the path nor the machine. Per sequence,
#     because --new-root is the directory holding that clip's assets, not a tree root -- getting
#     that wrong flattened all 30 Tier 1 references onto one mesh earlier today.
set -uo pipefail
cd /home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
export V2D_SCRATCH="$S"
PY="env -u PYTHONPATH ./.venv311/bin/python -I"
run_v2d() { $PY -c "import sys; sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; main()" "$@"; }

FAIL=0
for ARM in raw clean; do
  SRC=$S/ab_${ARM}/tier2/whole_body/soma
  DST=$S/ab_stage_${ARM}/whole_body/soma
  rm -rf "$S/ab_stage_${ARM}"; mkdir -p "$DST"
  for SEQDIR in "$SRC"/sequence_id=*; do
    [ -d "$SEQDIR" ] || continue
    SEQ="$(basename "$SEQDIR")"
    mkdir -p "$DST/$SEQ/robot_name=g1"
    cp -r "$SEQDIR/object" "$DST/$SEQ/object"
    run_v2d rebase-assets "$SEQDIR/robot_name=g1/data.parquet" \
      --new-root "/v2d/assets/ab_${ARM}/whole_body/soma/$SEQ/object" \
      -o "$DST/$SEQ/robot_name=g1/data.parquet" > /dev/null || { echo "rebase FAILED $ARM $SEQ"; FAIL=1; }
  done
done

# Prove it: every rebased path must point under /v2d/assets/ab_<arm>/... for its OWN sequence,
# and the file it names must exist in the staged tree at the matching relative location.
$PY - <<'PYEOF'
import glob, sys, pathlib
import pyarrow.parquet as pq
import os
S = os.environ["V2D_SCRATCH"]
bad = 0
n = 0
for arm in ("raw", "clean"):
    root = pathlib.Path(f"{S}/ab_stage_{arm}")
    for f in sorted(glob.glob(f"{root}/**/data.parquet", recursive=True)):
        n += 1
        seq = pathlib.Path(f).parent.parent.name
        t = pq.read_table(f)
        for col in ("object_mesh_paths", "object_urdf_paths"):
            for row in t.column(col).to_pylist():
                for p in row:
                    want = f"/v2d/assets/ab_{arm}/whole_body/soma/{seq}/object/"
                    if not p.startswith(want):
                        print(f"BAD {arm} {seq} {col}: {p}"); bad += 1; continue
                    local = root / p[len("/v2d/assets/" + f"ab_{arm}/"):]
                    if not local.exists():
                        print(f"MISSING LOCALLY {arm} {seq}: {local}"); bad += 1
print(f"checked {n} parquets, {bad} problems")
sys.exit(1 if (bad or n != 6) else 0)
PYEOF
RC=$?
echo "verification rc=$RC  (stage failures: $FAIL)"
for ARM in raw clean; do
  ( cd $S/ab_stage_${ARM} && find . -type f | sort | xargs md5sum ) > $S/ab_stage_${ARM}.md5
  echo "$ARM: $(wc -l < $S/ab_stage_${ARM}.md5) files, $(du -sh $S/ab_stage_${ARM} | cut -f1)"
done
[ "$RC" = 0 ] && [ "$FAIL" = 0 ] && echo "STAGE OK" || echo "STAGE NOT OK"
