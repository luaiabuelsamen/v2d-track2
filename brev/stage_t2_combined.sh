#!/usr/bin/env bash
# Assemble the 30 Tier 2 references for one run, from however many sources, and prove the result.
#
# The combined-recipe run takes its references from more than one place: tetool's bridge-cleaned
# clips where they exist and are not flagged, interp's finger-target clips where they exist, and
# our own released-input retarget everywhere else. Sources are declared in priority order and the
# first one that has a clip, and does not flag it, wins. Nothing is merged column-wise: the
# retargeter solves the whole body against the object trajectory, so finger joints from a
# raw-object solve would put the fingers where the *noisy* object was. A clip comes from exactly
# one solve, and provenance.json says which.
#
#   brev/stage_t2_combined.sh --source clean=$S/t2_clean/tier2 --source raw=$S/t2_raw/tier2 \
#                             --skip clean=7,22
#
# Three shape changes from what the retargeter emits, all of them lessons paid for:
#   - drop the tier2/ level, so the tree matches what run_tier2_full.sh asks for;
#   - stage each clip's support USDA into whole_body/reconstructed_stage/, where
#     _discover_support_surface looks: from the parquet it walks up to the nearest
#     `sequence_id=<seq>` ancestor and reads `<that>.parent.parent/reconstructed_stage/
#     <seq>_support.usda`. Without it the trainer builds a scene with no table, the object falls
#     at reset, and the clip dies on frame ~0 -- which cost us three full-length cells;
#   - rebase object_mesh_paths / object_urdf_paths, which the retargeter writes as absolute LOCAL
#     paths. Newton cannot stat a missing mesh, falls back to parsing the string as inline URDF
#     XML, and dies on "not well-formed" naming neither the path nor the machine.
set -uo pipefail
cd /home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
export V2D_SCRATCH="$S"
PY="env -u PYTHONPATH ./.venv311/bin/python -I"
run_v2d() { $PY -c "import sys; sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; main()" "$@"; }

MAP="$S/tier2_episode_map.txt"
MOUNT=t2_combined
OUT=""
BRIEFS="${V2D_BRIEFS:-$HOME/projects/_ops/briefs}"
FROM_BRIEFS=0
ALLOW_PARTIAL=0
LABELS=()
declare -A ROOT=() SKIP=() NPZ=() REF_DATASET=()
while [ $# -gt 0 ]; do
  case "$1" in
    --map) MAP="$2"; shift 2 ;;
    --mount) MOUNT="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --from-briefs) FROM_BRIEFS=1; shift ;;
    --allow-partial) ALLOW_PARTIAL=1; shift ;;
    --briefs) BRIEFS="$2"; shift 2 ;;
    --source)
      LABELS+=("${2%%=*}"); ROOT["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --npz) NPZ["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --skip) SKIP["${2%%=*}"]=",${2#*=}," ; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# --from-briefs: take the handover paths from the briefs directory rather than the command line, so
# that the launch command is one line with no path in it to mistype, and so that the same command
# stages whatever has landed. A file that is not there is a source that is skipped; its clips fall
# through to the next source. Each file holds one line:
#
#     <tree root> [<npz dir> [<flagged episodes, comma-separated>]]
#
# The npz is needed because a kit score has to be converted against the same reference the policy
# trained on; for a source that is not the released input, scoring without it fails
# rollout-to-npz's own frame check (on B3 it moved MPPE by 0.36 cm before it was caught).
if [ "$FROM_BRIEFS" = 1 ]; then
  for pair in fing:finger_refs_path.txt clean:clean_refs_path.txt; do
    LABEL="${pair%%:*}"; FILE="$BRIEFS/${pair#*:}"
    if [ ! -s "$FILE" ]; then
      echo "briefs: no $(basename "$FILE") -- skipping the $LABEL source"
      continue
    fi
    read -r TREE NPZ_DIR FLAGGED _ < <(grep -v '^[[:space:]]*#' "$FILE" | grep . | head -1)
    [ -n "${TREE:-}" ] || { echo "FATAL: $FILE names no path" >&2; exit 2; }

    # A handover comes in one of two shapes and they are told apart by what is in them, not by
    # what we hoped for. tetool's bridge_v1 is the second kind: a drop-in --dataset-root in which
    # only observation.object.pose differs from the release.
    if [ -d "$TREE/whole_body/soma" ]; then
      ROOT["$LABEL"]="$TREE"
    elif [ -d "$TREE/track_2" ]; then
      # A dataset root has to be retargeted before a policy can train on it, and the rollouts then
      # have to be kit-scored against that same root, because the reference the policy tracked is
      # the cleaned pose, not the released one.
      REF_DATASET["$LABEL"]="$TREE"
      ROOT["$LABEL"]="$S/t2_${LABEL}/tier2"
      if [ ! -d "${ROOT[$LABEL]}" ]; then
        {
          echo "FATAL: $FILE names a dataset root, whose retarget is not on disk yet:"
          echo "       ${ROOT[$LABEL]}"
          echo "       Retarget it first (~70 min on two CPU workers), then stage:"
          echo "       v2d retarget --dataset-root $TREE --repo \$S/repo_src \\"
          echo "         --soma-root \$SOMA --seq-root \$S/t2_seq_${LABEL} \\"
          echo "         --out-root \$S/t2_${LABEL} --log-dir \$S/t2_log_${LABEL} \\"
          echo "         --validation-dir \$S/t2_val_${LABEL} --tiers tier2 --workers 2"
        } >&2
        exit 2
      fi
      # A retarget still in flight is the dangerous case: staging now would quietly take raw for
      # every clip that is not finished yet and nothing downstream would say so. Count them.
      HAVE="$(find "${ROOT[$LABEL]}" -name data.parquet 2>/dev/null | wc -l)"
      WANT="$(grep -c '[^[:space:]]' "$MAP")"
      if [ "$HAVE" -lt "$WANT" ] && [ "$ALLOW_PARTIAL" != 1 ]; then
        {
          echo "FATAL: the $LABEL retarget has $HAVE of $WANT clips -- it is still running."
          echo "       Staging now would silently fall back to raw for the other $((WANT - HAVE))."
          echo "       Wait for $S/t2_${LABEL}/ALL_DONE, or pass --allow-partial to accept the mix."
        } >&2
        exit 2
      fi
      # The source's own machine-readable verdict on which clips it should not be used for. The
      # prose in the brief is for people; this is what the stage acts on.
      CEILING="$TREE/ceiling_per_clip.json"
      if [ -s "$CEILING" ]; then
        FLAGGED_BY_SOURCE="$($PY -c "
import json
flagged = json.load(open('$CEILING')).get('fall_back_to_raw') or []
print(','.join(str(int(e)) for e in flagged))
" 2>/dev/null)"
        [ -n "${FLAGGED_BY_SOURCE:-}" ] && FLAGGED="${FLAGGED:+$FLAGGED,}$FLAGGED_BY_SOURCE"
      fi
    else
      echo "FATAL: $TREE (from $FILE) is neither a retargeted tree (whole_body/soma) nor a dataset root (track_2)" >&2
      exit 2
    fi

    LABELS+=("$LABEL")
    [ -n "${NPZ_DIR:-}" ] && NPZ["$LABEL"]="$NPZ_DIR"
    [ -n "${FLAGGED:-}" ] && SKIP["$LABEL"]=",${FLAGGED},"
    echo "briefs: $LABEL -> ${ROOT[$LABEL]}${REF_DATASET[$LABEL]:+  (scored against $TREE)}${NPZ_DIR:+  npz $NPZ_DIR}${FLAGGED:+  falls back to raw on $FLAGGED}"
  done
  # The released-input retarget is always last: it is the fallback, never the preference.
  LABELS+=(raw); ROOT[raw]="$S/t2_raw/tier2"
fi
OUT="${OUT:-$S/$MOUNT}"
[ "${#LABELS[@]}" -gt 0 ] || { echo "FATAL: at least one --source LABEL=ROOT" >&2; exit 2; }
[ -f "$MAP" ] || { echo "FATAL: no episode map at $MAP" >&2; exit 2; }

# A source that was asked for but is not on disk is a staging bug, not a fallback: the whole point
# of declaring it is that its clips are meant to be in this run.
FAIL=0
for LABEL in "${LABELS[@]}"; do
  [ -d "${ROOT[$LABEL]}" ] || { echo "FATAL: source $LABEL is not a directory: ${ROOT[$LABEL]}" >&2; FAIL=1; }
done
[ "$FAIL" = 0 ] || exit 2

DST="$OUT/whole_body/soma"
STAGE_DIR="$OUT/whole_body/reconstructed_stage"
rm -rf "$OUT"; mkdir -p "$DST" "$STAGE_DIR"
: > "$OUT/sources.txt"

STAGED=0
while read -r EP SEQ; do
  [ -n "${EP:-}" ] || continue
  CHOSEN=""; CHOSEN_LABEL=""
  for LABEL in "${LABELS[@]}"; do
    case "${SKIP[$LABEL]:-}" in *",$EP,"*) continue ;; esac
    CANDIDATE="${ROOT[$LABEL]}/whole_body/soma/sequence_id=${SEQ}/robot_name=g1/data.parquet"
    if [ -s "$CANDIDATE" ]; then CHOSEN="$CANDIDATE"; CHOSEN_LABEL="$LABEL"; break; fi
  done
  if [ -z "$CHOSEN" ]; then
    echo "e$(printf '%02d' "$EP") NO SOURCE HAS THIS CLIP ($SEQ)"; FAIL=1; continue
  fi

  SEQD="sequence_id=${SEQ}"
  mkdir -p "$DST/$SEQD/robot_name=g1"
  cp -r "$(dirname "$(dirname "$CHOSEN")")/object" "$DST/$SEQD/object"
  SUP_SRC="$S/data/track_2/tier_2_synthetic_noise/support_surface/${SEQ}_support.usda"
  if [ -f "$SUP_SRC" ]; then
    cp "$SUP_SRC" "$STAGE_DIR/${SEQ}_support.usda"
  else
    echo "e$(printf '%02d' "$EP") support USDA MISSING at $SUP_SRC"; FAIL=1
  fi
  run_v2d rebase-assets "$CHOSEN" \
    --new-root "/v2d/assets/${MOUNT}/whole_body/soma/$SEQD/object" \
    -o "$DST/$SEQD/robot_name=g1/data.parquet" > /dev/null \
    || { echo "e$(printf '%02d' "$EP") rebase FAILED"; FAIL=1; }
  printf '%s %s %s %s\n' "$EP" "$SEQ" "$CHOSEN_LABEL" "$CHOSEN" >> "$OUT/sources.txt"
  STAGED=$((STAGED + 1))
done < "$MAP"

# Prove it, by the trainer's own rules rather than by having been careful.
MOUNT="$MOUNT" OUT="$OUT" MAP="$MAP" $PY - <<'PYEOF'
import glob
import os
import pathlib
import sys

import pyarrow.parquet as pq

out = pathlib.Path(os.environ["OUT"])
mount = os.environ["MOUNT"]
wanted = {line.split()[1] for line in open(os.environ["MAP"]) if line.strip()}
bad = 0


def discover(motion_file):
    """_discover_support_surface's rule, applied to the staged tree, which has the box's shape."""
    path = pathlib.Path(motion_file).resolve()
    parents = [path] + list(path.parents)
    robot = next((q.name.split("=", 1)[1] for q in parents if q.name.startswith("robot_name=")), None)
    for parent in parents:
        if parent.name.startswith("sequence_id="):
            seq = parent.name.split("=", 1)[1]
            stage = parent.parent.parent / "reconstructed_stage"
            names = ([f"{seq}_{robot}_support.usda"] if robot else []) + [f"{seq}_support.usda"]
            for name in names:
                if (stage / name).exists():
                    return stage / name
            return None
    return None


files = sorted(glob.glob(f"{out}/**/data.parquet", recursive=True))
seen = set()
supported = 0
for path in files:
    seq_dir = pathlib.Path(path).parent.parent.name
    seq = seq_dir.split("=", 1)[1]
    seen.add(seq)
    table = pq.read_table(path)
    if table.num_rows != 1:
        print(f"BAD {seq}: {table.num_rows} rows, expected the single nested row")
        bad += 1
    for column in ("object_mesh_paths", "object_urdf_paths"):
        for row in table.column(column).to_pylist():
            for asset in row:
                want = f"/v2d/assets/{mount}/whole_body/soma/{seq_dir}/object/"
                if not asset.startswith(want):
                    print(f"BAD {seq} {column}: {asset}")
                    bad += 1
                    continue
                local = out / asset[len(f"/v2d/assets/{mount}/"):]
                if not local.exists():
                    print(f"MISSING LOCALLY {seq}: {local}")
                    bad += 1
    usda = discover(path)
    if usda is None:
        print(f"SUPPORT UNRESOLVED {seq}")
        bad += 1
        continue
    text = usda.read_text(errors="replace")
    primitives = sum(text.count(f"def {kind}") for kind in ("Cylinder", "Cube", "Sphere"))
    supported += bool(primitives)

missing = wanted - seen
if missing:
    print(f"NOT STAGED: {len(missing)} clips -- {sorted(missing)[:3]}")
    bad += len(missing)
print(f"checked {len(files)} parquets, {bad} problems; "
      f"{supported}/{len(files)} resolve to a supported scene")
sys.exit(1 if bad else 0)
PYEOF
RC=$?

( cd "$OUT" && find . -type f | sort | xargs md5sum ) > "$OUT.md5"
NPZ_MAP=""
REF_MAP=""
for LABEL in "${LABELS[@]}"; do
  [ -n "${NPZ[$LABEL]:-}" ] && NPZ_MAP="$NPZ_MAP$LABEL=${NPZ[$LABEL]} "
  [ -n "${REF_DATASET[$LABEL]:-}" ] && REF_MAP="$REF_MAP$LABEL=${REF_DATASET[$LABEL]} "
done
NPZ_MAP="$NPZ_MAP" REF_MAP="$REF_MAP" $PY - "$OUT" "$MOUNT" <<'PYEOF'
import json
import os
import pathlib
import subprocess
import sys

out = pathlib.Path(sys.argv[1])
rows = [line.split(maxsplit=3) for line in (out / "sources.txt").read_text().splitlines() if line]
sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                     check=False).stdout.strip()
(out / "provenance.json").write_text(json.dumps({
    "what": "the staged Tier 2 references for one full-budget run",
    "git_sha": sha,
    "container_mount": f"/v2d/assets/{sys.argv[2]}",
    # Which reference each source's rollouts must be kit-scored against. The launcher reads this
    # rather than being told again on the command line.
    "npz_by_source": dict(
        item.split("=", 1) for item in os.environ.get("NPZ_MAP", "").split() if "=" in item
    ),
    # A source that arrived as a dataset root: its rollouts convert against this root, because the
    # reference the policy tracked is the cleaned pose in it, not the released one.
    "ref_dataset_by_source": dict(
        item.split("=", 1) for item in os.environ.get("REF_MAP", "").split() if "=" in item
    ),
    "clips": {
        f"e{int(episode):02d}": {
            "episode": int(episode),
            "sequence_id": sequence,
            "source": label,
            "source_parquet": path,
        }
        for episode, sequence, label, path in rows
    },
    "by_source": {
        label: sorted(int(e) for e, _s, l, _p in rows if l == label)
        for label in sorted({row[2] for row in rows})
    },
}, indent=2) + "\n")
print(f"provenance: {out / 'provenance.json'}")
PYEOF

echo "staged $STAGED clips from $(wc -l < "$MAP") in the map, $(wc -l < "$OUT.md5") files, $(du -sh "$OUT" | cut -f1)"
awk '{print $3}' "$OUT/sources.txt" | sort | uniq -c | sed 's/^/  source: /'
echo "verification rc=$RC  (stage failures: $FAIL)"
[ "$RC" = 0 ] && [ "$FAIL" = 0 ] && echo "STAGE OK" || { echo "STAGE NOT OK"; exit 1; }
