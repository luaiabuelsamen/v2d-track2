#!/usr/bin/env bash
# Kit-score every pulled checkpoint and choose one per clip, on the host, as the results arrive.
#
# Order matters and is deliberate: the z-drop selection runs FIRST, before any kit score exists for
# that clip, and it reads only objects.parquet and metrics.json. The kit scorer compares against
# the Tier 1 clean truth, so a kit score IS ground truth; if selection ran after scoring, no reader
# could tell whether the choice had been made on the answer. It cannot have been, because the score
# was not there yet.
#
# Idempotent and safe to call on a partial pull: a checkpoint whose score.json already exists is
# skipped, so the launcher can call this every 10 minutes for 20 hours.
#
#   brev/score_pulled.sh --results $S/t2_combined_results [--npz clean=$S/npz_clean]
set -uo pipefail
cd /home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
PY="env -u PYTHONPATH ./.venv311/bin/python -I"
run_v2d() { $PY -c "import sys; sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; main()" "$@"; }

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
ARGV=("$@")

RESULTS=""
DATA="$S/data"
KIT="$S/kit_extract/v2d_submission_kit"
PROVENANCE=""
declare -A NPZ=() REF_DATASET=()
while [ $# -gt 0 ]; do
  case "$1" in
    --results) RESULTS="$2"; shift 2 ;;
    --dataset-root) DATA="$2"; shift 2 ;;
    --kit-root) KIT="$2"; shift 2 ;;
    --provenance) PROVENANCE="$2"; shift 2 ;;
    --npz) NPZ["${2%%=*}"]="${2#*=}"; shift 2 ;;
    --ref-dataset) REF_DATASET["${2%%=*}"]="${2#*=}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$RESULTS" ] || { echo "FATAL: --results DIR" >&2; exit 2; }
[ -d "$KIT" ] || { echo "FATAL: no kit at $KIT" >&2; exit 2; }
PROVENANCE="${PROVENANCE:-$(dirname "$RESULTS")/t2_combined/provenance.json}"

# One pass at a time. Eight boxes finish clips within minutes of each other, and on 2026-10-26 two
# passes ran together: both saw no score.json for the same checkpoint, both converted into the same
# $CDIR/npz, and the first to finish deleted it under the other -- which is the FileNotFoundError on
# episode_000011.npz that left e11's selected checkpoint unscored. A pass is minutes, so waiting is
# cheap; running twice is not, on a Jetson that is allowed two CPU-heavy workers.
LOCK="$RESULTS/.score_pulled.lock"
if [ -z "${V2D_SCORE_PASS_LOCKED:-}" ] && command -v flock >/dev/null 2>&1; then
  export V2D_SCORE_PASS_LOCKED=1
  flock -w "${V2D_SCORE_LOCK_WAIT:-3600}" "$LOCK" bash "$SELF" "${ARGV[@]+${ARGV[@]}}"
  rc=$?
  # flock exits 1 when it gives up waiting; the pass itself exits 0 or 2 (its FATALs), so 1 here
  # means the lock, and the next pull calls this again -- every step of it is idempotent.
  [ "$rc" = 1 ] && echo "!! another scoring pass held $LOCK for ${V2D_SCORE_LOCK_WAIT:-3600}s;" \
    "skipping this one, the next pull will catch up" >&2
  exit $rc
fi

# The reference a rollout is converted against must be the reference it trained on. For a cleaned
# source that is not the released input, and rollout-to-npz's own frame check catches the mistake
# -- on B3 it moved MPPE by 0.36 cm before it was caught.
source_of() {
  [ -f "$PROVENANCE" ] || { echo raw; return; }
  $PY -c "
import json, sys
clips = json.load(open('$PROVENANCE'))['clips']
print(clips.get('$1', {}).get('source', 'raw'))
"
}

clip_ref_dataset() {
  # Empty means "the released root", which is this script's default.
  [ -f "$PROVENANCE" ] || return 0
  $PY -c "
import json
stage = json.load(open('$PROVENANCE'))
print(stage.get('ref_dataset_by_clip', {}).get('$1', ''))
"
}

for CLIP in "$RESULTS"/e[0-9][0-9]; do
  [ -d "$CLIP" ] || continue
  TAG="$(basename "$CLIP")"
  EP=$((10#${TAG#e}))
  ls -d "$CLIP"/policy_*/objects.parquet >/dev/null 2>&1 || continue

  # (1) choose, with no score in sight.
  run_v2d pick-checkpoint --cell "$TAG=$CLIP" -o "$CLIP/selection.json" >/dev/null 2>&1 \
    || { echo "$TAG: selection FAILED"; continue; }

  # (2) then score every checkpoint, for the record and for the registered bar.
  # Convert against the reference the policy actually tracked, and score against the release.
  # Those are two different roots for a cleaned source, and getting it wrong is not loud: on B3 it
  # moved MPPE by 0.36 cm and was only caught by rollout-to-npz's own frame check.
  LABEL="$(source_of "$TAG")"
  REF=()
  CONVERT_ROOT="$DATA"
  [ -n "${NPZ[$LABEL]:-}" ] && REF=(--reference-npz "${NPZ[$LABEL]}")
  [ -n "${REF_DATASET[$LABEL]:-}" ] && CONVERT_ROOT="${REF_DATASET[$LABEL]}"
  # The stage's per-clip map wins: a single tree can carry two references (interp's finger tree
  # carries the cleaned object trajectory on 29 clips and the released one on the clip the cleaner
  # flagged), and the clip is what the convert root is a property of.
  CLIP_ROOT="$(clip_ref_dataset "$TAG")"
  [ -n "$CLIP_ROOT" ] && CONVERT_ROOT="$CLIP_ROOT"
  for TRAJ in "$CLIP"/policy_*/objects.parquet; do
    CDIR="$(dirname "$TRAJ")"
    [ -s "$CDIR/score.json" ] && continue
    NPZ_DIR="$CDIR/npz.$$"
    if ! run_v2d rollout-to-npz "$TRAJ" --dataset-root "$CONVERT_ROOT" --episode "$EP" --tier tier2 \
         "${REF[@]+${REF[@]}}" --npz-out "$NPZ_DIR" -o "$CDIR/rollout_record.json" \
         > "$CDIR/rollout.log" 2>&1; then
      echo "$TAG $(basename "$CDIR"): rollout-to-npz FAILED"; tail -3 "$CDIR/rollout.log" | sed 's/^/    /'
      rm -rf "$NPZ_DIR"
      continue
    fi
    if ! run_v2d score --kit-root "$KIT" --dataset-root "$DATA" --tier tier2 \
         --pred "npz:$NPZ_DIR" --episodes "$EP" -o "$CDIR/score.json" \
         > "$CDIR/score.log" 2>&1; then
      echo "$TAG $(basename "$CDIR"): score FAILED"; tail -3 "$CDIR/score.log" | sed 's/^/    /'
      rm -f "$CDIR/score.json"; rm -rf "$NPZ_DIR"
      continue
    fi
    # The npz is 7 per clip and only the score is kept; the trajectory it came from is already here.
    rm -rf "$NPZ_DIR"
  done
done

# (3) the run-level record: the selected checkpoint per clip and the kit score it earned.
RESULTS="$RESULTS" PROVENANCE="$PROVENANCE" $PY - <<'PYEOF'
import json
import os
import pathlib

results = pathlib.Path(os.environ["RESULTS"])
provenance = pathlib.Path(os.environ["PROVENANCE"])
sources = {}
if provenance.is_file():
    sources = {tag: clip["source"] for tag, clip in json.loads(provenance.read_text())["clips"].items()}

clips = {}
for selection_path in sorted(results.glob("e[0-9][0-9]/selection.json")):
    tag = selection_path.parent.name
    selection = json.loads(selection_path.read_text())["episodes"][tag]
    chosen = selection["selected_environment_steps"]
    score_path = selection_path.parent / f"policy_{chosen}/score.json"
    scored = None
    if score_path.is_file():
        overall = json.loads(score_path.read_text())["overall"]
        scored = {
            "kit_add_auc": overall["add_auc"],
            "kit_spider_sr": overall["spider_sr"],
            "kit_maniptrans_sr": overall["maniptrans_sr"],
            "kit_mppe_cm": overall["mppe_cm"],
        }
    clips[tag] = {
        "episode": selection["episode"],
        "reference_source": sources.get(tag, "unknown"),
        "checkpoints_evaluated": len(selection["checkpoints"]),
        "drop_free_checkpoints": len(selection["drop_free_checkpoints"]),
        "selected_environment_steps": chosen,
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "selection_reason": selection["selection_reason"],
        "selected_kit_score": scored,
    }

scored_clips = [c for c in clips.values() if c["selected_kit_score"]]
summary = {
    "clips_with_results": len(clips),
    "clips_scored": len(scored_clips),
    "clips_with_no_drop_free_checkpoint": sum(1 for c in clips.values() if not c["drop_free_checkpoints"]),
}
for metric in ("kit_add_auc", "kit_spider_sr", "kit_maniptrans_sr", "kit_mppe_cm"):
    if scored_clips:
        summary[f"mean_{metric}"] = round(
            sum(c["selected_kit_score"][metric] for c in scored_clips) / len(scored_clips), 6
        )
record = {
    "what": "the combined-recipe Tier 2 run: one z-drop-selected checkpoint per clip, kit-scored",
    "selection_rule": "z-drop, on the input reference only; see results/checkpoint_selection.json",
    "summary": summary,
    "clips": clips,
}
(results / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")
for tag, clip in clips.items():
    score = clip["selected_kit_score"]
    print(
        f"{tag} [{clip['reference_source']}] {clip['checkpoints_evaluated']} ckpts, "
        f"{clip['drop_free_checkpoints']} drop-free -> "
        f"{clip['selected_environment_steps'] // 1_000_000}M  "
        + (f"kit AUC {score['kit_add_auc']:.4f} MPPE {score['kit_mppe_cm']:.2f} cm"
           if score else "not scored yet")
    )
print(json.dumps(summary, indent=2))
print(f"wrote {results / 'run_record.json'}")
PYEOF
