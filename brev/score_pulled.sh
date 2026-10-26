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

RESULTS=""
DATA="$S/data"
KIT="$S/kit_extract/v2d_submission_kit"
PROVENANCE=""
declare -A NPZ=()
while [ $# -gt 0 ]; do
  case "$1" in
    --results) RESULTS="$2"; shift 2 ;;
    --dataset-root) DATA="$2"; shift 2 ;;
    --kit-root) KIT="$2"; shift 2 ;;
    --provenance) PROVENANCE="$2"; shift 2 ;;
    --npz) NPZ["${2%%=*}"]="${2#*=}"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$RESULTS" ] || { echo "FATAL: --results DIR" >&2; exit 2; }
[ -d "$KIT" ] || { echo "FATAL: no kit at $KIT" >&2; exit 2; }
PROVENANCE="${PROVENANCE:-$(dirname "$RESULTS")/t2_combined/provenance.json}"

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

for CLIP in "$RESULTS"/e[0-9][0-9]; do
  [ -d "$CLIP" ] || continue
  TAG="$(basename "$CLIP")"
  EP=$((10#${TAG#e}))
  ls -d "$CLIP"/policy_*/objects.parquet >/dev/null 2>&1 || continue

  # (1) choose, with no score in sight.
  run_v2d pick-checkpoint --cell "$TAG=$CLIP" -o "$CLIP/selection.json" >/dev/null 2>&1 \
    || { echo "$TAG: selection FAILED"; continue; }

  # (2) then score every checkpoint, for the record and for the registered bar.
  LABEL="$(source_of "$TAG")"
  REF=()
  [ -n "${NPZ[$LABEL]:-}" ] && REF=(--reference-npz "${NPZ[$LABEL]}")
  for TRAJ in "$CLIP"/policy_*/objects.parquet; do
    CDIR="$(dirname "$TRAJ")"
    [ -s "$CDIR/score.json" ] && continue
    if ! run_v2d rollout-to-npz "$TRAJ" --dataset-root "$DATA" --episode "$EP" --tier tier2 \
         "${REF[@]+${REF[@]}}" --npz-out "$CDIR/npz" -o "$CDIR/rollout_record.json" \
         > "$CDIR/rollout.log" 2>&1; then
      echo "$TAG $(basename "$CDIR"): rollout-to-npz FAILED"; tail -3 "$CDIR/rollout.log" | sed 's/^/    /'
      continue
    fi
    if ! run_v2d score --kit-root "$KIT" --dataset-root "$DATA" --tier tier2 \
         --pred "npz:$CDIR/npz" --episodes "$EP" -o "$CDIR/score.json" \
         > "$CDIR/score.log" 2>&1; then
      echo "$TAG $(basename "$CDIR"): score FAILED"; tail -3 "$CDIR/score.log" | sed 's/^/    /'
      rm -f "$CDIR/score.json"
      continue
    fi
    # The npz is 7 per clip and only the score is kept; the trajectory it came from is already here.
    rm -rf "$CDIR/npz"
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
