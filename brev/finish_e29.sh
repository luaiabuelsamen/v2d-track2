#!/usr/bin/env bash
# Finish the entry once e29's re-roll has landed: adopt it, restage all 30, pack both tracks.
#
# The 2026-10-28 reboot left the entry 29 of 30 (docs/RECOVERY_2026-10-28.md). A29 re-rolls e29 on
# arm A's recipe; this is everything after it lands, in one command so the sequence is the same
# every time it is run:
#
#   1. take e29's selected checkpoint from the A29 run record -- the registered z-drop pick, which
#      score_pulled.sh has already applied; nothing is re-selected here;
#   2. copy that checkpoint into ship_ckpts_wave2/e29/policy_<steps>/, so the backup is whole again
#      and `v2d restage-entry` finds e29 the same way it finds the other 29;
#   3. rewrite the e29 row of results/tier2_entry_wave2.json, keeping the lost row in a history
#      field, because the new rollout is NOT the one that was lost and the record has to say so;
#   4. restage all 30 episodes, with no --allow-missing, so a gap is a refusal;
#   5. pack tier2 and tier1 through the kit, validate, and score locally.
#
# It never uploads. The last line prints the four scores, the CSV digest and the commit to report.
#
#   brev/finish_e29.sh [--commit <sha>] [--dry-run]
set -uo pipefail

REPO=/home/jetson3/projects/v2d-challenge
S="${V2D_SCRATCH:-/tmp/claude-1000/-home-jetson3-projects-v2d-challenge/111f0508-5f0a-4a3f-b52f-1fff9621b161/scratchpad}"
RUN="$S/second_wave/run_A29/results"
BACKUP="$REPO/ship_ckpts_wave2"
ENTRY="$REPO/results/tier2_entry_wave2.json"
KIT="$S/kit_extract/v2d_submission_kit"
DATA="$S/data"
OUT="$S/t2_entry_wave2"
COMMIT=""
DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2 ;;
    --run) RUN="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"
PY="env -u PYTHONPATH ./.venv311/bin/python -I"
run_v2d() {
  $PY -c "import sys; sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; raise SystemExit(main())" "$@"
}

# The commit the CSV rows quote has to be ON THE REMOTE, or the provenance points at code nobody
# else can read. Default to origin/main, and refuse if HEAD is ahead of it.
if [ -z "$COMMIT" ]; then
  COMMIT="$(git rev-parse origin/main)"
fi
if ! git merge-base --is-ancestor "$COMMIT" origin/main 2>/dev/null; then
  echo "FATAL: $COMMIT is not an ancestor of origin/main; push before packing from it" >&2
  exit 3
fi
AHEAD="$(git rev-list --count "$COMMIT"..HEAD 2>/dev/null || echo 0)"
[ "$AHEAD" = 0 ] || echo "note: HEAD is $AHEAD commit(s) past the commit being stamped ($COMMIT)"

RECORD="$RUN/run_record.json"
[ -f "$RECORD" ] || { echo "FATAL: no A29 run record at $RECORD -- has the re-roll landed?" >&2; exit 3; }

# ---------------------------------------------------------------- 1 & 2: adopt e29's pick
STEPS="$($PY -c "
import json,sys
print(json.load(open('$RECORD'))['clips']['e29']['selected_environment_steps'])
")" || { echo "FATAL: no e29 selection in $RECORD" >&2; exit 3; }
CELL="$RUN/e29/policy_$STEPS"
for f in objects.parquet metrics.json; do
  [ -f "$CELL/$f" ] || { echo "FATAL: $CELL/$f is missing" >&2; exit 3; }
done
echo "e29 selected checkpoint: $STEPS ($CELL)"

DEST="$BACKUP/e29/policy_$STEPS"
if [ "$DRY" = 1 ]; then
  echo "DRY: would copy $CELL -> $DEST"
else
  mkdir -p "$DEST"
  for f in objects.parquet metrics.json policy.safetensors score.json rollout_record.json; do
    [ -f "$CELL/$f" ] && cp -f "$CELL/$f" "$DEST/$f"
  done
  echo "e29 restored into the backup: $(ls "$DEST" | tr '\n' ' ')"
fi

# ---------------------------------------------------------------- 3: the entry record's e29 row
if [ "$DRY" = 1 ]; then
  echo "DRY: would rewrite the e29 row of $ENTRY"
else
  $PY - "$ENTRY" "$RECORD" "$DEST" <<'PYEOF' || exit 3
import hashlib, json, sys
from pathlib import Path

entry_path, record_path, dest = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
entry = json.loads(entry_path.read_text())
clip = json.loads(record_path.read_text())["clips"]["e29"]

policy = dest / "policy.safetensors"
digest = clip["selected_checkpoint_sha256"]
if policy.is_file():
    h = hashlib.sha256()
    with policy.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != digest:
        raise SystemExit(
            f"FATAL: {policy} digests {h.hexdigest()[:16]}... but the run record selected "
            f"{digest[:16]}..."
        )

rows = entry["episodes"]
old = next(row for row in rows if row["tag"] == "e29")
new = dict(old)
new["ships_from"] = "A (re-rolled 2026-10-28)"
new["selected_environment_steps"] = clip["selected_environment_steps"]
new["checkpoint_sha256"] = digest
new["kit_add_auc"] = clip["selected_kit_score"]["kit_add_auc"]
new["checkpoint_dir"] = str(dest)
new["lost_in_the_reboot"] = {
    "note": "the rollout this row replaced existed only in the wiped scratchpad; this is a "
    "re-roll of the same recipe and selection rule, not the same rollout",
    "ships_from": old["ships_from"],
    "selected_environment_steps": old["selected_environment_steps"],
    "checkpoint_sha256": old["checkpoint_sha256"],
    "kit_add_auc": old["kit_add_auc"],
}
rows[rows.index(old)] = new
entry["e29_rerolled"] = {
    "why": "the 2026-10-28 reboot wiped /tmp/claude-1000, and e29's rollout and checkpoint "
    "existed only there",
    "run_record": str(record_path),
    "selection_rule": clip["selection_reason"],
    "kit_add_auc_before": old["kit_add_auc"],
    "kit_add_auc_after": new["kit_add_auc"],
}
entry_path.write_text(json.dumps(entry, indent=1) + "\n")
print(
    f"e29 row rewritten: {old['selected_environment_steps']} -> "
    f"{new['selected_environment_steps']} steps, kit AUC {old['kit_add_auc']:.4f} -> "
    f"{new['kit_add_auc']:.4f}"
)
PYEOF
fi

# ---------------------------------------------------------------- 4: restage all 30
[ "$DRY" = 1 ] && { echo "DRY: would restage 30 and pack"; exit 0; }
run_v2d restage-entry --entry-record "$ENTRY" --checkpoints "$BACKUP" --released "$DATA" \
  --out "$OUT" --json "$S/restage_record.json" || exit 3

# ---------------------------------------------------------------- 5: pack both tracks
SUBMIT=/home/jetson3/projects/v2d-submit
URL="https://github.com/luaiabuelsamen/v2d-track2/commit/$COMMIT"
for TIER in tier2 tier1; do
  echo "== packing $TIER"
  ( cd "$SUBMIT" && env -u PYTHONPATH "$REPO/.venv311/bin/python" -I -c \
      "import sys; sys.path.insert(0,'src'); sys.argv=['v2d']+sys.argv[1:]; from v2d.cli import main; raise SystemExit(main())" \
      pack --tier "$TIER" --kit-root "$KIT" --dataset-root "$DATA" \
      --rollout-root "$OUT/rollouts" --frame-references "$OUT/frame_refs" \
      --work-dir "$OUT/pack_$TIER" --code-commit "$URL" \
      -o "$OUT/pack_${TIER}_record.json" ) > "$S/pack_$TIER.log" 2>&1 \
    || { echo "FATAL: packing $TIER failed; see $S/pack_$TIER.log" >&2; tail -5 "$S/pack_$TIER.log" >&2; exit 3; }
done

echo
echo "== the entry, not uploaded"
$PY - "$OUT" "$COMMIT" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path

out, commit = Path(sys.argv[1]), sys.argv[2]
for tier in ("tier2", "tier1"):
    record = json.loads((out / f"pack_{tier}_record.json").read_text())
    csv = Path(record["csv"])
    digest = hashlib.sha256(csv.read_bytes()).hexdigest()
    score = record["local_score"]
    print(
        f"{tier}: AUC {score['add_auc']:.5f}  SP-SR {score['spider_sr']:.5f}  "
        f"MP-SR {score['maniptrans_sr']:.5f}  MPPE {score['mppe_cm']:.5f} cm"
    )
    print(
        f"  {record['episodes']} episodes, {record['rows']:,} rows, packer {record['kit_packer']}, "
        f"validator {record['kit_validator']}, submittable {record['submittable']}"
    )
    print(f"  sha256 {digest}")
    print(f"  {csv}")
print(f"commit stamped in every row: {commit}")
PYEOF
