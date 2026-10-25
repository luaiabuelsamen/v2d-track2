#!/usr/bin/env bash
# Decide scene.support per clip, the way NVIDIA's own code resolves it.
#
# Sourced by the run scripts, which must already define run() (exec in the flash_chord container).
#
# Why this file exists: all three run scripts used to pass a hardcoded scene.support=false. That
# silently removed the table from the scene, so on any clip whose object rests on a surface the
# object fell at reset and the episode terminated on frame ~0. It cost us three full-length runs
# (results/ab350_raw_baseline.json) and the ~12 near-zero clips in the Tier 1 entry. The flag is
# not a global preference; it is a per-clip property of the data.
#
# The rule is the one gate_on_instance.sh already carried in support_flag_for: support is on iff
# that clip's support USDA declares collision primitives. 16 of the 30 released support files are
# the empty 125-byte stage (watering_can is one -- it is manipulated in mid-air), and turning
# support on for those makes build_scene raise "support surface contains no Cylinder or Cube
# collision shapes" and kills the clip after ~90 s of scene build. Both directions are wrong, so
# the flag has to be read from the file.
#
# Upstream resolves the file in robotic_grounding.tasks.scene_utils.scene_config
# ._discover_support_surface(motion_file): walk up from the parquet to the nearest
# `sequence_id=<seq>` ancestor, then look in `<that>.parent.parent/reconstructed_stage` for
# `<seq>_<robot>_support.usda` and fall back to `<seq>_support.usda`. We ask that function
# itself rather than reimplementing it, and only fall back to the mirrored path logic if it
# cannot be imported -- a silent disagreement between our guess and the trainer's resolver is
# exactly the failure this is here to prevent.

# support_decide <container-parquet-path>
# Prints two lines: "flag=scene.support=<true|false>" and "usda=<path or NONE>".
# Returns non-zero if the resolver finds nothing, which always means a staging bug: every clip
# ships a support USDA, even the ones with no primitives.
support_decide() {
  local parquet="$1" payload out
  payload="$(base64 -w0 <<'PYEOF'
import os, sys
from pathlib import Path

parquet = os.environ["V2D_PARQUET"]


def mirrored(motion_file):
    """The resolver's logic, mirrored, for the case where the import fails."""
    path = Path(motion_file).resolve()
    parents = [path] + list(path.parents)
    robot = next((p.name.split("=", 1)[1] for p in parents if p.name.startswith("robot_name=")), None)
    for parent in parents:
        if parent.name.startswith("sequence_id="):
            seq = parent.name.split("=", 1)[1]
            stage = parent.parent.parent / "reconstructed_stage"
            names = ([f"{seq}_{robot}_support.usda"] if robot else []) + [f"{seq}_support.usda"]
            for name in names:
                if (stage / name).exists():
                    return str(stage / name)
            return None
    return None


how = "upstream _discover_support_surface"
try:
    from robotic_grounding.tasks.scene_utils.scene_config import _discover_support_surface
    resolved = _discover_support_surface(parquet)
    check = mirrored(parquet)
    if resolved != check:
        print(f"WARNING: our mirrored path logic disagrees with the resolver: {check!r}", file=sys.stderr)
except Exception as error:  # noqa: BLE001 - any import problem falls back, loudly
    how = f"mirrored path logic ({type(error).__name__}: {error})"
    resolved = mirrored(parquet)

print(f"how={how}", file=sys.stderr)
if not resolved:
    print("usda=NONE")
    print("flag=scene.support=false")
    sys.exit(3)

text = Path(resolved).read_text(errors="replace")
primitives = sum(text.count(f"def {kind}") for kind in ("Cylinder", "Cube", "Sphere"))
print(f"primitives={primitives} bytes={Path(resolved).stat().st_size}", file=sys.stderr)
print(f"usda={resolved}")
print(f"flag=scene.support={'true' if primitives else 'false'}")
PYEOF
)"
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'flag=scene.support=true\nusda=DRYRUN\n'
    return 0
  fi
  out="$(run "echo '$payload' | base64 -d > /tmp/v2d_support_flag.py && V2D_PARQUET='$parquet' python /tmp/v2d_support_flag.py")"
  local status=$?
  printf '%s\n' "$out" | tr -d '\r'
  return $status
}

# support_flag_or_die <container-parquet-path> <label>
# Prints the hydra override to stdout and a one-line explanation to stderr. Aborts the run if the
# support file does not resolve, because training on a scene we cannot describe is how we burned
# three 350M-step cells.
support_flag_or_die() {
  local parquet="$1" label="${2:-clip}" report flag usda
  report="$(support_decide "$parquet")" || {
    {
      echo "FATAL: $label -- the support USDA did not resolve for"
      echo "       $parquet"
      echo "       _discover_support_surface returned None, so the trainer would build a scene with"
      echo "       no support surface. Every released clip ships a support USDA (16 of 30 are the"
      echo "       empty stage), so None means it was not staged into"
      echo "       <dataset-root>/whole_body/reconstructed_stage/<sequence_id>_support.usda."
      echo "       Fix the staging (brev/stage_ab_refs.sh) and relaunch; do not train like this."
      printf '%s\n' "$report" | sed 's/^/       /'
    } >&2
    return 1
  }
  flag="$(printf '%s\n' "$report" | sed -n 's/^flag=//p')"
  usda="$(printf '%s\n' "$report" | sed -n 's/^usda=//p')"
  [ -n "$flag" ] || { echo "FATAL: $label -- could not read a support flag from the preflight" >&2; return 1; }
  echo "support: $label -> $flag (from $usda)" >&2
  printf '%s' "$flag"
}
