#!/usr/bin/env bash
# Arm flags: compose one-variable recipe changes into a cell without forking the runner.
#
# An "arm" used to mean only which parquet the cell trained on (raw or clean). The arms we want
# next change the recipe instead, and each one is a patch to the flash_chord source in the training
# container plus, sometimes, a hydra override. ARM_FLAGS names them as a comma-separated list in
# any order; ARM_FLAGS_ORDER below is the canonical order, so the same SET of flags always produces
# the same slug, the same overrides and the same summary line, whatever order they were written in.
#
# Sourced by the run scripts. arm_flags_normalise / _slug / _overrides are pure and are unit-tested
# on the Jetson; arm_flags_apply additionally needs run() and CID (the flash_chord container).
#
# Why patches rather than a fork: flash_chord ships inside the image and we do not build it. A
# patch that refuses to apply (brev/patch_*.py -- atomic, idempotent, and fatal on changed upstream)
# is the only way to know the box is running the code we think it is. A cell either carries the
# change or dies before training, which is the opposite of the failure that cost us three cells.

ARM_FLAGS_ORDER="start_curriculum orientation_reward finger_targets"

# arm_flags_patch_script <flag>  -- flag names are the experiment's vocabulary, file names are the
# patch's, and they are kept apart on purpose: renaming a script must not rename an arm on the
# record.
arm_flags_patch_script() {
  case "$1" in
    start_curriculum)   echo "patch_start_curriculum.py" ;;
    orientation_reward) echo "patch_object_orientation.py" ;;
    finger_targets)     echo "patch_finger_targets.py" ;;
    *) return 2 ;;
  esac
}

# arm_flags_normalise [csv]  -- one validated flag per line, in canonical order, deduplicated.
# Empty input prints nothing and succeeds: no flags is the stock recipe, not an error.
arm_flags_normalise() {
  local csv="${1:-}" flag out=()
  csv=" ${csv//,/ } "
  for flag in $csv; do
    case " $ARM_FLAGS_ORDER " in
      *" $flag "*) ;;
      *)
        echo "FATAL: unknown arm flag '$flag'. Known flags: $ARM_FLAGS_ORDER" >&2
        return 2 ;;
    esac
  done
  for flag in $ARM_FLAGS_ORDER; do
    case "$csv" in *" $flag "*) out+=("$flag") ;; esac
  done
  [ "${#out[@]}" -gt 0 ] || return 0
  printf '%s\n' "${out[@]}"
}

# arm_flags_files [csv]  -- the patch scripts these flags need, one basename per line. This is the
# one place that knows flag -> file, and the launcher asks it rather than carrying its own list: a
# patch script the runner needs but the box does not have is the failure that killed both cells of
# the 2026-10-25 relaunch inside a second.
arm_flags_files() {
  local flags flag
  flags="$(arm_flags_normalise "${1:-}")" || return 2
  while read -r flag; do
    [ -n "$flag" ] && arm_flags_patch_script "$flag"
  done <<< "$flags"
  return 0
}

# arm_flags_slug [csv]  -- "" for the stock recipe, else "-sc", "-sc-or", ... Appended to the cell
# directory and the instance name so a flagged cell cannot overwrite the stock cell it is measured
# against.
arm_flags_slug() {
  local flags flag slug=""
  flags="$(arm_flags_normalise "${1:-}")" || return 2
  while read -r flag; do
    case "$flag" in
      start_curriculum)   slug="${slug}-sc" ;;
      orientation_reward) slug="${slug}-or" ;;
      finger_targets)     slug="${slug}-ft" ;;
    esac
  done <<< "$flags"
  printf '%s' "$slug"
}

# arm_flags_overrides [csv]  -- extra hydra overrides for the train command, space-separated.
arm_flags_overrides() {
  local flags flag out=""
  flags="$(arm_flags_normalise "${1:-}")" || return 2
  while read -r flag; do
    case "$flag" in
      # The curriculum patch moves the deployment-exact starts earlier, but a random-frame reset
      # still sets the applied VOC scale to reset.reset_voc_scale -- stock 1.0 -- and holds it for
      # the first 40 of the 50 freeze steps before ramping to the curriculum value over the last 10
      # (lifecycle/reset.py:advance_recon_body_reset). In the zero-VOC stages that re-places the
      # object the curriculum has just stopped placing, so the policy keeps being handed a grasp it
      # never has to establish. 0.0 is the late-stage curriculum value.
      #
      # Hydra's value is one number for the whole run, so this also removes the assist from the
      # random resets of stages 1-6. The curriculum file itself keeps those stages untouched, as
      # asked; if the early stages turn out to need the assist, reset.voc_decay_steps=50 is the
      # stage-safe alternative -- it makes the assist fade from reset_voc_scale to the curriculum
      # value across the whole freeze, which is a no-op while the curriculum value is 1.0 and a
      # fade only once it is 0.0.
      start_curriculum)   out="$out reset.reset_voc_scale=0.0" ;;
      orientation_reward) ;;  # the weight and std live in the config the patch edits
      finger_targets)     ;;  # whatever it needs is the patch's business
    esac
  done <<< "$flags"
  printf '%s' "${out# }"
}

# arm_flags_apply <csv> <label>
# Applies every flag's patch inside the container and prints the hydra overrides on stdout, with a
# line per patch on stderr. Aborts the cell if any patch is missing, refuses to apply, or reports
# nothing: a half-patched tree trains something nobody designed.
arm_flags_apply() {
  local csv="${1:-}" label="${2:-cell}" flags flag script here args report
  flags="$(arm_flags_normalise "$csv")" || return 2
  if [ -z "$flags" ]; then
    echo "arm flags: $label -> stock recipe, no patches" >&2
    printf ''
    return 0
  fi
  here="$(dirname "${BASH_SOURCE[0]}")"
  while read -r flag; do
    script="$here/$(arm_flags_patch_script "$flag")"
    if [ ! -f "$script" ]; then
      {
        echo "FATAL: $label -- arm flag '$flag' needs $(basename "$script"), which this checkout"
        echo "       does not have. Either the patch has not been written yet or the launcher did"
        echo "       not ship it; both are reasons not to train. Flags asked for: $csv"
      } >&2
      return 1
    fi
    args=""
    if [ "$flag" = orientation_reward ] && [ -n "${V2D_OBJECT_ORIENTATION_STD:-}" ]; then
      args="--std ${V2D_OBJECT_ORIENTATION_STD}"
    fi
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "arm flags: $label | would apply $(basename "$script") $args" >&2
      continue
    fi
    docker cp "$script" "$CID:/tmp/$(basename "$script")" >/dev/null 2>&1 || {
      echo "FATAL: $label -- could not copy $(basename "$script") into container $CID" >&2
      return 1
    }
    report="$(run "python /tmp/$(basename "$script") /workspace/src/flash_chord $args" 2>&1)" || {
      {
        echo "FATAL: $label -- $(basename "$script") refused to apply:"
        printf '%s\n' "$report" | sed 's/^/       /'
      } >&2
      return 1
    }
    case "$report" in
      *"patched:"*) ;;
      *)
        {
          echo "FATAL: $label -- $(basename "$script") reported no patch at all:"
          printf '%s\n' "$report" | sed 's/^/       /'
        } >&2
        return 1 ;;
    esac
    printf '%s\n' "$report" | sed "s/^/arm flags: $label | /" >&2
  done <<< "$flags"
  printf '%s' "$(arm_flags_overrides "$csv")"
}
