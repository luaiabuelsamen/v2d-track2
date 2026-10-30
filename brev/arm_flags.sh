#!/usr/bin/env bash
# Arm flags: compose one-variable recipe changes into a cell without forking the runner.
#
# An "arm" used to mean only which parquet the cell trained on (raw or clean). The arms we want
# next change the recipe instead, and each one is a patch to the flash_chord source in the training
# container plus, sometimes, a hydra override. ARM_FLAGS names them as a comma-separated list in
# any order; ARM_FLAGS_ORDER below is the canonical order, so the same SET of flags always produces
# the same slug, the same overrides and the same summary line, whatever order they were written in.
#
# The order is also load-bearing for two pairs. patch_start_curriculum.py verifies the stock
# thresholds line before it touches anything, so it must run BEFORE patch_voc_curriculum.py
# rewrites that line: start_curriculum therefore stays first and voc_curriculum last. And
# patch_object_orientation.py anchors its term names on the tuple ending in `force_closure`, which
# patch_contact_role.py appends two more names to, so orientation_reward must run BEFORE
# contact_role. Reordering either pair turns a working arm into a refused patch and a dead cell --
# tests/test_patch_second_wave.py and tests/test_patch_contact_role.py pin both directions.
#
# Sourced by the run scripts. arm_flags_normalise / _slug / _overrides are pure and are unit-tested
# on the Jetson; arm_flags_apply additionally needs run() and CID (the flash_chord container).
#
# Why patches rather than a fork: flash_chord ships inside the image and we do not build it. A
# patch that refuses to apply (brev/patch_*.py -- atomic, idempotent, and fatal on changed upstream)
# is the only way to know the box is running the code we think it is. A cell either carries the
# change or dies before training, which is the opposite of the failure that cost us three cells.

ARM_FLAGS_ORDER="start_curriculum orientation_reward finger_targets object_position body_joint_position contact_role voc_curriculum warm_start"

# arm_flags_patch_script <flag>  -- flag names are the experiment's vocabulary, file names are the
# patch's, and they are kept apart on purpose: renaming a script must not rename an arm on the
# record.
arm_flags_patch_script() {
  case "$1" in
    start_curriculum)   echo "patch_start_curriculum.py" ;;
    orientation_reward) echo "patch_object_orientation.py" ;;
    finger_targets)     echo "patch_finger_targets.py" ;;
    object_position)    echo "patch_object_position.py" ;;
    body_joint_position) echo "patch_body_joint_position.py" ;;
    contact_role)       echo "patch_contact_role.py" ;;
    voc_curriculum)     echo "patch_voc_curriculum.py" ;;
    # Last in the order on purpose: it edits config.py/runner.py/learner.py, which no other patch
    # touches, but a run that warm-starts must have its OBJECTIVE patches already in place so the
    # fine-tune optimises the same reward the actor was trained on.
    warm_start)         echo "patch_warm_start.py" ;;
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
      object_position)    slug="${slug}-op" ;;
      body_joint_position) slug="${slug}-bj" ;;
      contact_role)       slug="${slug}-cr" ;;
      voc_curriculum)     slug="${slug}-vc" ;;
      warm_start)         slug="${slug}-ws" ;;
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
      object_position)    ;;  # std and weight live in the config and curriculum the patch edits
      body_joint_position) ;;  # weight and std live in the config and curriculum the patch edits
      # The weights and the std live in the config the patch edits; the ROLES are data, read at
      # training time from $V2D_CONTACT_ROLES inside the container, which run_tier2_full.sh puts
      # into the train command because docker exec does not inherit the box's environment.
      contact_role)       ;;
      voc_curriculum)     ;;  # the thresholds live in the curriculum the patch edits
    esac
  done <<< "$flags"
  printf '%s' "${out# }"
}

# arm_flags_apply <csv> <label>
# Applies every flag's patch inside the container and prints the hydra overrides on stdout, with a
# line per patch on stderr. Aborts the cell if any patch is missing, refuses to apply, or reports
# nothing: a half-patched tree trains something nobody designed.
# Did a patch script actually do something? The guard exists because a script that silently
# changes nothing would put the stock recipe on the record under an arm's name.
#
# It must accept BOTH report dialects, and requiring the colon form cost us 15 clips' worth of
# bring-up on 2026-10-27: the per-file scripts print "already patched: <path>", while the
# parameterised ones (object position, VOC) print "already patched at std 0.08, weight unchanged"
# with no colon and return before any per-file line. On the second clip of a box the container is
# already patched, so that no-op report is the NORMAL case -- e11 on a-b05 and e25 on b1-b01 both
# died on it within a minute of starting, and every later clip on an A, B1 or B2 box would have.
arm_flags_report_ok() {
  # A patch script must leave evidence that its change is in place -- either it applied the change
  # now, or it found it already applied. The second case is the normal one on a RESTART, and the
  # scripts word it differently: patch_object_* say "already patched", patch_warm_start.py says
  # "already carries the warm-start keys". Matching only *patched* made every restart of a
  # warm-start box fail with "reported no patch at all" and left the box idling at 0% GPU, which is
  # how v2d-t1f-1 burned time on 2026-10-29. Both wordings count.
  case "$1" in
    *patched*|*"already carries"*) return 0 ;;
    *) return 1 ;;
  esac
}

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
    case "$flag" in
      orientation_reward)
        # Two knobs, as object_position already had. B4x moved the std alone and recovered 6% of
        # the oracle bound, because the term's weight stayed at 1.0 against body_joint_position's
        # 5.0: gradient without weight. Both are passed explicitly so the cell log shows the arm.
        [ -n "${V2D_OBJECT_ORIENTATION_STD:-}" ] && args="--std ${V2D_OBJECT_ORIENTATION_STD}"
        [ -n "${V2D_OBJECT_ORIENTATION_WEIGHT:-}" ] \
          && args="$args --weight ${V2D_OBJECT_ORIENTATION_WEIGHT}" ;;
      object_position)
        # Two knobs, one patch: Part A moves the std alone, the tail arm moves the weight alone.
        # Both are passed explicitly so the applied arm is readable in the cell log.
        [ -n "${V2D_OBJECT_POSITION_STD:-}" ] && args="--std ${V2D_OBJECT_POSITION_STD}"
        [ -n "${V2D_OBJECT_POSITION_WEIGHT:-}" ] \
          && args="$args --weight ${V2D_OBJECT_POSITION_WEIGHT}" ;;
      body_joint_position)
        [ -n "${V2D_BODY_JOINT_POSITION_WEIGHT:-}" ] \
          && args="--weight ${V2D_BODY_JOINT_POSITION_WEIGHT}"
        [ -n "${V2D_BODY_JOINT_POSITION_STD:-}" ] \
          && args="$args --std ${V2D_BODY_JOINT_POSITION_STD}" ;;
      contact_role)
        [ -n "${V2D_CONTACT_ROLE_WEIGHT:-}" ] && args="--weight ${V2D_CONTACT_ROLE_WEIGHT}"
        [ -n "${V2D_CONTACT_ROLE_STD:-}" ] && args="$args --std ${V2D_CONTACT_ROLE_STD}"
        [ -n "${V2D_CONTACT_CONSISTENCY_WEIGHT:-}" ] \
          && args="$args --consistency-weight ${V2D_CONTACT_CONSISTENCY_WEIGHT}"
        [ -n "${V2D_BODY_WEIGHT_IN_CONTACT:-}" ] \
          && args="$args --body-weight-in-contact ${V2D_BODY_WEIGHT_IN_CONTACT}" ;;
      voc_curriculum)
        [ -n "${V2D_VOC_ZERO_AT:-}" ] && args="--zero-at ${V2D_VOC_ZERO_AT}" ;;
      warm_start)
        : ;;  # no patch-time arguments: the actor path and start step are run-time config keys
    esac
    if [ "${DRY_RUN:-0}" = 1 ]; then
      echo "arm flags: $label | would apply $(basename "$script") $args" >&2
      continue
    fi
    # Through stdin, NOT `docker cp`. The training container is created with the creating ssh
    # session's agent socket bind-mounted (/tmp/ssh-XXXX/agent.NNN). Once that session ends the
    # socket is gone, and every `docker cp` into the container fails for good with
    # "mount ... /ssh-agent, flags: 0x5000: not a directory" -- it worked for the first clip of
    # each box on 2026-10-26 and then failed for every clip after, with the runner aborting
    # before training. Writing the file over an exec'd cat touches no mounts.
    docker exec -i "$CID" bash -lc "cat > /tmp/$(basename "$script")" < "$script" 2>/dev/null || {
      echo "FATAL: $label -- could not write $(basename "$script") into container $CID" >&2
      return 1
    }
    report="$(run "python /tmp/$(basename "$script") /workspace/src/flash_chord $args" 2>&1)" || {
      {
        echo "FATAL: $label -- $(basename "$script") refused to apply:"
        printf '%s\n' "$report" | sed 's/^/       /'
      } >&2
      return 1
    }
    if ! arm_flags_report_ok "$report"; then
      {
        echo "FATAL: $label -- $(basename "$script") reported no patch at all:"
        printf '%s\n' "$report" | sed 's/^/       /'
      } >&2
      return 1
    fi
    printf '%s\n' "$report" | sed "s/^/arm flags: $label | /" >&2
  done <<< "$flags"
  printf '%s' "$(arm_flags_overrides "$csv")"
}

# A self-test entry point, so the predicate above can be checked without a box or a container:
#   bash brev/arm_flags.sh --report-ok "already patched at std 0.08, weight unchanged"
# Only when this file is RUN. Sourcing it (the runner and the launcher both do) reaches nothing.
if [ "${BASH_SOURCE[0]}" = "${0}" ] && [ "${1:-}" = "--report-ok" ]; then
  arm_flags_report_ok "${2:-}"
  exit $?
fi
