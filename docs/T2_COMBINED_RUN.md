# The 30-clip Tier 2 run, combined recipe

Prepared 2026-10-26. **Not launched.** One seed, full budget, every clip, the recipe fixed below
before anything is created. This is a submission run, not an experiment: it decides nothing, and
the only choice made inside it is which checkpoint each clip submits, by a rule written down first.

## The recipe

| | value | where it comes from |
|---|---|---|
| steps | 349,999,104, checkpoint every 49,999,872 | stock, 7 checkpoints a clip |
| seed | 42 | stock |
| worlds | 2048 train, 1 eval | stock; 1 is the submission world count |
| `scene.support` | per clip, from that clip's own USDA | `brev/support_flag.sh`; 14 of 30 have primitives |
| START | `immediate_first_frame_probability` 0.25 from 209,999,872 and 0.50 from 288,749,568, plus `reset.reset_voc_scale=0.0` | `brev/patch_start_curriculum.py` + `brev/arm_flags.sh` |
| ORI | 13th reward term on the object quaternion, weight 1.0, **std 0.3 rad** | `brev/patch_object_orientation.py --std 0.3` |
| FING | finger targets, from interp's retarget | **blocked**, see below |
| CLEAN | bridge-cleaned object poses, from tetool | **blocked**, see below |

START and ORI are code; both patches refuse to apply to an upstream they do not recognise, are
idempotent, and are applied once per box before the first clip, so a box that cannot carry the
recipe never trains. FING and CLEAN are **references**, not code: they change the parquet a clip
trains against.

Nothing is merged column-wise. The retargeter solves the whole body against the object trajectory,
so finger joints from a raw-object solve would put the fingers where the *noisy* object was. A clip
comes from exactly one solve and `provenance.json` records which.

## What is blocked, and exactly what unblocks it

| arm | needs | where it is read from | if it never lands |
|---|---|---|---|
| CLEAN | one retargeted G1 tree of bridge-cleaned clips, plus the object npz they were retargeted from (the kit score has to be converted against the same reference the policy trained on) | path in `~/projects/_ops/briefs/clean_refs_path.txt`; clips it flags listed there or passed as `--skip clean=…` | that clip falls through to the released-input retarget |
| FING | one retargeted G1 tree with non-zero finger joints, solved **on top of CLEAN's object poses** so the two compose | path in `~/projects/_ops/briefs/` | the run goes without finger targets |

Neither file exists yet (checked 2026-10-26 06:20 UTC). If tetool hands over an npz only, it is one
command on this Jetson first, ~4 h on two CPU workers:

```bash
v2d retarget --dataset-root $DATA --object-npz <their npz dir> --tiers tier2   # -> a G1 tree
```

All 14 finger joints are identically **0** in both released tiers, so FING is the one arm whose
channel is currently absent rather than noisy — which is why it is worth a slot at all.

## The run

```bash
# 1. assemble the references, highest-priority source first (~30 s, verified by md5 and by the
#    trainer's own support resolver)
brev/stage_t2_combined.sh --source fing=<interp tree> --source clean=<tetool tree> \
                          --source raw=$S/t2_raw/tier2 --skip clean=<clips tetool flags>

# 2. the run itself
brev/launch_t2_combined.sh --go --npz clean=<tetool npz dir>
```

Without `--go` it prints the plan and creates nothing. Run today against the raw-only stage
(30 clips, 0 problems, 14/30 supported scenes) the plan is:

| | |
|---|---|
| boxes | up to 8 × `massedcompute_A6000_plus` at $0.68/h, ≤ 4 clips each, sequential |
| per box | ~20.7 h of work against a 24 h deletion authority |
| expected spend | **$105.67** |
| worst case | $130.56, if every box bills its full authority |
| budget | $230 authorized, $0.00 of it committed so far ($21.13 was spent under the earlier $30) |

The $90 figure is training alone (30 × 4.4 h × $0.68 = $89.76). Evaluating the 7 checkpoints adds
~0.7 h a clip, which is the difference.

**Time to first results: ~5.4 h after go** — ~13 min bring-up, 4.4 h training, ~39 min for the
seven evaluations, ~5 min of host scoring. e03, e09, e18 and e24 lead the queue on four different
boxes, so the first thing that comes back is directly comparable with the stock four-clip record.
**All 30 clips: ~21 h after go**, if 8 boxes come up.

Boxes take clips from a shared queue rather than a fixed list, because the org's concurrency limit
is unknown — 4 is confirmed, 8 is what this asks for, and `brev create --dry-run` only reports
instance types, not quota, so the limit can only be found by asking. If fewer boxes come up, those
boxes work the queue and the launcher prints the clips nobody claimed, for a second wave. 8 boxes ×
4 clips = 32 slots for 30 clips; at 4 boxes it is 16, and two waves are needed.

## Scoring and the choice of checkpoint

Every checkpoint is kit-scored on pull (`brev/score_pulled.sh`), and one checkpoint per clip is
chosen by the **z-drop** rule — Luai's decision, `v2d pick-checkpoint`'s default, reading only the
noisy input reference the policy trained against.

The order is deliberate: **the selection runs before any kit score exists for that clip.** The kit
scorer compares against the Tier 1 clean truth, so a kit score is ground truth; if selection ran
afterwards nobody could tell whether the choice had been made on the answer. A test plants scores
that would reverse the pick and proves they change nothing.

## What it will be compared against

The four-clip stock record, z-drop-selected, from `results/checkpoint_selection.json`:
**mean kit AUC 0.36635, mean MPPE 13.04 cm, 1 of 4 clips with no drop-free checkpoint** (the same
picks in evaluator terms are 0.274525 and 13.32 cm). The combined recipe is four changes at once,
so this run cannot attribute a gain to any one of them — it is the submission attempt, and the
per-arm attribution is what the held grid was for.

Beware the units: **0.2745 is the evaluator's AUC, 0.36635 is the kit's**, for the identical four
checkpoints. The kit scorer is what the board uses, so any bar must be written in kit AUC.

## Safety

Each box carries two deletion authorities — `gate_launch.sh`'s at 24 h and an independent backstop
at 24.5 h, so the launcher's own fires last. A box never starts a clip its remaining authority
cannot finish, because a box deleted mid-clip bills for the whole clip and returns nothing. Results
are pulled every 10 minutes, not at the end. Liveness is probed every 2 minutes: the runner process
gone with no `DONE`, or `train.log` not growing for 15 minutes, ends the clip loudly. References
are verified on the box by md5, not by file count, because a truncated parquet passes a count and
then fails with "Parquet magic bytes not found in footer" four hours in.
