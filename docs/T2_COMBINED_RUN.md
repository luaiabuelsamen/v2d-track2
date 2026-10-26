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
| FING | finger targets, from interp's retarget | **landed 08:02 UTC**, 30/30 clips, 150/150 md5 verified |
| CLEAN | bridge-cleaned object poses, from tetool | **landed**, retargeted here 30/30, 97/97 md5 verified |

START and ORI are code; both patches refuse to apply to an upstream they do not recognise, are
idempotent, and are applied once per box before the first clip, so a box that cannot carry the
recipe never trains. FING and CLEAN are **references**, not code: they change the parquet a clip
trains against.

Nothing is merged column-wise. The retargeter solves the whole body against the object trajectory,
so finger joints from a raw-object solve would put the fingers where the *noisy* object was. A clip
comes from exactly one solve and `provenance.json` records which.

## What landed, and what the stage proved about it

Both references are in. The stage takes all 30 clips from interp's finger tree, which was solved on
top of our retarget of tetool's `bridge_v1`:

| checked | result |
|---|---|
| finger tree md5 | 150/150 files match the handover's own list |
| what the finger solve changed | exactly the 14 Dex3 joints (`left/right_hand_{index,middle,thumb}_*`), on every clip; no other column, no other joint |
| which object trajectory each clip carries | 29 clips reproduce the cleaned retarget **exactly** (delta 0.0 m); e01 reproduces the released one, which is correct — the cleaner flagged e01, so interp solved its fingers on the released poses |
| support | 14/30 resolve to a supported scene, unchanged |

That last row is why the convert root is now recorded **per clip** and not per source. One tree
carries two references, so `v2d match-object-reference` measures which one each clip holds and
`provenance.json` gets a `ref_dataset_by_clip` map that `brev/score_pulled.sh` reads first: 29 clips
convert against `bridge_v1`, e01 against the release. Keying it by source would have scored e01
against a reference its policy never tracked, which is the silent kind of wrong — on B3 the same
mistake moved MPPE by 0.36 cm and only `rollout-to-npz`'s frame check caught it. A clip whose object
trajectory matches **no** declared source is fatal: that is the "fingers solved against the noisy
object" failure the handover warns about, and its kit score would mean nothing.

## What was blocked, and exactly what unblocked it

| arm | needs | where it is read from | if it never lands |
|---|---|---|---|
| CLEAN | one retargeted G1 tree of bridge-cleaned clips, plus the object npz they were retargeted from (the kit score has to be converted against the same reference the policy trained on) | path in `~/projects/_ops/briefs/clean_refs_path.txt`; clips it flags listed there or passed as `--skip clean=…` | that clip falls through to the released-input retarget |
| FING | one retargeted G1 tree with non-zero finger joints, solved **on top of CLEAN's object poses** so the two compose | path in `~/projects/_ops/briefs/` | the run goes without finger targets |

Both landed on 2026-10-26: `clean_refs_path.txt` as a drop-in `--dataset-root` (retargeted here in
~70 min on two CPU workers, worst rigid-fit residual 0.0 mm / orientation 0.042°, handed back to
interp at `~/projects/_ops/briefs/clean_retarget_path.txt`), and `finger_refs_path.txt` at 08:02 UTC
as a retargeted tree built on that retarget.

All 14 finger joints are identically **0** in both released tiers, so FING is the one arm whose
channel is currently absent rather than noisy — which is why it is worth a slot at all.

## The run

```bash
# 1. assemble the references (~30 s, verified by md5 and by the trainer's own support resolver).
#    --from-briefs reads the handover paths itself, so no path is typed and none can be mistyped.
brev/stage_t2_combined.sh --from-briefs

# 2. the run itself. The npz each source must be scored against comes from the stage's
#    provenance.json, so there is nothing to keep in sync by hand.
brev/launch_t2_combined.sh --go
```

`--from-briefs` reads `~/projects/_ops/briefs/`, in priority order:

| file | what it names | shape |
|---|---|---|
| `finger_refs_path.txt` | interp's finger-target tree | `<tree root>` and optionally `<npz dir>` and `<flagged episodes, comma-separated>` |
| `clean_refs_path.txt` | tetool's bridge-cleaned tree | same three fields |

A file that is absent is a source that is skipped, so the same command stages whatever has landed
and falls through to the released-input retarget for the rest. Each stage prints which source every
clip came from and records it per clip in `provenance.json`.

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

## A shipping run, not an attribution run

**Luai chose this knowingly.** The combined recipe changes four things at once, so whatever comes
back, this run cannot say which of START, ORI, FING or CLEAN earned it. It is the submission
attempt: one seed, every clip, the best recipe we can assemble, inside the remaining budget and the
nine days to the freeze. Per-arm attribution is what the four-clip grid was for, and the grid was
dropped in favour of shipping. If the run comes back below the bar, the honest next question is
which arm hurt, and answering it costs another full-budget run per arm — that is the price of this
choice, accepted up front.

## The bar

The four-clip stock record, z-drop-selected, **kit scorer**, from
`results/checkpoint_selection.json` and `docs/AB4_DESIGN.md`:

| | stock, four clips, z-drop-selected |
|---|---|
| mean **kit** AUC | **0.36635** |
| mean kit MPPE | 13.04 cm |
| clips with no drop-free checkpoint | 1 of 4 |

The bar is **kit AUC against the kit-scored stock selection, 0.36635**, on e03/e09/e18/e24 — the
four clips that have a stock record — plus no increase in the number of clips with no drop-free
checkpoint. The kit scorer is what the board uses, which is why the bar lives there.

The evaluator's AUC for the identical four checkpoints is 0.274525. It is on the record for
continuity with the per-checkpoint tables, and it is **not** the bar: using 0.2745 as a kit-AUC
threshold would adopt a recipe well below stock.

## Safety

Each box carries two deletion authorities — `gate_launch.sh`'s at 24 h and an independent backstop
at 24.5 h, so the launcher's own fires last. A box never starts a clip its remaining authority
cannot finish, because a box deleted mid-clip bills for the whole clip and returns nothing. Results
are pulled every 10 minutes, not at the end. Liveness is probed every 2 minutes: the runner process
gone with no `DONE`, or `train.log` not growing for 15 minutes, ends the clip loudly. References
are verified on the box by md5, not by file count, because a truncated parquet passes a count and
then fails with "Parquet magic bytes not found in footer" four hours in.

## Addendum, 2026-10-26 20:30 UTC — the four-clip read: PASS

The four bar clips finished and were kit-scored (`results/t2_combined_bar_read.json`).

| clip | stock pick | stock kit AUC | combined pick | combined kit AUC | Δ | drop-free, stock → combined |
|---|---|---|---|---|---|---|
| e03 | 99M | 0.3556 | 299M | **0.6589** | +0.303 | 0/7 → 2/7 |
| e09 | 349M | 0.2409 | 299M | 0.2448 | +0.004 | 4/7 → 3/7 |
| e18 | 349M | 0.3826 | 249M | **0.7362** | +0.354 | 7/7 → 7/7 |
| e24 | 299M | 0.4863 | 299M | **0.7574** | +0.271 | 3/7 → 5/7 |
| **mean** | | **0.36635** | | **0.59932** | **+0.233** | 1 of 4 → **0 of 4** |

Mean kit MPPE 13.04 → **3.61 cm**. Both registered conditions pass: the mean beats the bar by
0.233, and the count of clips with no drop-free checkpoint fell from 1 to 0 rather than rising.

Two things this does **not** say. It does not attribute the gain — four variables moved at once,
which is what a shipping run buys. And e09 is flat, so the mean rests on three clips; with one
evaluation world per checkpoint a single clip's AUC carries real variance.
