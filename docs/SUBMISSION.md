# The submission path, and why the Tier 1 fallback is not an entry yet

The Tier 1 entry is the insurance: one valid submission to any track before the
**2026-11-04 17:00 EST** freeze is what keeps the $250 Brev credit. This documents the packaging
path, which is built, proven end to end on real data, and deliberately unable to produce a
submittable CSV from what exists today.

## The path works

`scripts/pack_tier1_submission.py --dry-run` produced
`track2_tier1_NOT_FOR_SUBMISSION.csv`: **22,989 rows, row ids identical to the kit's own
`data/track_2_sample_submission.csv`**, packed by the organisers' `tools/pack_submission.py`
rather than a reimplementation of it, and scored locally through the kit's own metric code
(`results/tier1_package.json`). So the machinery is not the risk. What goes into it is.

## It must not be submitted

| | |
|---|---|
| local score | **AUC 1.0000 · SP-SR 1.0000 · MP-SR 1.0000 · MPPE 0.00005 cm** |
| what the trajectory is | the object carried kinematically by the retargeted G1, no physics |
| residual against the released Tier 1 input | **0.000085 mm**, worst frame of 22,989 |

A perfect score is the diagnosis, not the achievement. Track 2 asks for "the rolled-out object
trajectory" — what a trained policy achieved in simulation. Our retargeter carries the object
with the hands, so its output is a rigid transform of its own input to within a micrometre
(`results/retarget_all.json` already measured 0.0 mm; the packer's frame-0 conversion
independently gives 8.5e-5 mm). For Tier 1 the input *is* the scoring reference, so this CSV is
the answer key handed back. Uploading it would put us first on all four Tier 1 leaderboards with
a trajectory no robot produced.

The provenance makes the same point. The kit wants three digests: the evaluator script's, the
evaluation code's, and the policy checkpoint's. We have none. In the dry run
`eval_script_sha256` is the hash of the kit's *released* `evaluate_policy.py`, which we never
ran, and `eval_code_sha256` and `checkpoint_sha256` carry the literal string
`NOT_FOR_SUBMISSION` — not 64 hex characters, impossible to mistake for a digest, and the packer
copies it into **all 22,989 rows**, so the file labels itself however it is renamed.

## Two guards

1. **Ours.** Without `--dry-run`, `pack_tier1_submission.py` refuses the kinematic path outright
   and prints why. A submittable CSV requires `--rollout-root` pointing at a real FlashCHORD
   export: schema `flash_chord_object_trajectories_v1`, carrying both achieved and reference
   poses, with genuine hashes.
2. **Theirs.** The kit's packer independently refuses a submission that records no
   `eval_script_sha256` or `eval_code_sha256`. The dishonest path is blocked at both ends.

## What is actually needed

One real FlashCHORD rollout per episode. That is the Brev gate (`brev/README.md`), which needs
Luai to redeem the credit and approve the run; its export already matches the schema the packer
wants, so `--rollout-root` is then a one-command swap. **30 clips of rollout is the whole
remaining gap between here and a valid entry** — there is no packaging work left.

## Two bugs found in the kit's packer

Both are the organisers', both are on the submission path, and both are worth knowing before the
freeze:

1. **It contradicts itself about `checkpoint_sha256`.** Line 437 explicitly permits the set to be
   empty, with a comment explaining that per-sequence policies legitimately carry many values.
   Line 488 then rejects *any* blank provenance cell because "Kaggle rejects a submission
   containing nulls". So a submission must record a checkpoint hash for every episode, and the
   permission at 437 is unreachable.
2. **Its error path for that case crashes.** The handler tries to name the offending episodes with
   `int(row.split("_")[1])`, but a row id is `track_2/e000000/w000/f000000/o0`, so the split
   yields `'2/e000000/w000/f000000/o0'` and `int()` raises `ValueError` instead of printing the
   intended message. Anyone submitting without checkpoint hashes gets a traceback rather than the
   explanation. Reproduce by removing `checkpoint_sha256` from the staged parquet metadata in
   `pack_tier1_submission.py`.

Neither blocks us, because we stamp both fields. Worth reporting upstream.
