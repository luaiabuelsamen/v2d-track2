# Tier 3: getting a valid entry, and testing the one idea that could make it good

Pre-registered before launch. Tier 3 is the largest score gain available — we have never
submitted to it and the final score weights all three tiers — so the first objective is a *valid*
entry, not a good one. The second objective, on the same box, is to find out cheaply whether
fine-tuning fixes the reference-swap penalty we measured tonight.

## Why the raw reconstruction cannot just be uploaded

It scores AUC 0.228 / SP-SR 0.333 / MP-SR 0.033 / MPPE 66.6 cm against truth, over the full
22,989-frame grid, and it is honest work. But the kit's packer tolerates an empty
`checkpoint_sha256` and **requires** a non-empty, uniform `eval_code_sha256`
(`tools/pack_submission.py:420-445`). A reconstruction is not an evaluator run, so there is no
honest value for that field, and inventing one is the exact misrepresentation
`src/v2d/submit/tier1.py`'s `NOT_FOR_SUBMISSION` guard was written to prevent. The tier's own
`input_note` says to *"drive your policy from that"*, which is both the rules-clean route and the
one that produces real provenance.

## Rules position, settled rather than assumed

- Alignment uses **one** rigid transform from frame 0 of Tier 1, which FAQ PR 207 permits
  explicitly. Nothing after frame 0 is read, no scale is fitted, no per-frame correction applied.
  The Tier 1 caption is never read; the object identity comes from `meta/episodes_metadata.jsonl`'s
  `object`/`mesh` fields, never `task`.
- Tier 3's configured `data_root` (`track_2/baseline`) is **not released to participants** — the
  hub carries only `tier_1_multiview_caption` and `tier_2_synthetic_noise`. The support USDAs in
  those two are byte-identical, so the scene assets are tier-invariant released data. What Tier 3
  withholds is the *trajectory*, and ours came from video.

## Launch: one box, two passes

`massedcompute_A6000_plus` at $0.68/h. The box is built, patched and holding the assets after
pass 1, so pass 2 pays only for its own GPU time.

| pass | what | clips | est. | cost |
|---|---|---|---|---|
| 1 | eval only: the 30 shipped actors driven by the Tier 3 reference | 30 | ~1.0 h | ~$0.70 |
| 2 | warm-start fine-tune, 50M steps, on the Tier 3 reference | 3 | ~2.0 h | ~$1.40 |

Plus ~0.4 h of bring-up and image build. **Total ~$2.50** against $30.38 of credit. Pass 1 alone
yields a submittable entry, so a failure in pass 2 costs us nothing we needed.

Pass 1 reuses `brev/t1_eval_clean.sh` unchanged — it is parameterised by `ASSETS`/`PLAN`/`OUT`/
`NAME`, and it has just run 30 clips end to end — pointed at the Tier 3 staged tree.

## Pass 2's pre-registered bar

The question is narrow: **does fine-tuning recover what a cold reference swap costs?** Tonight's
Tier 1 run measured the cost — mean AUC 0.46523 → 0.28937 when the reference moved by ~1 cm, with
only 2 of 30 clips improving. The Tier 3 reference is 66 cm away on average, so pass 1 is expected
to be worse still. Fine-tuning is the designed remedy (`brev/patch_warm_start.py`), and it has
never run on a GPU, which is the other reason to try it on three clips first.

Clips: **e26, e18, e08** — the three best-reconstructed (2.65, 3.95, 4.66 cm mean position error;
1.79°, 2.31°, 1.63° mean orientation drift). Chosen deliberately: if fine-tuning cannot help where
the reconstruction is nearly right, it will not help anywhere, and this is the cheapest place to
find that out. All three ship from arm A, so the recipe is one variable rather than two.

Both arms are scored on **the same input reference** (the Tier 3 reconstruction) with the kit
scorer against Tier 1 truth.

> **Adopt fine-tuning for the remaining clips only if the three-clip mean kit AUC beats pass 1's
> three-clip mean on the same clips, and no clip falls by more than 0.02.** Ties go to pass 1,
> which is free. If it clears, fine-tuning the other 27 costs ~$11.6 and still fits the credit.

Pass 2 is also a smoke test with teeth: `brev/run_warm_start.sh` proves the four warm-start config
keys exist on the dataclasses before training, and then greps each clip's log for the patch's own
`warm start: actor from` line, treating its absence as a failed fine-tune. Without that check a
missing key would silently produce a 50M-step run from scratch wearing a fine-tune's name.

## What gets submitted

Whichever of the two the bar selects, per clip, packed with `v2d pack --tier tier3
--frame-references <the reconstruction npz tree>` — Tier 3 releases no input, so that flag is
required and the submit repo refuses without it. Then the kit validator, a local score, and upload
to the four `v2d-challenge-track2-tier3-*` boards, all four of which are already joined.

## Smoke-test evidence, recorded before the spend (2026-10-28)

Everything below is free and was run before the box was created, so a failure would have cost
nothing. Each line is a command that was actually executed, not a plan.

| Check | Result |
|---|---|
| whole-body retarget against the reconstruction, 30 clips, 1 worker | **30/30**, worst rigid-fit residual **0.0001 mm**, worst orientation residual **0.033°**, 0 failures |
| staging (`brev/stage_t1_eval.sh --motion $S/t3/motion/tier2`) | **30/30** refs, **30/30** with a support USDA, manifest **210 lines == 210 files** (nothing written after the manifest) |
| driver preflight | 30/30 refs, 30/30 checkpoints, plan 30 clips, brev logged in |
| `v2d pack --tier tier3` end to end, `--motion-root` stand-in, `--dry-run` | `kit_packer: pass`, `kit_validator: pass`, **22,989 rows** |
| the packer's own frame check on our npz | one rigid transform per episode, max residual **2.72e-07 m / 1.55e-07 rad** |
| the audit on a stand-in | correctly **fails**: "30 episode(s) reproduce the policy input to under 0.1 mm, which no simulated policy does" |

The last row is the one worth keeping. It is the guard that distinguishes a policy rollout from a
replay of its own input, it fires on the stand-in, and it is therefore load-bearing evidence that
the real pack is a real pack. The stand-in's score (AUC 0.2280 / SP-SR 0.3333 / MP-SR 0.0333 /
MPPE 66.60 cm) also reproduces the independently measured raw-reconstruction score exactly, which
cross-checks the packer against the scorer.

Two contract details that were settled by reading rather than by spending:

* **The scorer needs no tier flag.** `build_frames` reads the released Tier 1 trajectory as the
  stand-in for the host's `multi_view_ground_truth` for every tier, and that stand-in is validated
  rather than assumed: our local Tier 2 numbers matched the public board to five decimals.
  `v2d score --tier` is only a label and has no `tier3` choice.
* **Tier 3 is the one tier where our own npz is the frame reference.** `input_path: null` means
  there is no released input to fit the rigid transform against, so both `rollout-to-npz` and the
  kit packer are pointed at `$S/t3/npz_world` — which is in `released_track2_world` by
  construction, anchored at frame 0 with a 0.0000 cm residual on all 30 clips.

## A prediction, registered before pass 1 returns

The clean-Tier-1 experiment measured what happens when a shipped actor is driven by a reference it
was not trained on: AUC **0.465 → 0.289**, and that swap was *tiny* — the clean Tier 1 track sits
about 1 cm from the `t2_clean` reference these actors trained against. The Tier 3 reconstruction is
far further away: mean position error **66 cm**, with 20 of 30 clips losing the object outright.

So the honest prediction is that pass 1 scores **below** the raw reconstruction's 0.228, somewhere
in 0.10–0.20, and that the ten well-tracked clips carry almost all of it. Stating it now matters
for three reasons:

1. **It does not change whether pass 1 is worth running.** Our Tier 3 today is *nothing*. Any valid
   entry is the whole gain, and this is the only route with honest provenance — the raw
   reconstruction cannot supply `eval_code_sha256`, because no evaluator ran.
2. **It is what pass 2 is for.** If the actors are degraded by a reference they never saw, the fix
   is to adapt them to it rather than swap references under them, which is exactly what the
   warm-start patch does. The clean-Tier-1 result is therefore the argument *for* pass 2, not a
   reason to doubt it.
3. **It makes a cheap sanity check possible.** If pass 1 comes back uniformly near zero —
   including on e26, e18 and e08, whose reconstructions track to a few centimetres — that is not a
   policy result, it is a broken pipeline, and the box should be read before anything is packed.
   A spread that tracks reconstruction quality is the signature of a working pipeline.

Point 3 also supersedes most of the value of the still-owed two-clip Tier 2-input control: a
broken eval image depresses every clip uniformly, so a score spread that correlates with
reconstruction quality is itself evidence the image is sound. The control stays available on the
pass-2 box, where it costs two extra evaluations on a box already paid for.
