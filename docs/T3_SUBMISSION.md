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
