# Wave 3 options, costed

Written 2026-10-27 21:40 UTC, while the second wave's last cell (B4x e26 on `v2d-sw-b4x-b17`)
trains. Nothing here is launched or approved. Each option states what it buys, what it costs at
**$0.68 per box-hour**, and the bar it would be judged against.

## Where we stand

| entry | AUC | SP-SR | MP-SR | MPPE cm |
|---|---|---|---|---|
| Tier 2, submitted 05:18 UTC (first wave) | 0.3908 | 0.5667 | 0.1667 | 17.36 |
| Tier 2, wave-2 ships, packed pre-e26 | **0.4631** | 0.6667 | 0.4667 | 14.14 |
| Tier 1, insurance entry (409,600 steps) | 0.134 | — | — | — |
| CHORD baseline, Tier 1 board | 0.51038 | — | — | — |

The wave-2 numbers come from a real kit pack of the staged entry (30 episodes, 22,989 rows,
validator pass). MP-SR went from 5 of 30 clips to 14.

**Budget.** $187.84 of the $230 authorisation is committed; b17 adds about $1.6 before it is
deleted, leaving roughly **$40 and change**. Every option below except a trimmed (d) therefore
needs Luai to raise the cap, and that is said per option rather than buried.

**Cost model.** 4.4 box-hours per clip (3.5 h of training at ~25k env steps/s plus seven
evaluations) and 0.6 h of bring-up per box: $2.99 per clip, $0.41 per box. It predicted $88 for
the 30-clip combined run, which billed $88.22, and $2.99 for a one-clip box, which billed $2.90.
Boxes are packed to stay under the 24 h deletion authority.

## (a) All 30 clips, precision recipe, 2 seeds, on tetool's bridge_v2

60 cells of 350M steps, 4 clips per box, 15 boxes at 18.2 h each.

- **Cost: $185.64.** Needs +$145 over what is left.
- Buys two things at once that we cannot currently separate: a better reference (bridge_v2) and a
  second seed. The seed spread we measured on e24 was ±0.03 AUC, so a two-seed mean is worth
  having before any single-clip claim is made; bridge_v2 is unmeasured until tetool reports.
- Bar: the two-seed mean per clip beats that clip's wave-2 record, and the adoption rule picks per
  clip as usual, so a worse seed cannot make the entry worse.
- Note it is the only option that re-runs the 20 clips that are already working, which is where
  the least headroom is: 12 of them are within 0.1 AUC of their reference ceiling.

## (b) The nine clips below 0.30, at 700M steps

`e28 e19 e25 e04 e13 e01 e14 e07 e05` — the nine clips whose shipping rollout is under 0.30 AUC
after adoption. One variable bundle: `body_joint_position` weight 1, object weight 5, termination
at 20 cm, double the steps (700M, 14 checkpoints).

- **Cost: $60.79** (9.6 h per clip, 2 clips per box, 4 boxes of 2 and one of 1, 19.8 h per box).
  Needs +$21.
- These nine hold 0.17 of the 0.46 mean between them. Six are drop failures, where the deep dive
  showed the policy loses the object mid-clip rather than tracking it badly, so more steps plus a
  looser termination is the change with the clearest mechanism behind it.
- Bar: per clip, beat its own wave-2 record AUC with no new drop, under the same input-only
  adoption rule. A clip that does not beat its record simply keeps the rollout it has.
- Risk to name: 700M steps is past every curriculum threshold, so stages 6-7 run for 350M steps
  with VOC at zero. That is untested; the 50M-step read on each cell is the abort signal.

## (c) A third seed on the 18 Part A clips

18 cells of 350M steps on the shipping precision recipe, 4 per box, 5 boxes.

- **Cost: $55.90.** Needs +$15.
- Buys variance reduction only: no new mechanism, no new reference. Adoption keeps the best
  rollout per clip, so the expected gain is the mean of the per-clip maxima over three seeds
  rather than one — on the measured ±0.03 spread, worth perhaps +0.01 to +0.02 of mean AUC.
- Bar: same input-only adoption per clip. It cannot lose, which is also why it is the weakest
  option per dollar: $56 for about a third of what option (b)'s nine clips could move.

## (d) A Tier 1 run: 30 clips on the clean tier_1 references

Tier 1 is a third of the overall track rank and our entry there is the 409,600-step insurance
policy at 0.134, which is last. This option trains for Tier 1 properly: the **clean** tier_1
reference as input, precision recipe (`object_position` std 0.08, `object_orientation` std 0.3,
START curriculum on, CLEAN off because the input needs no cleaning), one seed, 349,999,104 steps,
support per clip from the data, z-drop selection, 4 clips per box over 8 boxes at 18.2 h each.

- **Cost: $93.02.** Needs +$53 over what is left. Trimmed to the 12 weakest clips it is **$37.13**
  and fits inside the remaining $40.
- **Pre-registered bar: the 30-clip kit AUC on Tier 1 beats the CHORD baseline's 0.51038.** Below
  that, the run is not adopted and the free Tier 1 pack of the wave-2 rollouts stands.
- Selection and adoption: z-drop per clip, then adoption against the insurance entry on the
  input-only rule. One honest difference from Tier 2: on Tier 1 the input reference *is* the clean
  reference, so "input-only" no longer means "blind to truth" there. It is still the registered
  rule and Tier 1 has no input constraint, but it is not the same guarantee, and the record should
  say so rather than reuse the Tier 2 wording.
- What makes this cheaper than it looks: most of the Tier 1 gap closes for **$0** by packing the
  wave-2 rollouts as a Tier 1 CSV, which scores **0.4631** against the insurance entry's 0.134.
  That pack is validated and sitting on disk. Option (d) is therefore buying the last stretch,
  0.4631 → past 0.51038, not the whole climb — and the clean input is the reason to expect it:
  every wave-2 rollout was trained to track a noisy object, which on Tier 1 is a handicap.

### (d) is staged and ready; the launch is one command

Staged 2026-10-27 22:05 UTC, nothing launched:

- `$S/t1_precision` — 30 clips from our own clean tier_1 retarget, 182 files, 859 MB, 30 support
  USDAs staged where `_discover_support_surface` looks (14 resolve to a supported scene, the same
  14 as Tier 2). Manifest `$S/t1_precision.md5` verifies all 182 files.
- `$S/tier1_episode_map.txt` — the 30 episode/sequence pairs.
- `provenance.json` records the reference dataset per clip, which is the released root; the entry
  packer converts a Tier 1 run against the tier_1 pose via `v2d stage-entry-wave2 --tier tier1`.

Two hardcoded `--tier tier2` calls had to go first: `brev/score_pulled.sh` converted and scored
every pulled checkpoint against Tier 2, so a Tier 1 run would have been scored against a noisy
input its policies never saw, and the number would have looked comparable. Both scripts now take
`--tier`, defaulted to tier2, and a test pins it.

```
cd ~/projects/v2d-challenge && brev/launch_t2_combined.sh --go \
  --map $S/tier1_episode_map.txt --stage $S/t1_precision --out $S/wave3/run_T1 \
  --tier tier1 --name-prefix v2d-t1p-b \
  --arm-flags start_curriculum,orientation_reward,object_position \
  --arm-env V2D_OBJECT_POSITION_STD=0.08 \
  --boxes 8 --clips-per-box 4 --create-stagger 20
```

`object_position` std 0.08 and `orientation_reward` std 0.3 are the precision recipe; START is the
curriculum patch; no CLEAN and no FING because the tier_1 references are already clean and carry
our own finger solve. 10-minute pulls, dead-runner check, auto-replacement and the 24 h deletion
authority come from the launcher unchanged. **This is not approved: it needs the cap raised from
$230 by at least $53 for all 30 clips, or trimming to 12 clips at $37.13.**

## Ranking, by what the money moves

1. **(d) trimmed, $37** — the only option that fits the remaining budget, aimed at the third of
   the rank where we are last, with the clearest mechanism (clean input instead of noisy).
2. **(b), $61** — the largest AUC headroom per dollar on Tier 2, with a named failure mode.
3. **(d) full, $93** — the same idea as (d) trimmed with all 30 clips.
4. **(a), $186** — two variables at once, and mostly re-runs clips with little headroom.
5. **(c), $56** — cannot lose, cannot win much.

Free and already done, independent of all of the above: the Tier 1 pack of the wave-2 ships
(`--tier tier1`, byte-identical CSV to the Tier 2 entry, validator pass, 0.4631) and the Tier 2
wave-2 entry itself. Neither is submitted.
