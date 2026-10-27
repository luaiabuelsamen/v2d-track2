# Final plan

Written 2026-10-27 22:40 UTC under the standing order: **no `brev create`, no launch, no
submission, no push until Luai says go, per item.** Everything below is built, staged and
smoke-tested on free compute, and stops at the create command.

## Budget

The card is off. The budget is the remaining Brev credit: **$38.76 at 21:42 UTC**, less what
`v2d-sw-b4x-b17` burned finishing e26 at $0.82/h. That box is **gone** — its launcher deleted it
after e26's pull and score, and `brev ls` reports no instances in the org. Nothing is running.

| | $ |
|---|---|
| credit at 21:42 UTC | 38.76 |
| b17's last 87 min, to its self-deletion at 23:09 UTC (686 min, $7.77 in all) | −0.99 |
| **credit now** | **≈ 37.77** |
| (1) the two packs | 0.00 |
| (2) T3-VALID | −2.00 |
| **available for (3) T3-FULL** | **≈ 35.77** |

Cross-checked against the box ledger rather than taken on trust: 16,678 box-minutes at $0.68/h is
$189.02, plus the $21.13 spent before this authorisation, is $210.15 of the $250 credit — leaving
$39.85, which agrees with the $37.77 above to within $2 of billing granularity. b17 was the last
box, and `brev ls` is empty.

b17's own launcher deleted it, as designed; `brev ls` confirms the org is empty.

---

## (1) The wave-2 packs — ready to submit on your word

The entry assembled from the second wave's adopted rollouts, packed and validated on this
machine, **not uploaded**.

| | AUC | SP-SR | MP-SR | MPPE cm |
|---|---|---|---|---|
| on the boards now (first wave) | 0.3908 | 0.5667 | 0.1667 | 17.36 |
| **wave-2 ships, packed** | **0.4631** | **0.6667** | **0.4667** | **14.14** |

MP-SR goes from 5 clips of 30 to 14. Both CSVs are byte-identical (sha256 `ae4e2d12…`): the tier
only picks the track name, so the same predictions are a Tier 2 entry and a Tier 1 entry. Tier 1
has no input constraint, and our Tier 1 entry on the board is the 409,600-step insurance run at
**0.134**, so the same file is worth +0.33 AUC there for nothing.

    v2d stage-entry-wave2 --adoption results/second_wave_adoption.json \
      --stage-root $S --first-wave-provenance results/t2_combined_provenance.json \
      --first-wave-results $S/t2_combined_run/results --released $S/data \
      --out $S/t2_entry_wave2
    # then, on ws/submit, once per track:
    v2d pack --tier tier2 --kit-root $KIT --dataset-root $S/data \
      --rollout-root $S/t2_entry_wave2/rollouts \
      --frame-references $S/t2_entry_wave2/frame_refs \
      --work-dir $S/t2_entry_wave2/pack_tier2 --code-commit <pushed commit URL>
    v2d pack --tier tier1 ... --work-dir $S/t2_entry_wave2/pack_tier1 ...

e26 has landed, and these are the **final** packs: 30 episodes, `partial: false`, the kit
validator passing on both tracks, recorded in `results/tier2_entry_wave2.json`. The CSV is
byte-for-byte what the dry pack produced, because e26's B4x rollout lost the adoption check (see
below), so the numbers above are unchanged by its arrival.

**One thing gates the upload, and it is yours:** a **push**. Provenance puts a commit URL in all
22,989 rows, and these CSVs quote `e8ffe8f`, which is `origin/main`. Every commit since is local.
Submitting code we have not pushed is what your own rule forbids, so the upload waits on the push
and on your word.

### What the input-only selection rule cost, measured

Six clips hold a rollout that kit-scores *higher* than the one that ships, turned down because it
sits further from the noisy input reference: e06 (−0.157), e05 (−0.076), e26 (−0.044), e13
(−0.022), e28 (−0.010), e00 (−0.001). That is **0.0103 of mean AUC left on the table**. It is the
price of a rule that reads no ground truth, which is the only kind of rule we may register, and I
am recording it rather than quietly reaching for the better number.

---

## (2) T3-VALID — a valid Tier 3 entry on every clip, ~$2

We have **no Tier 3 entry at all**, and Tier 3 is a third of the track. This buys one for about
the price of a coffee: the insurance budget, 409,600 steps per clip, one checkpoint, all 30 clips
on interp's reconstructed references.

    brev/second_wave.sh --groups T3-VALID --go

| | |
|---|---|
| group line | `T3-VALID\|0..29\|start_curriculum,orientation_reward,object_position\|V2D_OBJECT_POSITION_STD=0.08\|2\|t3_recon\|tier3\|409600:409600\|…` |
| boxes | 2 × A6000, 15 clips each, sequential |
| box-hours | 0.7 h/box by the model; the Tier 1 insurance run did 30 clips in 114 box-minutes |
| dollars | **$0.95 modelled, ~$2.00 against that precedent** |
| bar | the kit validator accepts a 30-episode `track_2_tier3` CSV, and every episode's provenance shows the reconstruction — not Tier 1 truth, not the Tier 2 trajectory. Any score beats the nothing we have. |

**Blocked on one thing, and it is not compute.** interp's reconstruction tree is not staged
(`$S/t3_recon` does not exist; the planner says `NOT STAGED YET` rather than quietly using
something else). The packer gap has closed: closer added tier3 to the wrapper's `TRACKS` at 22:57
UTC (`ws/submit` `f97b67c`, `track_2_tier3`, which needs `--frame-references`), so a Tier 3 entry
packs with the same command as the other two.

---

## (3) T3-FULL — full-length Tier 3 on as many clips as the credit buys

Nine clips, ordered by interp's reconstruction quality, at the precision recipe and the full
349,999,104 steps.

    brev/second_wave.sh --groups T3-FULL --go

| | |
|---|---|
| group line | `T3-FULL\|<9 clips>\|start_curriculum,orientation_reward,object_position\|V2D_OBJECT_POSITION_STD=0.08\|3\|t3_recon\|tier3\|\|…` |
| boxes | 3 × A6000, 3 clips each, sequential, 13.8 h/box (inside the 24 h authority) |
| box-hours | 41.4 |
| dollars | **$28.15 expected, $32.04 against the launcher's own more conservative model** |
| clip count | **9** — ten would need $35.44 and there is $34.90 |
| bar | per clip, the full-length rollout beats its T3-VALID rollout under the input-only adoption rule (distance to interp's reconstruction, which *is* the Tier 3 input). Entry-level: the 30-clip Tier 3 mean rises over T3-VALID. |

The clip list is a placeholder until interp ranks the reconstructions. Drop the ranked episode
numbers into `$S/second_wave/clips_T3-FULL.txt` and the planner takes them from there — no edit to
the table, and the plan prints which list it used.

---

## (4) Staged, waiting on an NVIDIA top-up only

Not launchable inside the credit. Each is staged or stageable and dry-runs clean.

| arm | what it changes | boxes | $ | bar |
|---|---|---|---|---|
| T1 | the clean tier_1 references, precision recipe, 1 seed, scored against tier1 | 8 | 99.01 (launcher: 105.67) | the 30-clip Tier 1 kit AUC beats CHORD's 0.51038 |
| W3A | precision recipe on tetool's bridge_v2, seed 42 | 8 | 99.01 | per clip, beats its wave-2 record under the adoption rule |
| W3B | the same, seed 43 — the two-seed mean | 8 | 99.01 | reported with W3A; adoption keeps the better per clip |
| W3R | the nine clips under 0.30: object weight 5, body weight 1, 700M steps | 5 | 61.88 | per clip, beats its own record |

T1 is fully staged: `$S/t1_precision`, 30 clips, 182 files, manifest verified, 14 of 30 resolving
to a supported scene. W3A/W3B need `$S/w3_bridge_v2` from tetool; W3R needs `$S/w3_regrasp` with
interp's re-grasped e04 and e25.

**W3R carries one change I could not write.** "Termination at 20 cm" is not in any file we hold —
not the objective config, not the curriculum, not the saved W&B config. Writing a patch against a
guessed key name would be a patch that reports success and changes nothing, which is the failure
mode every other patch here exists to prevent. It needs one `grep` inside the training image, so
it is the first thing to run on the first box that comes up, and until then W3R is the weight
change and the longer budget only.

---

## SMOKE TEST

Everything below runs on this machine with no GPU and no network, as
`tests/test_final_plan_smoke.py` (11 tests) plus the suites it leans on. `pytest -q` is green:
**414 tests**.

| what | evidence |
|---|---|
| **the patch stack, and the second clip of a box** | `test_the_full_patch_stack_applies_and_the_second_pass_is_an_accepted_no_op` — all four patches applied to a verbatim copy of upstream, then applied again; each no-op report is fed through the runner's real guard. This is the failure that killed e11 and e25 and would have killed 15 more clips. |
| **config compose** | `test_every_patched_config_still_parses_and_carries_its_arm` — the patched objective and curriculum parse as YAML and carry std 0.08, object weight 5, body weight 1, in the config *and* in every curriculum stage, which is the mapping `set_weights` actually applies. A real `--cfg job` needs the training image; it is the first command of the first cell. |
| **md5 manifest gate** | `test_the_tier_one_tree_matches_its_manifest` — 182 of 182 files verify against `$S/t1_precision.md5`. The other trees are not staged, and the planner refuses rather than launching without one. |
| **launcher dry-run** | `test_the_launcher_dry_run_names_every_box_the_map_and_the_arm` — prints `v2d-t1p-b01` … `b08`, the map, the tier and the arm flags, creates nothing. |
| **planner refusals** | `test_an_arm_whose_tree_is_not_staged_says_so_rather_than_planning_around_it`, `test_the_planner_refuses_to_go_over_the_cap_it_was_given`, `test_a_longer_budget_is_priced_as_the_gpu_hours_it_is`. |
| **packer end to end** | the two packs in (1): kit packer and kit validator, 30 episodes, 22,989 rows, `kit_validator: pass`, local score 0.4631 / 0.6667 / 0.4667 / 14.14, identical sha256 across the two tracks. |
| **adoption over three seeds** | `test_the_adoption_rule_chains_over_a_three_seed_tree` — three candidates on one clip chain through the incumbent, the seed closest to the *input* wins, and `verdict_reads_ground_truth` is False on every one. |

## What I will not do without a word from you

Create a box, launch an arm, upload a CSV, push a commit, send mail. The only thing moving is
b17 finishing e26, and it deletes itself when that is pulled and scored.
