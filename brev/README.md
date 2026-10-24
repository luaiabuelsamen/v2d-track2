# Brev: the Day-4 two-clip FlashCHORD gate

Everything here is **off** until Luai redeems the credit code and runs the launcher with
`--go`. No script in this directory creates an account, redeems a code, submits, posts or
pushes. `gate_launch.sh` without `--go` only prints the live price table and the plan.

## Why a gate at all

The whole 11-day plan rests on one unmeasured number: **how long one FlashCHORD G1 clip
takes**. CHORD's paper says "∼2 hours per sequence on a single L40S", but that is CHORD's own
recipe, not V2D's `g1_recon_body_flash_sac`, whose published terminal budget is
**349,999,104 environment steps at 2,048 worlds**. A user in repo issue #148 measured "about
2 days for one task" for the *PPO* lane on an RTX 6000 Ada and the organizers said that
matched their own numbers. If FlashCHORD is closer to the PPO figure than to CHORD's, the
entry is dead and we want to know on day 4, not day 9.

So the gate runs in two phases, cheap first:

| Phase | What it does | Steps | Wall clock | Cost on an A6000 |
| --- | --- | --- | --- | --- |
| `probe` | Throughput and memory only, on both clips. Extrapolates hours/clip. | 10 M | ~20–40 min | **~$0.40** |
| `full` | One complete clip, then evaluate → pack → score locally. | 350 M | 3–8 h | **~$2–5** |

`probe` is what answers the go/no-go. `full` proves the whole pipeline end to end and gives
the first real per-clip score.

## GPU choice and prices

FlashCHORD requires **driver ≥ 580** and a **48 GiB GPU with ≥ 46,000 MiB free**. A Brev
instance was observed running driver 580.126.09 / CUDA 13.0, which matches the repo's stated
requirement, and `jax[cuda13]` in the FlashCHORD pin.

Brev resells capacity from several providers, so prices move. These are from a published
`brev search --max-boot-time 3 --sort price` snapshot (Oct 2026), and the launcher re-reads
live prices every run:

| Brev type | GPU | VRAM | $/h | Fits? |
| --- | --- | ---: | ---: | --- |
| `hyperstack_A6000` | A6000 | 48 GB | **0.60** | **yes — cheapest that fits** |
| `massedcompute_A6000_plus` | A6000 | 48 GB | 0.68 | yes |
| `massedcompute_L40S` | L40S | 48 GB | 1.06 | yes (but see note) |
| `massedcompute_RTX6000Ada` | RTX 6000 Ada | 48 GB | 1.16 | yes |
| `excesssupply_RTX5090` | RTX 5090 | 32 GB | 0.78 | **no — under 46,000 MiB** |
| `excesssupply_RTX4090` | RTX 4090 | 24 GB | 0.72 | **no** |

**Note on the L40S:** NVIDIA's own Brev GPU-types page lists the L40S as **44 GB**, which is
*below* the 46,000 MiB floor, while listing the L40 and RTX 6000 Ada at 48 GB. The
`brev search` snapshot above reports 48 GB for the same card. Until that is resolved on a
live instance, prefer the **A6000**: it is both the cheapest and unambiguously 48 GB. The
launcher refuses any type whose reported VRAM is under 46 GB.

## Cost model, with the credit as the budget

$250 of credit at $0.60/h is **about 416 A6000-hours**. Taking CHORD's 2 h/clip on an L40S
and assuming the A6000 is 1.5–2× slower gives **3–4 h/clip, so $1.80–2.40 per clip-policy**:

| Item | Clips | A6000-hours | $ |
| --- | ---: | ---: | ---: |
| Gate, both phases | 2 | 4–9 | **4–6** |
| One full 30-clip pass (Tier 2) | 30 | 90–120 | **54–72** |
| A second pass after a fix | 30 | 90–120 | **54–72** |
| Dev runs on a 5-clip subset, ~4 variants | 20 | 60–80 | **36–48** |
| Eval-only re-runs, packing, slack | — | 10–20 | **6–12** |
| **Total** | | **254–349** | **~$154–210** |

That fits inside $250 with roughly 20–35 % headroom, and the **binding constraint is
wall-clock, not money**: one 30-clip pass is 90–120 GPU-hours, which is 4–5 days on one
instance. Run **3–4 instances in parallel** to land a pass inside a day. Whether a new Brev
org can hold 4 concurrent instances is unverified — check it on day 1, because the whole
schedule depends on it.

Storage is billed while an instance exists, even stopped ($0.0001314/GB/h ≈ $0.10 per 100 GB
per day), and some Brev types cannot be stopped at all, only deleted. The launcher therefore
**deletes** the instance on exit rather than stopping it, and runs a watchdog that deletes it
after `--max-hours` no matter what.

## The two clips

Chosen from the measured per-episode table in `results/tilt_yaw_split.json`, both resting on
the **ground plane** so the gate does not also depend on the support-surface frame problem:

| Role | Episode | Object | Frames | Size | Why |
| --- | --- | --- | ---: | ---: | --- |
| easy | `e24` | watering_can | 748 | 28 cm | small object, floor contact, mid-length |
| hard | `e05` | big_black_box | 1003 | 86 cm | longest episode, large object, floor contact |

## Pass criteria

The gate passes only if all three hold. Write the answers into `results/gate/`.

1. **Cost.** Measured hours/clip extrapolated to 30 clips is **≤ 120 A6000-hours (≤ $72)**.
2. **Plumbing.** The packed 2-clip CSV passes `tools/preflight.py`, and
   `scripts/score_track2.py` scores it without error.
3. **Quality floor.** Per-clip local AUC **> 0.20**. The CHORD baseline scores 0.510 AUC
   averaged over all 30 Tier 1 clips, so a single trained clip well under 0.20 means the
   recipe is not tracking at all, not that the clip is hard.

If 1 fails, drop to Tier 1 only and a single pass. If 2 fails, fix the packer before
spending anything else. If 3 fails, stop and re-read the recipe.

## Run it

```bash
# After `brev login`, from the repo root. Prints prices and the plan, changes nothing:
brev/gate_launch.sh

# Throughput probe (~$0.40):
brev/gate_launch.sh --phase probe --go

# Full single-clip gate, after the probe looks sane:
brev/gate_launch.sh --phase full --max-hours 10 --go
```

Teardown is automatic. If the launcher dies between create and delete it prints the exact
`brev delete` command; run it, then confirm with `brev ls` that nothing is RUNNING.

## The gate trains on a cleaned reference

`gate_launch.sh --phase full` refuses a motion parquet that was retargeted from the raw Tier 2
reference. `retarget_all.py` writes a `provenance.json` at its motion root recording the object
source; the launcher walks up from the parquet to find it and checks `cleaned_reference`.

The reason is measured, not assumed. Scored with the kit's own metric code
(`results/cleanup_ladder.json`), a perfect tracker of each reference would get:

| reference | AUC | SP-SR | MP-SR | MPPE cm |
|---|---|---|---|---|
| Tier 2 as released | 0.4247 | 0.9667 | 1.0000 | 2.3605 |
| **support-contact cleaned** | **0.4660** | **1.0000** | 1.0000 | **2.3053** |
| tilt oracle (not usable) | 0.6541 | 1.0000 | 1.0000 | 2.0672 |

So a full-phase run against the raw reference spends $2–5 measuring a reference we already know
is worse on three of four metrics. Produce the cleaned one first:

    python scripts/clean_support_contact.py --dataset-root DATA --npz-out NPZ_CLEAN \
        -o results/clean_support_contact.json
    python scripts/retarget_all.py --dataset-root DATA --repo REPO --soma-root SOMA \
        --seq-root SEQ --out-root OUT --log-dir LOGS --validation-dir VAL \
        --tiers tier2 --object-npz NPZ_CLEAN -o results/arm_support_cleaned.json

then point `--g1-parquet` at a clip under `OUT`. `--allow-raw-reference` overrides the check for
a deliberate baseline comparison; nothing overrides it silently.

**The oracle rows are not a target.** `tilt_oracle` is built from the clean Tier 1 trajectory and
cannot be used by a submission. It is the upper bound the deployable cleaner reaches for, and it
currently captures 18 % of that bound's AUC gain.
