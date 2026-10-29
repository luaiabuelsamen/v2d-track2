# What it would actually take to reach the top, costed — 2026-10-29

Written after pulling the twelve public boards (`$S/leaderboard_positions.json`). Everything here
is measured, and the uncomfortable part is stated first.

## Where we are

| board | rank | us | 1st | 2nd | gap |
|---|---|---|---|---|---|
| tier1-auc | 12/23 | 0.52687 | 0.89354 | 0.83047 | 0.367 |
| tier1-sp-sr | 12/22 | 0.66666 | 0.96666 | 0.90000 | 0.300 |
| tier1-mp-sr | 11/22 | 0.50000 | 0.93333 | 0.86666 | 0.433 |
| tier1-mppe | 11/22 | 13.90 cm | 2.78 | 3.68 | 11.1 cm |
| tier2-auc | 5/18 | 0.46522 | 0.56768 | 0.52267 | 0.102 |
| tier2-sp-sr | 5/18 | 0.66666 | 0.80000 | 0.73333 | 0.133 |
| tier2-mp-sr | 5/18 | 0.46666 | 0.80000 | 0.60000 | 0.333 |
| tier2-mppe | 4/18 | 14.13 cm | 9.28 | 10.84 | 4.8 cm |
| tier3-auc | **3/10** | 0.25059 | 0.33311 | 0.32937 | 0.083 |
| tier3-sp-sr | 7/10 | 0.16666 | 0.40000 | 0.40000 | 0.233 |
| tier3-mp-sr | 6/10 | 0.03333 | 0.10000 | 0.10000 | 0.067 |
| tier3-mppe | 6/10 | 54.44 cm | 31.34 | 35.01 | 23.1 cm |

## The finding that matters most: money alone does not get us there

Our 30 Tier 1 clips split cleanly. **19 clips at or above 0.35 average 0.7228. Eleven below 0.35
sum to 2.074** (e28 e19 e25 e04 e13 e01 e14 e21 e07 e05 e09). Suppose every weak clip were lifted
to the level of our *good* clips:

| scenario | entry mean AUC | rank it would buy |
|---|---|---|
| today | 0.52687 | 12th |
| all 11 weak clips reach 0.60 | 0.67776 | ~8th |
| all 11 reach 0.70 | 0.71442 | ~6th |
| all 11 reach 0.80 — better than our current average | 0.75109 | ~5th |
| 1st place | 0.89354 | — |

**The leader's 30-clip mean (0.894) is higher than the average of our nineteen best clips (0.723).**
So first place is not a matter of fixing the failures; it needs a per-clip tracker that is better
than ours almost everywhere. Our individual ceiling is there — e08 scores 0.9236 at 1.43 cm, and
the leader's whole-entry MPPE is 2.78 cm — but we reach it on about six clips, not thirty. No
amount of credit spent on the current recipe closes 0.367.

## The three things worth buying, cheapest first

**1. Tier 3 MPPE, ~$2.30 — launching now.** Five clips with real headroom (e03 e25 e01 e29 e16).
Headroom = pass-1 MPPE minus the reconstruction's own error against truth; see
`docs/T3_MPPE_PUSH.md` for why raw MPPE is the wrong selector and nearly cost a box. Takes Tier 3
MPPE from 54.44 to about 42-46 cm, 6th → 4th. Total headroom on every untouched clip is only
~363 cm (−12 cm on the mean), so **Tier 3 MPPE 1st at 31.3 cm is unreachable by fine-tuning at any
price** — the binding constraint there is reconstruction quality: ten clips have reconstruction
floors of 90-271 cm, where our policy already beats its own input.

**2. Tier 1 weak clips, ~$36 (11 clips x 4.4 box-h x $0.68) — the best use of a credit top-up.**
These eleven never adopted a warm start, and the measured rule says they never will: 0 of 4 clips
below 0.35 ever adopted while 11 of 19 above it did. They need full-length training *on the clean
Tier 1 reference*, not a 50M-step re-fit of a policy that was trained on noise. Expected 0.60-0.80
per clip on the evidence that clean-reference training is what made every strong clip strong;
entry mean 0.68-0.75, i.e. **12th → 5th-8th**. This is the single largest rank move available for
money, and it needs about seven times the credit that is left.

**3. Tier 2's real gap is a method, not a budget.** Tier 2 scores a policy driven by the *noisy*
reference against *clean* truth. Our policies are trained to track the reference they are given, so
they faithfully reproduce the noise. The structural fix is a **dual-reference objective: noisy
reference as the policy's observation, clean released trajectory as the reward target** — a
denoising tracker rather than a tracking denoiser. The clean trajectory is released data and is
never fed to the policy at inference, so this is a training-time supervision change, not an input
violation. It needs a patch to `recon_body` (the objective currently reads one reference for both
roles) plus ~$12 for a 5-clip warm-start pilot at 50M steps. **Not started: the patch is a day of
work and there is $5 left**, but it is the only idea on the table that could close a 0.10 AUC gap
rather than a 0.02 one.

## What not to spend on

- **More Tier 1 warm starts.** Closed on measurement: every clip above the 0.35 adoption line has
  been run, and the seven never run are all below 0.30.
- **A second generation of fine-tuning.** 0 of 3 adopted; e16 went 0.6469 → 0.2963 and lost both
  gates. The gain comes from re-fitting to a *new* reference, never from more steps.
- **Tier 2 re-uploads.** The board already holds our best Tier 2 file; the second-wave candidate
  scored 0.46313, worse.
- **Negative-headroom Tier 3 clips** (e04 −47 cm, e19 −183, e14 −114, e27 −141). The policy there
  already beats the reconstruction, and the input-only rule would adopt a regression.

## Spend to date

~$111 of NVIDIA Brev credit across the first wave, the second wave, Tier 3 reconstruction, the
Tier 1 warm-start programme and the strong-four arm. **~$5 left**, and the card is off. The Nov 4
2026 5:00 p.m. EST freeze is the deadline; all three tiers are already on the board, so a
top-up buys rank, not eligibility.
