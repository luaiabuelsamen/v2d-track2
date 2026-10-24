# The margin B3 is judged against

**Written and committed before B3 launched.** B3 compares three training arms on one clip, so the
question that decides whether any difference it finds is real is: how much does the same
configuration move when only the seed changes?

## Measured, from two B2 seeds

Same arm, same reference (clean Tier 1 e24), same 10,000,384 steps. Only `reset.seed` differs.

| metric | seed 42 | seed 43 | **seed-to-seed spread** |
|---|---|---|---|
| evaluator AUC | 0.2161 | 0.2061 | **0.0100** |
| evaluator MPPE cm | 14.233 | 16.193 | **1.959** |
| object position error m | 0.1365 | 0.1566 | 0.0201 |
| object orientation error ° | 73.83 | 74.07 | 0.233 |

## What follows

**MPPE cannot judge B3 at this budget.** Its seed spread is **1.96 cm**, while the entire
difference between B1 (409,600 steps) and B2 (10,000,384 steps) — a 24× change in training
budget — was **0.015 cm**. The noise is **130× the signal**. Any B3 ranking by MPPE at 10M steps
would be reading seeds, not arms.

**AUC can, within ±0.010.** Its spread is 0.0100 against a B1→B2 difference of 0.0744, so roughly
7× signal-to-noise. With two seeds per arm the standard error of an arm's mean is about
0.0100/√2 ≈ 0.007, so a difference between two arms needs to clear roughly **0.014 AUC** before
it means anything.

**SP-SR and MP-SR carry no information at all here.** Both are 0.0 for B1 and for both B2 seeds,
because the policy never gets inside 10 cm / 0.5 rad at 0.12–2.9 % of the 350M-step recipe. They
cannot distinguish arms and should not be reported as if they could.

## Decision rule for B3, fixed in advance

1. Rank arms on **AUC**, averaged over the two seeds.
2. Call a difference real only if it exceeds **0.014 AUC**. Below that, report the arms as
   indistinguishable at this budget and say what budget would be needed.
3. Report MPPE, SP-SR and MP-SR as measured, each annotated with the fact that its resolution at
   this budget is worse than any plausible arm difference. Do not rank on them.
4. A single-seed result is not a result. If a run fails and leaves an arm with one seed, that arm
   is reported as incomplete rather than compared.

The judging metric is Luai's call and he set MPPE and AUC. This document does not overrule that;
it records what each metric can and cannot resolve at 10M steps, so that whichever rule he
chooses is applied with its error bars visible.

## Caveat

Two seeds give a spread, not a standard deviation. ±0.010 AUC is a one-sample estimate of the
seed noise and could itself be off by a factor of two. It is the best available number and it is
enough to rule MPPE out, which is the decision it was needed for.
