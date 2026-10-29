# Tier 1 generation-2 chain: pre-registered (2026-10-29 11:10 UTC)

## The observation this is built on

The generation-1 fine-tune splits the thirty clips cleanly. Among the fifteen that ship their final
checkpoint, five of the eight scored so far adopt, and the gains are large:

| clip | shipped | gen 1 | Δ |
|---|---|---|---|
| e00 | 0.51775 | 0.86544 | **+0.3477** |
| e22 | 0.66864 | 0.86031 | **+0.1917** |
| e12 | 0.79036 | 0.88393 | **+0.0936** |
| e16 | 0.56394 | 0.64688 | **+0.0829** (and gains an MP-SR pass) |
| e24 | 0.89422 | 0.95672 | **+0.0625** |

Those five clips are a direct measurement that 50M steps of adaptation to the clean Tier 1
reference is **not** saturated on them: every one improved AUC *and* MPPE, and none traded a
success gate away. A clip that behaves like that is in the regime where the actor is still moving
toward its reference, not oscillating around it.

The question this arm asks is the obvious next one: **does a second 50M buy more?**

## What runs

For each clip that gen 1 adopted, warm-start from **gen 1's own checkpoint** rather than the
shipped one, and train 349,999,104 → 399,998,976. Everything else is the gen-1 recipe: clean Tier 1
reference, `start_curriculum + orientation_reward(std 0.3) + object_position(std 0.08)`, seed 42,
cold critic, actor LR ×0.1 for 2,000 updates, support per clip, `world_count=1` evaluation.

This needed one change, `brev/run_warm_start.sh`'s new `WS_START`: the patch defaults
`warm_start_environment_steps` to 299,999,232, which is correct off a 350M checkpoint and wrong off
a fine-tuned one — the step counter and the curriculum would rewind 50M and redo training the actor
has already had. `WS_START=349999104` tells it where the actor really is, and the runner refuses a
chain whose origin is not below its target so the mistake cannot be made silently.

## The bar, registered before launch

Per clip, the **same no-regression rule** decides: gen 2 ships only if its kit AUC improves on the
rollout that currently ships *and* spider_sr does not fall, maniptrans_sr does not fall, and
mppe_cm does not rise. Ties and regressions keep gen 1.

The arm counts as worth repeating only if **at least two clips adopt**. One adoption out of five is
noise at this sample size and would mean the gen-1 fine-tune is where the available gain was.

**Stated now:** the honest prior is that this returns less than gen 1 did. Gen 1 closed a
distribution shift — a reference the actor had never seen — and that is a one-time debt. Gen 2 has
no shift to close, so it is ordinary extra training, and the clips most likely to gain are the ones
furthest from their own ceiling (e16 at 0.647 and e22 at 0.860 have more room than e24 at 0.957).
If it returns nothing, that is a clean negative result about where the warm start's value comes
from, and it costs about $3.

## Cost

| | clips | box-h | at $0.60/h |
|---|---|---|---|
| one box, sequential | 5–8 | ~4.5 | **$2.70** |

Five clips are adopted now; the seven still training can add to that list, and any that do are
included if the box has deadline left. Budget: $15.54 committed of the $30.38 credit, the two live
boxes will take about $2.50 more, so this fits with roughly $9 unspent.

## Why this and not the alternatives

* **The fifteen non-final-checkpoint clips.** Measured and rejected: every one lost (e20 −0.235,
  e11 −0.079, e02 −0.056, e01 −0.011). z-drop having picked an earlier checkpoint *is* a
  measurement that late-stage training hurts that clip, and the fine-tune re-runs exactly that
  stage. Warm-starting them at their own selected step instead would redo a segment the original
  run already produced and selection already rejected.
* **A second seed.** Adoption already keeps the better of what we hold, so a second seed's value is
  bounded by the spread between seeds, which the four-clip checkpoint study put near the noise
  floor. A second generation moves the actor; a second seed re-rolls the dice.
* **Tier 2.** Its input may not be the clean trajectory, so none of this transfers. Tier 2 is also
  the one tier already comfortably above its baseline (0.46523 against CHORD's 0.374), and Tier 1
  is the one below (0.49118 against 0.51038). The marginal dollar belongs on Tier 1.

## Result, 2026-10-29 13:10 UTC — the chain fails its bar and was stopped early

| clip | shipped | gen 1 (350M) | gen 2 (400M) | gen2 - gen1 |
|---|---|---|---|---|
| e00 | 0.5178 | 0.8654 | 0.8078 | **-0.0576** |
| e12 | 0.7904 | 0.8839 | 0.8013 | **-0.0826** |
| e16 | 0.5639 | 0.6469 | running | |
| e22 | 0.6686 | 0.8603 | not run | |
| e24 | 0.8942 | 0.9567 | not run | |

The registered bar was "at least two clips adopt". Both scored clips lost, and lost by more than
most gen-1 clips gained. **A further 50M steps on an already-fine-tuned actor is not free; it is
negative.** The fine-tune's gain comes from re-running the last curriculum stage against a *new*
reference, and once that has happened the clip is at its peak — more of the same stage walks it off.

e22 (gen-1 0.8603) and e24 (0.9567) are the two most saturated clips in the queue, which is the
same shape that just lost twice, so they were **deliberately not run** and the box was deleted after
e16 — the one clip of the three with real headroom left (gen 1 gained only +0.083 there). That saved
about 2 box-hours, ~$1.35, on a question the first two clips had already answered.

Adoption is unaffected either way: a gen-2 clip only ships if it beats its gen-1 rollout, and
neither did.

## What the chain result rules out elsewhere, 2026-10-29 13:20 UTC

The chain is the control for "can we buy score with more steps on the reference we already used",
and the answer is no. That closes off the obvious Tier 2 move without spending anything:

* **Tier 2 cannot be improved by re-training on bridge_v1.** The wave-2 policies were trained
  0 -> 350M on exactly that reference, so a fine-tune onto it is more of the same final stage — the
  thing that cost e00 0.058 and e12 0.083. A Tier 2 gain would need a *different* legal reference.
* **bridge_v2 does not exist.** `~/projects/_ops/briefs/clean_refs_path.txt` still points at
  `v2d-clean-refs/bridge_v1`; the v2 tree tetool was staging never landed. So there is no new Tier 2
  reference to fine-tune onto, and nothing to buy there with the remaining credit.

Tier 1 and Tier 3 both *did* gain from the fine-tune because in both cases the reference changed:
Tier 1 to the released clean trajectory, Tier 3 to our own video reconstruction. The mechanism is
re-fitting to a new reference, not extra training. That is why the last of the credit goes to
Tier 3 (0.25059, the weakest board, 21 clips never fine-tuned) and not to Tier 2.

## Final: 0 of 3, and my reason for keeping e16 was wrong

| clip | gen 1 (350M) | gen 2 (400M) | delta |
|---|---|---|---|
| e00 | 0.8654 | 0.8078 | -0.0576 |
| e12 | 0.8839 | 0.8013 | -0.0826 |
| e16 | 0.6469 | **0.2963** | **-0.3506** |

e16 also loses both success gates (SP-SR 1.0 -> 0, MP-SR 1.0 -> 0) and its MPPE goes 3.0 -> 61.4 cm:
the policy stops holding the object altogether.

**I kept e16 for the wrong reason.** The argument was that gen 1 had gained only +0.083 there, so
e16 had "the most headroom" of the three remaining clips, while e22 (0.8603) and e24 (0.9567) were
saturated. e16 then lost the most by a factor of four. Headroom was not the variable; the arm
damages a clip wherever it sits. Dropping e22 and e24 was the right call and the reasoning I
published for it does not survive its own test case — had I run them, the likely outcome was two
more losses, which is the same conclusion by a worse route.

The arm fails its registered two-adoption bar 0 of 3, and `results/t1_chain_adoption.json` is what
keeps those rollouts out of the entry — the record exists because the packer had already shipped
two of them once on directory order alone.

**Cost of the overrun:** the broken waiter let the box start e22 at 13:15 and I caught it at 13:35
by reading the box directly, so about 20 minutes, ~$0.23, on a clip whose result was not wanted.
