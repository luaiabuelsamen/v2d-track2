# Tier 1, the four strong clips the prune skipped — pre-registered before launch

## What the prune got wrong, and what it got right

When the Tier 1 fine-tune queues were pruned, the rule was "run only clips that ship their final
checkpoint", on the reasoning that a clip whose z-drop pick was an *earlier* checkpoint had already
measured that late-stage training hurt it, and the fine-tune re-runs exactly 299,999,232 ->
349,999,104. Eleven clips were dropped on that rule, saving ~$4.80.

With all 18 scored, the real split is not final-checkpoint, it is **how good the clip already is**:

| shipped AUC | n | adopted | mean delta |
|---|---|---|---|
| < 0.35 | 3 | **0** | +0.0019 |
| >= 0.35 | 15 | **8** | +0.0441 |

Every one of the eight adoptions came from a clip already at 0.4758 or better. The fine-tune does
not rescue a weak clip; it sharpens a clip that already tracks. (e18 and e29 show the converse:
shipping a final checkpoint is what makes a gain *possible*, not what makes one happen — both ship
final checkpoints and both lost.)

Four of the eleven pruned clips are strong: **e03 0.7806, e08 0.6654, e15 0.6653, e27 0.7795.**
Those were dropped for the wrong reason, and they are the cheapest AUC left on Tier 1.

## The run

Four clips, two boxes, two clips each, gen-1 recipe unchanged (START + ORI 0.3 + object_position
std 0.08, seed 42, warm start from the shipped actor at 299,999,232 -> 349,999,104, scene.support
per clip). Clean Tier 1 references, same staging as gen 1. ~1.1 h per clip, so ~2.4 box-hours:
**about $1.70 of compute, ~$3 with bring-up.**

## The bar, registered now

Adoption is the same no-regression rule as the rest of the Tier 1 fine-tune
(`tools/t1/adopt.py`): a clip replaces its shipped rollout only if kit AUC rises **and** spider_sr
does not fall **and** maniptrans_sr does not fall **and** mppe_cm does not rise. Reading the kit
score is legal on Tier 1 and only on Tier 1, because `config/tracks.json` names the released Tier 1
trajectory — the permitted input — as the configured scoring reference.

There is therefore **no downside case**: a clip that regresses contributes nothing and ships its
old rollout. The question is only whether the $3 buys anything.

**Registered expectation.** At the measured strong-clip rate (8 of 15 adopt, mean adopted gain
+0.169), four clips project to about +0.36 of summed AUC, i.e. **+0.012 on the 30-clip entry mean**:
0.50838 -> about **0.520**, against CHORD's 0.51038. The arm counts as having worked if at least
one clip adopts; it counts as having been worth launching if the entry mean crosses 0.51038.

**The seven weak clips are deliberately not run**: e05 e07 e13 e14 e19 e25 e28, all shipping under
0.27, where the measured adoption rate is 0 of 3 and the mean delta is +0.002. Paying ~$5 for them
would be buying a number the data says is zero.

## e09 lands and sharpens the rule, 2026-10-29 13:15 UTC

e09 was the last gen-1 clip: **shipped 0.30029 -> fine-tuned 0.22399, -0.07629, rejected.** It is
the fourth clip below 0.35 to be fine-tuned and the fourth to fail, so the split now reads **0 of 4
below 0.35, 8 of 15 at or above it.** Nineteen clips scored; the eight adoptions are unchanged
(e00 e06 e12 e16 e22 e23 e24 e26) and the entry stands at 0.5103 by per-clip tally, 0.50838 packed.

This is the clearest version of the finding: the warm-start fine-tune is a *sharpening* tool. It
takes a policy that already tracks the object and re-fits it to a new reference. A policy that
drops the object has nothing to sharpen, and 50M more steps of the final curriculum stage will not
teach it to hold on. Every clip left in the Tier 1 queue is therefore chosen on shipped AUC alone.

## Bring-up failure, 2026-10-29 13:20-13:30 UTC — the instance type was delisted mid-flight

Both boxes came back **FAILURE / CREATE_FAILED** after 28 minutes in DEPLOYING. The cause was not
the ~1/3 random create-failure rate: **`hyperstack_A6000` disappeared from the Brev catalog between
the 12:52 launch and the 13:27 retry.** Preflight had passed it at 12:52 ("is in the catalog") and
refused it at 13:27, listing the 48 GB cards that remain:

    massedcompute_A6000        $0.68
    massedcompute_A6000_base   $0.68
    massedcompute_A6000_plus   $0.68
    massedcompute_L40S         $1.06

That also explains why *both* boxes failed simultaneously, which random provisioning failure would
not. Relaunched as `v2d-t1s4-a2` / `v2d-t1s4-b2` on `massedcompute_A6000` at $0.68/h — the rate the
cost projection already assumed, so the ~$1.70 estimate is unchanged.

**The catalog churns in both directions.** `massedcompute_A6000_plus` vanished earlier today and
`hyperstack_A6000` was then the cheapest 48 GB card; now the reverse. Any launcher that pins a type
will eventually create nothing, so the type belongs in preflight (where it is) and overridable at
the command line (`ITYPE=`, added).

Three plumbing fixes went with the relaunch, each one a thing that would have lost the run quietly:
the box name is overridable so a replacement keeps its letter's staged tree and results directory;
the puller maps `v2d-t1s4-a*` by pattern, so a replacement's rollouts still land where the packer
looks; and the bring-up watcher reads the STATUS column for `FAILURE` rather than waiting out a
20-minute SSH poll against an instance that will never boot.

## The runner raced its own image build, 13:34-13:45 UTC

Both relaunched boxes wrote `FATAL: no container; run gate_on_instance.sh build first` into
`run.log` within four minutes of bring-up, while `gate_on_instance.sh build` was still running and
`docker images` was empty. So the eval stage ran before the image existed, failed, and exited —
and nothing would have started training once the build finished. GPU 0%, box billing.

`gate_on_instance.sh` starts the container with `docker/run.sh start latest 0 &` and then waits
only **100 s** for it to appear. A cold box has no image, so that `start` triggers a full build of
several minutes, the 100 s wait expires, and the build phase returns while the image is still
being made. The runner's `|| FAILED` guard does not catch it because the build step itself does not
report failure.

**Recovery, no re-provisioning:** a waiter per box polls for a real image and then starts
`run_warm_start.sh` with the launcher's exact env. It will not act while the launcher's own
`t1_runner.sh` is alive — that process starts the runner itself once its build returns, and a
second runner on the same plan would put two trainings on one GPU and two claims on one clip. The
first version of the waiter checked only for a running training, which left a race window of one
poll interval; it now stands down for the owner and acts only when nothing else can.

This is a cold-box defect, so **the Tier 3 launch will hit it too** and the same waiter applies.
