"""Is a clip's score set by the clip, or by the attempt? Priced from the checkpoint curves.

This decides where money goes. The donor transfer adopted 3 of 9 clips, and that single number is
consistent with two different worlds:

* **per-clip** -- three clips were rescuable and the other six never will be, so extra attempts on
  a failed clip buy nothing and the budget should go somewhere else entirely;
* **per-attempt** -- any of them can come good and 1-in-3 is just the rate, so attempts are the
  thing to buy and $41 of breadth rescues most of the twelve broken Tier 1 clips.

All nine transfers used one donor (e08's fine-tuned actor), so that experiment holds the attempt
fixed and cannot separate the two. But the run itself is a repeated measurement nobody mined: each
clip has seven checkpoints, the z-drop rule already picks the best of them, and the spread between
them is attempt-to-attempt noise *within* one run. Six of the nine transfer results moved the score
by less than that spread, which means they are not evidence of an unresponsive clip at all.

What this reports, per clip and over the twelve broken ones:

* the within-run spread (std and range over the last five checkpoints), the noise floor any arm
  has to clear to be called a result;
* how much best-of-7 selection already beats the final checkpoint, which is what one attempt buys;
* a bootstrap of best-of-14 against best-of-7 -- the expected gain from a *second* attempt -- with
  the correlation caveat stated in the output rather than hidden.

Checkpoints along one run are correlated and often still trending, so resampling them as
independent draws overstates what a second attempt gives. The estimate is therefore reported
separately for the clips whose last five checkpoints are not monotone, where the trend is small and
the spread really is noise, and that figure is the one to quote.

Usage:  python tools/spend/attempt_value.py [--json out.json]
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEEP_DIVE = ROOT / "results" / "metrics_deep_dive.json"
TRANSFER = ROOT / "results" / "t1xfer_adoption.json"
RNG_SEED = 0
DRAWS = 20000

# The twelve Tier 1 clips that carry the tier's whole deficit (docs/WHAT_IT_TAKES.md).
BROKEN = ("e25", "e04", "e19", "e13", "e01", "e14", "e21", "e07", "e28", "e09", "e02", "e17")


def load_curves() -> dict[str, dict]:
    clips = json.loads(DEEP_DIVE.read_text())["clips"]
    return {c["tag"]: c for c in clips}


def best_of(values: list[float], n: int, rng) -> float:
    """Mean best-of-n when n draws are resampled with replacement from `values`."""
    total = 0.0
    for _ in range(DRAWS):
        total += max(rng.choice(values) for _ in range(n))
    return total / DRAWS


def analyse() -> dict:
    import random

    curves = load_curves()
    rng = random.Random(RNG_SEED)

    rows = []
    for tag, clip in sorted(curves.items()):
        curve = clip["checkpoint_curve"]
        aucs = curve["kit_add_auc"]
        last5 = aucs[-5:]
        monotone_tail = all(b >= a for a, b in itertools.pairwise(last5))
        rows.append(
            {
                "clip": tag,
                "shape": curve["shape"],
                "n_checkpoints": len(aucs),
                "final": aucs[-1],
                "best_of_7": max(aucs),
                "selection_gain": max(aucs) - aucs[-1],
                "tail_std": st.pstdev(last5),
                "tail_range": max(last5) - min(last5),
                "monotone_tail": monotone_tail,
                "broken": tag in BROKEN,
                "best_of_14_minus_7": best_of(aucs, 14, rng) - best_of(aucs, 7, rng),
            }
        )

    def agg(subset: list[dict], key: str) -> float:
        return st.fmean([r[key] for r in subset]) if subset else float("nan")

    broken = [r for r in rows if r["broken"]]
    noisy = [r for r in rows if not r["monotone_tail"]]
    broken_noisy = [r for r in broken if not r["monotone_tail"]]

    transfer = json.loads(TRANSFER.read_text())
    floor = agg(rows, "tail_range")
    inside = [c["clip"] for c in transfer["clips"] if abs(c["d_auc"]) < floor]

    return {
        "what": "Whether a second attempt on a failed clip is worth buying, priced from the "
        "checkpoint curves of the 30-clip run.",
        "noise_floor": {
            "mean_tail_std": agg(rows, "tail_std"),
            "mean_tail_range": agg(rows, "tail_range"),
            "max_tail_range": max(r["tail_range"] for r in rows),
            "what_it_is": "spread of kit AUC over each clip's last five checkpoints, within one "
            "run -- so it is attempt noise with the clip, the recipe and the reference all held "
            "fixed",
        },
        "the_transfer_reread": {
            "attempts": len(transfer["clips"]),
            "adopted": transfer["adopted"],
            "moved_less_than_the_noise_floor": inside,
            "reading": "Of nine transfer attempts, the ones that 'failed' mostly moved the score "
            "by less than the within-run noise floor. A result inside the noise is not evidence "
            "that the clip cannot be rescued; it is no evidence either way. That is the central "
            "correction to reading 3-of-9 as a per-clip property.",
        },
        "what_one_attempt_already_buys": {
            "mean_best_of_7_over_final": agg(rows, "selection_gain"),
            "mean_over_the_twelve_broken": agg(broken, "selection_gain"),
            "note": "The z-drop rule already takes the best of seven checkpoints, so every "
            "'attempt' is a best-of-7 and this is the part we are not paying extra for.",
        },
        "what_a_second_attempt_adds": {
            "all_30_clips": agg(rows, "best_of_14_minus_7"),
            "non_monotone_tail_only": agg(noisy, "best_of_14_minus_7"),
            "broken_and_non_monotone": agg(broken_noisy, "best_of_14_minus_7"),
            "n_non_monotone": len(noisy),
            "n_broken_non_monotone": len(broken_noisy),
            "caveat": "Checkpoints along one run are correlated and often still trending, so "
            "resampling them as independent draws overstates a second attempt. The "
            "non-monotone-tail figure is the one to quote; the all-30 figure is an upper bound.",
        },
        "clips": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = analyse()

    nf = out["noise_floor"]
    print(
        f"noise floor over the last five checkpoints: std {nf['mean_tail_std']:.4f}, "
        f"range {nf['mean_tail_range']:.4f} mean, {nf['max_tail_range']:.4f} max\n"
    )
    print(
        f"{'clip':5} {'shape':11} {'final':>7} {'best7':>7} "
        f"{'sel+':>7} {'tailsd':>7} {'+2nd':>7} broken"
    )
    for r in out["clips"]:
        print(
            f"{r['clip']:5} {r['shape']:11} {r['final']:7.4f} {r['best_of_7']:7.4f} "
            f"{r['selection_gain']:+7.4f} {r['tail_std']:7.4f} "
            f"{r['best_of_14_minus_7']:+7.4f} {'yes' if r['broken'] else ''}"
        )
    sa = out["what_a_second_attempt_adds"]
    print(
        f"\nsecond attempt adds: {sa['all_30_clips']:+.4f} over all 30 (upper bound), "
        f"{sa['non_monotone_tail_only']:+.4f} over the {sa['n_non_monotone']} non-monotone, "
        f"{sa['broken_and_non_monotone']:+.4f} over the {sa['n_broken_non_monotone']} broken ones"
    )
    inside = out["the_transfer_reread"]["moved_less_than_the_noise_floor"]
    print(f"\ntransfer results inside the noise floor: {inside}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))
        print(f"-> {args.json}")


if __name__ == "__main__":
    main()
