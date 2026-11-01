"""The noise-floor re-reading of the donor transfer, pinned.

The whole revision in `docs/CHEAPEST_RANKS.md` rests on one comparison -- a measured per-clip
difference against the within-run spread -- so the two ways of getting it wrong are pinned here:
calling a difference a result when it is inside the noise, and resampling checkpoints as if more
draws of the same arm were worth what a different arm is worth.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "attempt_value", ROOT / "tools" / "spend" / "attempt_value.py"
)
attempt_value = importlib.util.module_from_spec(_spec)
sys.modules["attempt_value"] = attempt_value
_spec.loader.exec_module(attempt_value)


def test_best_of_n_rises_with_n_but_is_bounded_by_the_observed_max() -> None:
    """More draws of the same distribution can never beat that distribution's best."""
    import random

    rng = random.Random(0)
    values = [0.10, 0.20, 0.30, 0.40]
    seven = attempt_value.best_of(values, 7, rng)
    fourteen = attempt_value.best_of(values, 14, rng)
    assert seven < fourteen <= max(values)


def test_a_single_valued_run_gains_nothing_from_more_draws() -> None:
    """A clip whose checkpoints all agree has no noise to harvest, so a retry is worth zero."""
    import random

    rng = random.Random(0)
    flat = [0.25] * 7
    assert attempt_value.best_of(flat, 14, rng) == attempt_value.best_of(flat, 7, rng) == 0.25


def test_the_twelve_broken_clips_are_the_ones_the_doc_names() -> None:
    assert len(attempt_value.BROKEN) == 12
    assert set(attempt_value.BROKEN) == {
        "e25",
        "e04",
        "e19",
        "e13",
        "e01",
        "e14",
        "e21",
        "e07",
        "e28",
        "e09",
        "e02",
        "e17",
    }


def test_transfer_results_inside_the_noise_floor_include_an_adopted_clip() -> None:
    """e19 was adopted on +0.0428 AUC, which is inside the noise. That is the finding.

    If a future change to the noise floor stopped flagging e19, the claim in the doc -- that an
    adoption can sit inside the error bar -- would silently stop being true.
    """
    out = attempt_value.analyse()
    inside = out["the_transfer_reread"]["moved_less_than_the_noise_floor"]
    assert "e19" in inside, "e19 adopted on a difference smaller than the within-run spread"
    assert "e05" not in inside, "e05's +0.385 is a real effect, well outside the noise"
    assert len(inside) >= 5


def test_a_second_attempt_at_the_same_arm_is_worth_less_than_a_rank_point() -> None:
    """The number that killed the $41 repeat-the-arm plan."""
    out = attempt_value.analyse()
    gain = out["what_a_second_attempt_adds"]["all_30_clips"]
    assert 0.0 < gain < 0.02, gain


def test_the_recorded_json_matches_a_fresh_run() -> None:
    recorded = json.loads((ROOT / "results" / "attempt_value.json").read_text())
    fresh = attempt_value.analyse()
    assert recorded["noise_floor"] == fresh["noise_floor"]
    assert (
        recorded["the_transfer_reread"]["moved_less_than_the_noise_floor"]
        == (fresh["the_transfer_reread"]["moved_less_than_the_noise_floor"])
    )
