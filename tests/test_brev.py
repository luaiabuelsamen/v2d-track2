"""Tests for the GPU picker.

The regression these exist for: the previous awk picker guessed which column was which and got
it wrong, reading TARGET_DISK as the VRAM and CAPABILITY as the price. On the live table that
ranked a 16 GB V100 at $2.81/h ahead of a 48 GB A6000 and proposed a card that cannot run the
job. Every test below is a property that failure violated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from v2d.brev import (
    MIN_VRAM_GB,
    eligible,
    is_dead,
    is_shell_ready,
    lookup,
    parse_search,
    pick_cheapest,
    status_of,
)

LIVE = Path(__file__).parent / "data" / "brev_search_2026-10-24.txt"


@pytest.fixture(scope="module")
def offers():
    found, skipped = parse_search(LIVE.read_text())
    assert not skipped, f"the live table should parse completely; skipped {skipped[:2]}"
    return found


def test_parses_every_row(offers):
    assert len(offers) == 343


def test_gpu_names_containing_spaces_do_not_shift_the_columns(offers):
    """12 rows name the GPU "RTX PRO Server 6000". Splitting on whitespace moves the price."""
    spacey = [offer for offer in offers if " " in offer.gpu]
    assert len(spacey) == 12
    big = lookup(offers, "g7e.2xlarge")
    assert big is not None
    assert big.gpu == "RTX PRO Server 6000"
    assert big.vram_per_gpu_gb == 96.0
    assert big.usd_per_hour == pytest.approx(4.04)


def test_the_broken_picker_s_choice_is_rejected(offers):
    """A V100 is 16 GB. It must never clear the floor, whatever it costs."""
    v100 = lookup(offers, "paperspace_V100")
    assert v100 is not None
    assert v100.vram_per_gpu_gb == 16.0
    assert not v100.meets()
    assert v100 not in eligible(offers)


def test_cheapness_never_beats_the_floor(offers):
    """hyperstack_A4000 is the cheapest card on the table at $0.18/h, and has 16 GB."""
    cheapest_overall = min(offers, key=lambda offer: offer.usd_per_hour)
    assert cheapest_overall.instance_type == "hyperstack_A4000"
    assert not cheapest_overall.meets()
    assert pick_cheapest(offers).instance_type != cheapest_overall.instance_type


def test_every_eligible_offer_clears_the_floor(offers):
    assert eligible(offers), "the live table should contain usable cards"
    assert all(offer.vram_per_gpu_gb >= MIN_VRAM_GB for offer in eligible(offers))


def test_eligible_is_ordered_cheapest_first(offers):
    prices = [offer.usd_per_hour for offer in eligible(offers)]
    assert prices == sorted(prices)


def test_pick_is_the_cheapest_card_that_fits(offers):
    pick = pick_cheapest(offers)
    assert pick is not None
    assert pick.vram_per_gpu_gb == 48.0
    assert pick.usd_per_hour == pytest.approx(0.68)
    # Among the three $0.68/h A6000s, the one with the most vCPUs. Observed on a 6-vCPU box:
    # FlashCHORD's startup is single-threaded and CPU-bound, sitting at 101 % of one core with
    # the GPU at 0 %, so vCPUs are worth a tie-break that costs nothing.
    assert pick.instance_type == "massedcompute_A6000_plus"
    assert pick.vcpus == 12


def test_the_vcpu_tie_break_never_costs_money(offers):
    """Price stays the primary key; vCPUs only order offers that already cost the same."""
    ranked = eligible(offers)
    assert ranked[0].usd_per_hour == min(offer.usd_per_hour for offer in ranked)
    prices = [offer.usd_per_hour for offer in ranked]
    assert prices == sorted(prices)


def test_a_vcpu_floor_can_be_required(offers):
    assert all(offer.vcpus >= 16 for offer in eligible(offers, min_vcpus=16))
    assert len(eligible(offers, min_vcpus=16)) < len(eligible(offers))


def test_vcpus_are_parsed_for_the_known_rows(offers):
    """hyperstack_A6000 is cheaper AND has 28 vCPUs; it is simply not always available."""
    plus = lookup(offers, "massedcompute_A6000_plus")
    assert plus is not None and plus.vcpus == 12
    base = lookup(offers, "massedcompute_A6000_base")
    assert base is not None and base.vcpus == 6


def test_multi_gpu_totals_do_not_satisfy_a_single_device_floor(offers):
    """FlashCHORD needs >=46,000 MiB on one device, so 2x16 GB advertising 32 GB total is out."""
    multi = [
        offer
        for offer in offers
        if offer.count > 1
        and offer.total_vram_gb >= MIN_VRAM_GB
        and offer.vram_per_gpu_gb < MIN_VRAM_GB
    ]
    assert multi, "the table should contain at least one such trap"
    for offer in multi:
        assert offer not in eligible(offers)


def test_an_unparseable_table_yields_nothing_rather_than_a_guess():
    offers, skipped = parse_search("some unrelated output\nwith no header at all\n")
    assert offers == []
    assert len(skipped) == 2
    assert pick_cheapest(offers) is None


def test_a_table_without_the_named_type_still_picks_something_that_fits(offers):
    """hyperstack_A6000 at $0.60 was listed in the morning and gone by the afternoon."""
    assert lookup(offers, "hyperstack_A6000") is None
    pick = pick_cheapest(offers)
    assert pick.meets()


def test_floor_is_configurable_and_monotonic(offers):
    assert len(eligible(offers, 80.0)) < len(eligible(offers, 46.0))
    assert all(offer.vram_per_gpu_gb >= 80.0 for offer in eligible(offers, 80.0))


def test_a_price_column_with_a_dollar_sign_is_read_as_money(offers):
    assert all(offer.usd_per_hour > 0 for offer in offers)


# ------------------------------------------------------------------ readiness

# Verbatim rows observed while the 2026-10-24 probe was coming up and going down.
DEPLOYING = (
    " v2d-gate-1024-1414  DEPLOYING  BUILDING   NOT READY  6i30q50mr  massedcompute_A6000  A6000"
)
RUNNING_NOT_READY = (
    " v2d-gate-1024-1414  RUNNING    COMPLETED  NOT READY  6i30q50mr  massedcompute_A6000  A6000"
)
RUNNING_READY = (
    " v2d-gate-1024-1414  RUNNING    COMPLETED  READY      6i30q50mr  massedcompute_A6000  A6000"
)
DELETING = (
    " v2d-gate-1024-1414  DELETING   COMPLETED  NOT READY  6i30q50mr  massedcompute_A6000  A6000"
)


def test_running_but_shell_not_ready_is_not_ready():
    """The bug that wasted a probe: brev says RUNNING before SSH exists."""
    assert not is_shell_ready(RUNNING_NOT_READY)


def test_deploying_is_not_ready():
    assert not is_shell_ready(DEPLOYING)


def test_running_and_shell_ready_is_ready():
    assert is_shell_ready(RUNNING_READY)


def test_not_ready_is_not_matched_as_ready_by_substring():
    """ "NOT READY" contains "READY", so the negative has to be checked first."""
    assert "READY" in RUNNING_NOT_READY
    assert not is_shell_ready(RUNNING_NOT_READY)


def test_dead_states_are_recognised():
    assert is_dead(DELETING)
    assert is_dead(" x FAILED")
    assert not is_dead(RUNNING_READY)
    assert not is_dead(DEPLOYING)


# The row brev printed for the box that failed on 2026-10-26. BUILD is EMPTY, so every column
# after it shifts left -- which is why the status is read by position and not by offset.
FAILURE_ROW = " v2d-t2-b08  FAILURE               NOT READY  hwuicymq  massedcompute_A6000_plus  A6000"


def test_the_word_brev_prints_for_a_failed_box_is_dead():
    """FAILURE, not FAILED. Missing it cost a 20 min SSH wait and an unreplaced box."""
    assert is_dead(FAILURE_ROW)
    assert not is_shell_ready(FAILURE_ROW)


def test_an_empty_build_column_does_not_shift_the_status_read():
    assert status_of(FAILURE_ROW) == "FAILURE"
    assert status_of(" v2d-t2-b06  RUNNING  COMPLETED  READY  9a5gj9ep7  massedcompute_A6000_plus") == "RUNNING"


def test_a_dead_word_outside_the_status_column_is_not_a_dead_instance():
    """The substring form would retire a healthy box over its own name or machine type."""
    assert not is_dead(" box-stopped  RUNNING  COMPLETED  READY  id  type  A6000")
    assert not is_dead(" b01  DEPLOYING  BUILDING  NOT READY  id  deleted_pool_a6000  A6000")


def test_a_row_too_short_to_have_a_status_falls_back_to_the_substring_form():
    assert is_dead("FAILURE")
    assert not is_dead("")
