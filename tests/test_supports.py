"""Known-answer tests for support matching.

The asymmetry here is the whole design, and both directions have a concrete failure behind them:
`watering_can` is held in mid-air and must be refused, while `traffic_cone` penetrates its table
by 2.3 cm and must be accepted.
"""

from __future__ import annotations

import textwrap

import pytest

from v2d.supports import (
    ABOVE_TOLERANCE_CM,
    BELOW_TOLERANCE_CM,
    candidate_heights,
    match_support,
    read_primitives,
)

FLOOR_ONLY = [0.0]
FLOOR_AND_TABLE = [0.0, 0.771]  # traffic_cone's table top, 77.1 cm


def test_an_object_resting_on_the_floor_matches_the_floor():
    assert match_support(0.002, FLOOR_ONLY) == 0.0


def test_penetration_below_the_floor_is_unbounded():
    """A 20 degree tilt on an 86 cm box buries a corner ~15 cm. That is the error being fixed."""
    assert match_support(-0.15, FLOOR_ONLY) == 0.0
    assert match_support(-1.0, FLOOR_ONLY) == 0.0


def test_hovering_above_the_floor_is_refused():
    """watering_can, held steady in mid-air while pouring, sits 13-14 cm up with no primitive."""
    assert match_support(0.13, FLOOR_ONLY) is None


def test_the_above_tolerance_boundary():
    above = ABOVE_TOLERANCE_CM / 100.0
    assert match_support(above - 1e-6, FLOOR_ONLY) == 0.0
    assert match_support(above + 1e-6, FLOOR_ONLY) is None


def test_a_table_resting_object_matches_the_table():
    assert match_support(0.773, FLOOR_AND_TABLE) == pytest.approx(0.771)


def test_penetration_below_a_table_is_bounded_but_allows_the_real_case():
    """traffic_cone measures 74.8 cm against a 77.1 cm top: 2.3 cm of tilt-driven penetration."""
    assert match_support(0.748, FLOOR_AND_TABLE) == pytest.approx(0.771)
    below = BELOW_TOLERANCE_CM / 100.0
    assert match_support(0.771 - below + 1e-6, FLOOR_AND_TABLE) == pytest.approx(0.771)
    assert match_support(0.771 - below - 1e-6, FLOOR_AND_TABLE) is None


def test_a_floor_segment_is_never_mistaken_for_a_table_segment():
    """The safety argument for the 10 cm window: the lowest table top is 46.9 cm up."""
    for height in (-0.2, -0.05, 0.0, 0.02):
        assert match_support(height, [0.0, 0.469]) == 0.0


def test_a_segment_between_floor_and_table_matches_neither():
    assert match_support(0.30, FLOOR_AND_TABLE) is None


def test_squeegee_sweeping_the_floor_does_not_match_a_distant_chair_top():
    """Its primitives are at 69.8 and 71.6 cm; the rests are at 3.9-8.3 cm."""
    candidates = [0.0, 0.698, 0.716]
    for height in (0.039, 0.059, 0.053, 0.083):
        assert match_support(height, candidates) is None


def test_two_tops_a_few_millimetres_apart_are_interchangeable():
    """e25 ships tops 0.36 cm apart. Confusing them is harmless: they are the same plane."""
    matched = match_support(0.781, [0.0, 0.7833, 0.7869])
    assert matched in (pytest.approx(0.7833), pytest.approx(0.7869))


# ---------------------------------------------------------------- usda parsing

CYLINDER_TABLE = textwrap.dedent("""\
    #usda 1.0
    (
        defaultPrim = "support_surfaces"
        metersPerUnit = 1
        upAxis = "Z"
    )

    def Xform "support_surfaces"
    {
        def Cylinder "object_0" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {
            uniform token axis = "Z"
            double height = 0.04
            double radius = 0.12478255645333762
            double3 xformOp:translate = (-0.293, 0.643, 0.7511897739673112)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    """)

VERTICAL_PANEL = textwrap.dedent("""\
    #usda 1.0
    (
        defaultPrim = "support_surfaces"
        metersPerUnit = 1
        upAxis = "Z"
    )

    def Xform "support_surfaces"
    {
        def Cube "chair_back" ( prepend apiSchemas = ["PhysicsCollisionAPI"] )
        {
            double size = 1
            double3 xformOp:translate = (-0.5324, -0.486, 0.715)
            double3 xformOp:scale = (0.02, 0.12, 0.27)
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
        }
    }
    """)

EMPTY = textwrap.dedent("""\
    #usda 1.0
    (
        defaultPrim = "support_surfaces"
        metersPerUnit = 1
        upAxis = "Z"
    )

    def Xform "support_surfaces"
    {
    }
    """)


def _write(tmp_path, text, name="s.usda"):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_a_cylinder_s_top_is_its_centre_plus_half_its_height(tmp_path):
    """The bug this guards: reading the RADIUS as the half-height put the top 10 cm too high."""
    found = read_primitives(_write(tmp_path, CYLINDER_TABLE))
    assert len(found) == 1
    assert found[0].top_height_m == pytest.approx(0.7511897739673112 + 0.02)
    assert found[0].horizontal


def test_a_panel_thin_in_x_is_not_a_surface_you_rest_on(tmp_path):
    """vacuum's only primitive is a chair back: something you lean against, pinning no tilt."""
    found = read_primitives(_write(tmp_path, VERTICAL_PANEL))
    assert len(found) == 1
    assert not found[0].horizontal
    assert candidate_heights(_write(tmp_path, VERTICAL_PANEL, "p.usda")) == [0.0]


def test_an_episode_with_no_primitive_offers_only_the_floor(tmp_path):
    assert candidate_heights(_write(tmp_path, EMPTY)) == [0.0]


def test_no_primitive_file_at_all_offers_only_the_floor():
    assert candidate_heights(None) == [0.0]


def test_candidate_heights_includes_a_horizontal_top(tmp_path):
    heights = candidate_heights(_write(tmp_path, CYLINDER_TABLE))
    assert heights[0] == 0.0
    assert heights[1] == pytest.approx(0.7711897739673112)
