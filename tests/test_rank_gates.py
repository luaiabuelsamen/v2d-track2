"""The two mistakes that hid a rank point for a week.

The first is ranking by metric instead of by rank: the challenge sums per-metric ranks, so a swap
that spends AUC inside its cushion to buy MPPE past the team above is a gain the Pareto criterion
in `results/t1_entry_pareto_optimal.json` rejects. The second is that a tie does not promote --
Kaggle breaks ties by submission time and every team we tie with uploaded earlier -- so matching
the score above us is worth nothing and must not read as a rank gained.

The margin side pins the gate arithmetic: SP-SR is an AND of two means with different units, and
reporting the wrong one sends a run after the wrong reward term.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools_t1_shim import gate_margins, rank_swap


def test_tie_ranks_behind_not_level():
    board = [0.9, 0.7, 0.5]
    assert rank_swap.rank(board, "spider_sr", 0.7) == 3, "a tie must rank behind the team it ties"
    assert rank_swap.rank(board, "spider_sr", 22 / 30) == 2, "one more clip promotes"
    assert rank_swap.rank(board, "add_auc", 0.70001) == 2, "a continuous board promotes on any gain"


def test_lower_is_better_for_mppe_only():
    board = [11.997, 13.044]
    assert rank_swap.rank(board, "mppe_cm", 12.060) == 2
    assert rank_swap.rank(board, "mppe_cm", 11.996) == 1
    assert rank_swap.rank(board, "mppe_cm", 14.0) == 3, "both are nearer the reference"
    assert rank_swap.rank(board, "add_auc", 14.0) == 1, "the same number tops an ascending board"


def test_page_token_line_and_odd_rows(tmp_path):
    """The leaderboard CSV opens with a page token, quotes names holding commas, and may run short.

    A quoted name is not a reason to drop the row -- it parses, and dropping it would understate
    how many teams are ahead. A row with no score is, and must not read as 0.0, which on MPPE
    would look like a perfect competitor.
    """
    (tmp_path / "tier1-auc.csv").write_text(
        "Next Page Token = abc\n"
        "teamId,teamName,submissionDate,score\n"
        "1,Someone,2026-10-01 00:00:00,0.9\n"
        '2,"Javier Medel, MSc.",2026-10-02 00:00:00,0.2\n'
        "3,Us,2026-10-03 00:00:00,0.5\n"
        "4,Truncated\n"
    )
    assert rank_swap.read_board(tmp_path, "add_auc", "Us") == [0.9, 0.2]


def test_aggregate_swaps_replace_rather_than_append():
    entry = {
        "e00": dict.fromkeys(rank_swap.METRICS, 1.0),
        "e01": dict.fromkeys(rank_swap.METRICS, 3.0),
    }
    agg = rank_swap.aggregate(entry, {"e01": dict.fromkeys(rank_swap.METRICS, 1.0)})
    assert agg["add_auc"] == 1.0, "a swap must replace its clip, not add a 31st"


def _frames(position_m: float, orientation_rad: float):
    """One episode at a constant offset: a yaw of 2*theta about z reads back as theta of error."""
    frames = 4
    half = orientation_rad / 2.0
    solution = pd.DataFrame(
        {
            "row_id": [f"track_2/e000016/w000/f{i:06d}/o0" for i in range(frames)],
            **{f"ref_{c}": 0.0 for c in gate_margins.POSITION},
            "ref_quat_x": 0.0,
            "ref_quat_y": 0.0,
            "ref_quat_z": 0.0,
            "ref_quat_w": 1.0,
        }
    )
    submission = pd.DataFrame(
        {
            "row_id": solution["row_id"],
            "pos_x": position_m,
            "pos_y": 0.0,
            "pos_z": 0.0,
            "quat_x": 0.0,
            "quat_y": 0.0,
            "quat_z": np.sin(half),
            "quat_w": np.cos(half),
        }
    )
    return solution, submission


def test_sp_sr_is_an_and_of_two_gates():
    inside = gate_margins.margins(*_frames(0.05, 0.4))[0]
    assert inside["spider_sr"] is True

    position_only = gate_margins.margins(*_frames(0.11, 0.4))[0]
    assert position_only["spider_sr"] is False
    assert "SP pos by 1.00 cm" in gate_margins.describe(position_only)
    assert "SP ori" not in gate_margins.describe(position_only)

    orientation_only = gate_margins.margins(*_frames(0.05, 0.522))[0]
    assert orientation_only["spider_sr"] is False
    assert "SP ori by 0.022 rad" in gate_margins.describe(orientation_only)
    assert "SP pos" not in gate_margins.describe(orientation_only)


def test_e16_shape_passes_mp_sr_while_failing_sp_sr():
    """0.522 rad is over SP-SR's 0.5 rad and under MP-SR's 30 degrees; both must be reported."""
    row = gate_margins.margins(*_frames(0.0186, 0.522))[0]
    assert row["spider_sr"] is False
    assert row["maniptrans_sr"] is True
    assert np.degrees(row["ori_rad"]) < gate_margins.MP_ORIENTATION_DEG


def test_geodesic_angle_folds_the_double_cover():
    a = np.array([[0.0, 0.0, 0.0, 1.0]])
    assert gate_margins.geodesic_angle(a, a)[0] == 0.0
    assert gate_margins.geodesic_angle(a, -a)[0] == 0.0, "q and -q are the same rotation"


def test_truncated_success_rate_is_snapped_to_clip_counts():
    """16/30 prints as 0.53333 on the board and computes as 0.5333333 for us.

    Compared raw, our own entry looks ahead of a team on exactly the same 16 clips, and the tool
    reports a rank point that does not exist. Both sides have to be read as clip counts.
    """
    board = [0.53333]  # 16/30 as Kaggle prints it
    assert rank_swap.rank(board, "maniptrans_sr", 16 / 30) == 2, "same 16 clips is a tie, not a win"
    assert rank_swap.rank(board, "maniptrans_sr", 17 / 30) == 1
    assert rank_swap.rank(board, "maniptrans_sr", 15 / 30) == 2

    # AUC is continuous, so nothing is snapped there.
    assert rank_swap.rank([0.53333], "add_auc", 16 / 30) == 1
