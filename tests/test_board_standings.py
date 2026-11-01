"""The field's rank sums, and the two ways of counting that would flatter us.

The standings decide where money goes, so the arithmetic that produces them is pinned here:
a tie ranks behind, a truncated success rate is snapped to clip counts, and one team that
uploaded under two spellings of its name stays one team.
"""

from __future__ import annotations

from pathlib import Path

from tools_t1_shim import standings

HEADER = "teamId,teamName,submissionDate,score\n"


def write_board(path: Path, rows: list[tuple[str, str, float]]) -> None:
    body = "".join(f"1,{team},{when},{score}\n" for team, when, score in rows)
    path.write_text(HEADER + body)


def board(rows: list[tuple[str, str, float]], tmp_path: Path, name: str = "b.csv"):
    path = tmp_path / name
    write_board(path, rows)
    return standings.read_board(path, {})


def test_tie_ranks_behind_when_the_other_team_uploaded_first(tmp_path: Path) -> None:
    rows = [("them", "2026-10-01", 0.5), ("us", "2026-10-02", 0.5)]
    b = board(rows, tmp_path)
    assert standings.rank_of(b, "auc", "us") == 2
    assert standings.rank_of(b, "auc", "them") == 1


def test_mppe_is_lower_is_better(tmp_path: Path) -> None:
    b = board([("them", "2026-10-01", 20.0), ("us", "2026-10-02", 10.0)], tmp_path)
    assert standings.rank_of(b, "mppe", "us") == 1
    assert standings.rank_of(b, "auc", "us") == 2


def test_truncated_success_rate_is_snapped_to_clip_counts(tmp_path: Path) -> None:
    """16/30 publishes as 0.53333; our own entry computes 0.5333333. That is a tie."""
    b = board([("them", "2026-10-01", 0.53333), ("us", "2026-10-02", 0.5333333)], tmp_path)
    assert standings.rank_of(b, "sp-sr", "us") == 2, "snapped, so it is a tie and ties rank behind"
    assert standings.rank_of(b, "auc", "us") == 1, "unsnapped, the extra digits would read as ahead"


def test_one_clip_more_really_does_promote(tmp_path: Path) -> None:
    b = board([("them", "2026-10-01", 0.53333), ("us", "2026-10-02", 17 / 30)], tmp_path)
    assert standings.rank_of(b, "sp-sr", "us") == 1


def test_case_variant_names_fold_onto_one_team(tmp_path: Path) -> None:
    """MBZUAI uploaded as both "...Group" and "...GROUP"; two rows are not two teams."""
    canon: dict[str, str] = {}
    one = tmp_path / "one.csv"
    two = tmp_path / "two.csv"
    write_board(one, [("MBZUAI-RCL Group", "2026-10-01", 0.9)])
    write_board(two, [("MBZUAI-RCL GROUP", "2026-10-01", 0.8)])
    a = standings.read_board(one, canon)
    b = standings.read_board(two, canon)
    assert set(a) == set(b) == {"MBZUAI-RCL Group"}


def test_earliest_submission_is_the_row_kept(tmp_path: Path) -> None:
    """Kaggle publishes one row per team, but a merged board must keep the tie-break date."""
    b = board(
        [("us", "2026-10-05", 0.5), ("us", "2026-10-01", 0.5), ("them", "2026-10-03", 0.5)],
        tmp_path,
    )
    assert b["us"][1] == "2026-10-01"
    assert standings.rank_of(b, "auc", "us") == 1


def test_absence_is_charged_only_in_the_tied_last_column(tmp_path: Path) -> None:
    """A team on one board must not out-rank a team on twelve, which `strict` alone would do."""
    for tier in ("1", "2", "3"):
        for metric in ("auc", "sp-sr", "mp-sr", "mppe"):
            rows = [("broad", "2026-10-02", 0.5), ("deep", "2026-10-01", 0.9)]
            if tier != "1" or metric != "auc":
                rows = [rows[0]]
            write_board(tmp_path / f"tier{tier}-{metric}.csv", rows)

    out = standings.standings(tmp_path)
    rows = {r["team"]: r for r in out["standings"]}
    assert rows["deep"]["strict"] == 1, "one board, first place"
    assert rows["broad"]["strict"] == 2 + 11, "second on the contested board, first on eleven"
    assert rows["deep"]["entered"] == 1
    assert rows["broad"]["place_tied_last"] == 1, "breadth wins once absence is charged"
    assert rows["deep"]["place_strict"] == 1, "and loses when it is not"


def test_a_short_row_is_skipped_not_read_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    path.write_text(HEADER + "1,us,2026-10-01,0.5\n1,truncated\n")
    b = standings.read_board(path, {})
    assert set(b) == {"us"}
