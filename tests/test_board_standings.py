"""The ranking rule, pinned against the organisers' own implementation.

Three of these tests previously asserted the opposite of the truth: that a tie ranks behind,
on the assumption that Kaggle's submission-time ordering decided the metric rank that feeds
the FAQ's sum. The official aggregator settles it -- leaderboard.js lines 74-75, "teams with
the same score share a rank, and the next team skips past them" -- and under the real rule a
gain pays nothing until it crosses a *distinct* score. That error made every near-miss look
like a rank for sale and priced a purchase that was worth zero, so the step-function case is
pinned here by name.

Also pinned: the tiers are three separate standings and never summed (line 251, and there is
no rank sum anywhere in that file); baselines do not occupy ranks (line 82); and a team
missing any of a tier's four metrics is unranked rather than charged a last place (lines
66-71), which replaces the old strict/tied_last guesswork.
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


def write_tier(tmp_path: Path, tier: str, scores: dict[str, dict[str, float]]) -> None:
    """scores[metric][team] -> one CSV per metric for that tier."""
    for metric, by_team in scores.items():
        write_board(
            tmp_path / f"tier{tier}-{metric}.csv",
            [(team, "2026-10-01", value) for team, value in by_team.items()],
        )


def test_a_tie_shares_the_rank_regardless_of_who_uploaded_first(tmp_path: Path) -> None:
    b = board([("them", "2026-10-01", 0.5), ("us", "2026-10-02", 0.5)], tmp_path)
    assert standings.competition_rank(b, "auc", "us") == 1
    assert standings.competition_rank(b, "auc", "them") == 1


def test_the_next_team_skips_past_a_tie(tmp_path: Path) -> None:
    rows = [("a", "d", 0.9), ("b", "d", 0.9), ("c", "d", 0.9), ("d", "d", 0.5)]
    b = board(rows, tmp_path)
    assert [standings.competition_rank(b, "auc", t) for t in ("a", "b", "c")] == [1, 1, 1]
    assert standings.competition_rank(b, "auc", "d") == 2, "1+distinct better scores, not 1+teams"


def test_a_gain_inside_the_same_rung_buys_nothing(tmp_path: Path) -> None:
    """The error that cost real money: 21, 22 and 23 of 30 clips are all one rank.

    tier1-sp-sr holds us at 21/30 with another team on the same score and the next distinct
    score at 25/30. Buying one clip, or two, does not cross anybody.
    """
    rows = [("ahead", "d", 25 / 30), ("them", "d", 21 / 30), ("us", "d", 21 / 30)]
    b = board(rows, tmp_path)
    assert standings.competition_rank(b, "sp-sr", "us") == 2
    for clips in (22, 23, 24):
        b["us"] = clips / 30
        assert standings.competition_rank(b, "sp-sr", "us") == 2, f"{clips} clips still rank 2"
    b["us"] = 25 / 30
    assert standings.competition_rank(b, "sp-sr", "us") == 1, "only crossing a distinct score pays"


def test_mppe_is_lower_is_better(tmp_path: Path) -> None:
    b = board([("them", "d", 14.0), ("us", "d", 12.0)], tmp_path)
    assert standings.competition_rank(b, "mppe", "us") == 1
    assert standings.competition_rank(b, "auc", "us") == 2


def test_truncated_success_rate_is_snapped_to_clip_counts(tmp_path: Path) -> None:
    """16/30 publishes as 0.53333; unsnapped it reads as a distinct score instead of a tie."""
    b = board([("them", "d", 0.53333), ("us", "d", 16 / 30)], tmp_path)
    assert standings.competition_rank(b, "sp-sr", "us") == 1
    assert standings.competition_rank(b, "sp-sr", "them") == 1, "snapped, so they share the rank"
    assert standings.competition_rank(b, "auc", "us") == 1, "unsnapped, the digits read as ahead"
    assert standings.competition_rank(b, "auc", "them") == 2


def test_baselines_do_not_occupy_a_rank(tmp_path: Path) -> None:
    b = board([("CHORD Baseline", "d", 0.9), ("us", "d", 0.5)], tmp_path)
    assert set(b) == {"us"}, "the aggregator drops is_baseline rows before ranking"
    assert standings.competition_rank(b, "auc", "us") == 1


def test_case_variant_names_fold_onto_one_team(tmp_path: Path) -> None:
    canonical: dict[str, str] = {}
    write_board(tmp_path / "x.csv", [("MBZUAI-RCL Group", "d", 0.5)])
    write_board(tmp_path / "y.csv", [("MBZUAI-RCL GROUP", "d", 0.6)])
    a = standings.read_board(tmp_path / "x.csv", canonical)
    b = standings.read_board(tmp_path / "y.csv", canonical)
    assert set(a) == set(b) == {"MBZUAI-RCL Group"}


def test_the_best_row_is_kept_not_the_earliest(tmp_path: Path) -> None:
    """Submission time no longer breaks ties, so a duplicated team keeps its better score."""
    write_board(tmp_path / "c.csv", [("us", "2026-10-01", 0.40), ("us", "2026-10-02", 0.60)])
    assert standings.read_board(tmp_path / "c.csv", {}, "auc")["us"] == 0.60
    write_board(tmp_path / "d.csv", [("us", "2026-10-01", 14.0), ("us", "2026-10-02", 12.0)])
    assert standings.read_board(tmp_path / "d.csv", {}, "mppe")["us"] == 12.0, "lower is better"


def test_a_short_row_is_skipped_not_read_as_zero(tmp_path: Path) -> None:
    path = tmp_path / "b.csv"
    path.write_text(HEADER + "1,us,2026-10-01,0.5\n1,them,2026-10-02,\n")
    assert set(standings.read_board(path, {})) == {"us"}


def test_an_incomplete_team_is_unranked_not_charged_a_last_place(tmp_path: Path) -> None:
    """A team that entered one board of a tier is not a team we are behind."""
    write_tier(
        tmp_path,
        "1",
        {
            "auc": {"us": 0.5, "partial": 0.9},
            "sp-sr": {"us": 0.5},
            "mp-sr": {"us": 0.5},
            "mppe": {"us": 12.0},
        },
    )
    table = standings.standings(tmp_path)["tiers"]["tier1"]
    assert table["unranked_incomplete"] == ["partial"]
    assert [r["team"] for r in table["standings"]] == ["us"]
    assert table["standings"][0]["rank_sum"] == 4, "first on all four, with partial excluded"


def test_tiers_are_three_standings_and_are_never_summed(tmp_path: Path) -> None:
    for tier in ("1", "2"):
        write_tier(
            tmp_path,
            tier,
            {
                m: {"us": 12.0 if m == "mppe" else 0.5, "them": 0.9 if m == "auc" else 0.4}
                for m in standings.METRICS
            },
        )
    out = standings.standings(tmp_path)
    assert set(out["tiers"]) == {"tier1", "tier2"}, "a tier with four boards is one standing"
    assert "rank_sum" not in out, "there is no cross-tier total"
    assert out["tiers"]["tier1"]["standings"] == out["tiers"]["tier2"]["standings"]


def test_the_standings_table_shares_places_too(tmp_path: Path) -> None:
    write_tier(
        tmp_path,
        "1",
        {m: {"a": 0.5, "b": 0.5, "c": 0.1} for m in standings.METRICS},
    )
    rows = {r["team"]: r for r in standings.standings(tmp_path)["tiers"]["tier1"]["standings"]}
    assert rows["a"]["place"] == rows["b"]["place"] == 1
    assert rows["c"]["place"] == 3, "two teams tied for first, the third is third"
