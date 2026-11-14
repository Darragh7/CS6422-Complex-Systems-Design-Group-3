import pandas as pd
import pytest

from ai.training import features as fe
from ai.training import preprocessing as pp

ROWS = [
    # Div, Date, HomeTeam, AwayTeam, FTHG, FTAG
    ("E0", "2024-08-17", "Man United", "Fulham", 1, 0),
    ("E0", "2024-08-24", "Fulham", "Man United", 0, 0),
    ("E0", "2025-02-01", "Arsenal", "Man United", 2, 2),
    ("SP1", "2024-08-18", "Ath Madrid", "Girona", 3, 1),
    ("D1", "2023-08-19", "Bayern Munich", "Dortmund", 4, 1),
]


@pytest.fixture
def static_csv(tmp_path):
    path = tmp_path / "matches.csv"
    pd.DataFrame(ROWS, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).assign(
        HS=10, FTR="H"  # extra columns must be ignored
    ).to_csv(path, index=False)
    return str(path)


def test_load_static_csv_matches_api_shape(static_csv):
    df = pp.load_static_csv(static_csv)
    assert list(df.columns) == pp.API_COLUMNS + ["competition", "season"]
    assert len(df) == 5
    assert (df["status"] == "FINISHED").all()
    assert str(df["date"].dt.tz) == "UTC"
    assert df["date"].is_monotonic_increasing
    assert df["id"].is_unique
    assert set(df["competition"]) == {"PL", "PD", "BL1"}


def test_load_static_csv_uses_canonical_team_names(static_csv):
    df = pp.load_static_csv(static_csv)
    assert {"Manchester United", "Atletico Madrid", "Bayern Munich"} <= set(df["home_team"])
    assert "Borussia Dortmund" in set(df["away_team"])
    assert "Man United" not in set(df["home_team"]) | set(df["away_team"])


def test_season_is_start_year_so_spring_matches_belong_to_the_autumn_season(static_csv):
    df = pp.load_static_csv(static_csv)
    arsenal = df[df["home_team"] == "Arsenal"].iloc[0]
    assert arsenal["season"] == 2024  # played 2025-02-01
    assert set(df["season"]) == {2023, 2024}


def test_filters_accept_both_code_styles_and_lists(static_csv):
    assert len(pp.load_static_csv(static_csv, competition="PL")) == 3
    assert len(pp.load_static_csv(static_csv, competition="E0")) == 3
    assert len(pp.load_static_csv(static_csv, competition=["PL", "SP1"])) == 4
    assert len(pp.load_static_csv(static_csv, seasons=[2023])) == 1
    assert len(pp.load_static_csv(static_csv, competition="PL", seasons=[2024])) == 3


def test_missing_columns_raise_a_clear_error(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Div": ["E0"], "Date": ["2024-08-17"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        pp.load_static_csv(str(path))


def test_incomplete_rows_and_unknown_leagues_are_dropped(tmp_path):
    path = tmp_path / "m.csv"
    pd.DataFrame(
        [("E0", "2024-08-17", "Arsenal", "Chelsea", 1, 0),
         ("E0", "2024-08-18", "Arsenal", "Chelsea", None, None),
         ("ZZ", "2024-08-19", "Arsenal", "Chelsea", 1, 1)],
        columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"],
    ).to_csv(path, index=False)
    assert len(pp.load_static_csv(str(path))) == 1


def test_merge_keeps_api_row_when_same_match_is_in_both_sources(static_csv):
    static = pp.load_static_csv(static_csv, competition="PL")
    api = pd.DataFrame([{"id": 501, "home_team": "Manchester United FC", "away_team": "Fulham FC",
                         "date": "2024-08-16T19:00:00Z", "status": "FINISHED",
                         "home_score": 1, "away_score": 0}])
    merged = pp.merge_matches(api, static)
    assert len(merged) == 3  # the Man United v Fulham match appears once
    kept = merged[(merged["home_team"] == "Manchester United") & (merged["away_team"] == "Fulham")]
    assert list(kept["id"]) == [501]
    assert merged["date"].is_monotonic_increasing
    assert set(merged["home_team"]) == {"Manchester United", "Fulham", "Arsenal"}


def test_merge_of_nothing_is_an_empty_api_shaped_frame():
    assert list(pp.merge_matches().columns) == pp.API_COLUMNS
    assert list(pp.merge_matches(pd.DataFrame(columns=pp.API_COLUMNS)).columns) == pp.API_COLUMNS


def test_static_history_flows_through_the_existing_pipeline_stages(tmp_path):
    teams = ["Arsenal", "Chelsea", "Liverpool", "Everton"]
    rows = [("E0", f"2024-09-{i + 1:02d}", teams[i % 4], teams[(i + 1) % 4], i % 3, (i + 1) % 2)
            for i in range(24)]
    path = tmp_path / "m.csv"
    pd.DataFrame(rows, columns=["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]).to_csv(path, index=False)
    static = pp.load_static_csv(str(path), competition="PL")
    df = pp.label_result(pp.clean(pp.load_api_matches(static.to_dict("records"))))
    features = fe.build_features(df)
    assert list(features.columns) == fe.FEATURE_COLUMNS
    assert len(features) == 24 and not features.isna().any().any()
    assert set(df["result"]) <= {"H", "D", "A"}
