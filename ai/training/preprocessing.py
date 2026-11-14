"""ScoreLine data preparation stages (2: acquire/parse, 3: inspect, 4: clean,
5: group/time split, 7a: label). Stage 6 (explore) lives in notebooks only."""
import pandas as pd

from .teams import COMPETITION_CODES, canonical_team

API_COLUMNS = ["id", "home_team", "away_team", "date", "status", "home_score", "away_score"]


def load_api_matches(matches: list[dict]) -> pd.DataFrame:
    """Stage 2: parse normalized matches from backend/app/services/football_api.py."""
    return pd.DataFrame(matches, columns=API_COLUMNS)


STATIC_REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
STATIC_COLUMNS = API_COLUMNS + ["competition", "season"]


def _season_start(dates: pd.Series) -> pd.Series:
    """Season start year (Aug..Jun seasons): 2024-10-05 and 2025-03-01 are both season 2024."""
    return dates.dt.year - (dates.dt.month < 7)


def load_static_csv(path: str, competition=None, seasons=None) -> pd.DataFrame:
    """Stage 2: load the cleaned football-data.co.uk history (ai/ml_pipeline/clean.py output).

    Expects the columns Div, Date, HomeTeam, AwayTeam, FTHG, FTAG (extra columns are ignored).
    Returns API_COLUMNS plus `competition` (PL, PD, BL1, SA, FL1) and `season` (start year),
    so it can be fed to load_api_matches / merge_matches like API data. Team names are
    converted to canonical names (see teams.py). Dates are date-only, stored as midnight UTC.

    competition: one code or a list, either style (PL or E0). seasons: list of start years.
    """
    raw = pd.read_csv(path)
    missing = [c for c in STATIC_REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"static CSV {path} is missing columns: {missing}")
    df = raw.dropna(subset=STATIC_REQUIRED).copy()
    df["Div"] = df["Div"].astype(str).str.strip()
    df = df[df["Div"].isin(COMPETITION_CODES)]
    dates = pd.to_datetime(df["Date"], utc=True)
    home = df["HomeTeam"].map(canonical_team)
    away = df["AwayTeam"].map(canonical_team)
    out = pd.DataFrame({
        "id": ("csv-" + df["Div"] + "-" + dates.dt.strftime("%Y%m%d") + "-"
               + home.str.replace(" ", "") + "-" + away.str.replace(" ", "")),
        "home_team": home,
        "away_team": away,
        "date": dates,
        "status": "FINISHED",
        "home_score": df["FTHG"].astype(int),
        "away_score": df["FTAG"].astype(int),
        "competition": df["Div"].map(COMPETITION_CODES),
        "season": _season_start(dates),
    })
    if competition is not None:
        wanted = [competition] if isinstance(competition, str) else list(competition)
        codes = {COMPETITION_CODES.get(c, c).upper() for c in wanted}
        out = out[out["competition"].isin(codes)]
    if seasons is not None:
        out = out[out["season"].isin(list(seasons))]
    return out[STATIC_COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def merge_matches(*frames: pd.DataFrame) -> pd.DataFrame:
    """Stage 2/4: combine match tables from several sources into one timeline.

    Team names become canonical and dates UTC. A match is identified by (home, away, season):
    in a league each ordered pair meets once per season, and this survives the small kickoff-time
    differences between sources. When a match appears in several frames the EARLIER argument
    wins, so pass API data first (real kickoff times and ids), static history after it.
    """
    parts = []
    for rank, frame in enumerate(f for f in frames if len(f)):
        part = frame.copy()
        part["home_team"] = part["home_team"].map(canonical_team)
        part["away_team"] = part["away_team"].map(canonical_team)
        part["date"] = pd.to_datetime(part["date"], utc=True)
        part["_rank"], part["_season"] = rank, _season_start(part["date"])
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=API_COLUMNS)
    df = pd.concat(parts, ignore_index=True).sort_values(["_rank", "date"], kind="stable")
    df = df.drop_duplicates(["home_team", "away_team", "_season"], keep="first")
    return df.drop(columns=["_rank", "_season"]).sort_values("date", kind="stable").reset_index(drop=True)


def inspect(df: pd.DataFrame) -> dict:
    """Stage 3: quick summary (shape, nulls, statuses) for sanity checks."""
    return {
        "rows": len(df),
        "nulls": df.isna().sum().to_dict(),
        "statuses": df["status"].value_counts().to_dict(),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: keep finished matches with scores, parse dates, sort by date."""
    df = df[df["status"] == "FINISHED"].dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    # TODO: dedupe by id, normalise team names across sources.
    return df.sort_values("date").reset_index(drop=True)


def label_result(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 7a: add 'result' label: H (home win), D (draw), A (away win)."""
    df = df.copy()
    diff = df["home_score"] - df["away_score"]
    df["result"] = "D"
    df.loc[diff > 0, "result"] = "H"
    df.loc[diff < 0, "result"] = "A"
    return df


def time_split(df: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 5: chronological split (no shuffling) so the test set is the future."""
    df = df.sort_values("date").reset_index(drop=True)
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[:cut], df.iloc[cut:]
