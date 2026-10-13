"""ScoreLine data preparation stages (2: acquire/parse, 3: inspect, 4: clean,
5: group/time split, 7a: label). Stage 6 (explore) lives in notebooks only."""
import pandas as pd

API_COLUMNS = ["id", "home_team", "away_team", "date", "status", "home_score", "away_score"]


def load_api_matches(matches: list[dict]) -> pd.DataFrame:
    """Stage 2: parse normalized matches from backend/app/services/football_api.py."""
    return pd.DataFrame(matches, columns=API_COLUMNS)


def load_static_csv(path: str) -> pd.DataFrame:
    """Stage 2: load Sriram's cleaned historical CSV."""
    # TODO: map CSV columns onto API_COLUMNS once the file format is agreed.
    raise NotImplementedError("Static CSV loader not implemented yet")


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
