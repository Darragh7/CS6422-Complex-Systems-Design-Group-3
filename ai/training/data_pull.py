"""Stage 2 (acquire): pull historical finished matches from football-data.org v4.

Results are normalised to the same fields preprocessing.load_api_matches expects
and cached per competition/season as CSV under ai/training/data/ (git-ignored via
the root *.csv rule).

CLI: python -m ai.training.data_pull --competition PL --seasons 2024 2025
"""
import argparse
import os
import time
from pathlib import Path

import httpx
import pandas as pd

from .preprocessing import API_COLUMNS

DEFAULT_BASE_URL = "https://api.football-data.org/v4"
DATA_DIR = Path(__file__).parent / "data"
# Free tier allows ~10 requests/minute, so leave >= 6s between calls.
MIN_INTERVAL_SECONDS = 6.5


class DataPullError(Exception):
    """Raised when football-data.org does not return usable data."""


def normalize_match(match: dict) -> dict:
    """Same shape as backend/app/services/football_api.normalize_match.

    Kept local so training doesn't import backend settings.
    """
    full_time = (match.get("score") or {}).get("fullTime") or {}
    return {
        "id": match["id"],
        "home_team": (match.get("homeTeam") or {}).get("name") or "TBD",
        "away_team": (match.get("awayTeam") or {}).get("name") or "TBD",
        "date": match.get("utcDate"),
        "status": match.get("status"),
        "home_score": full_time.get("home"),
        "away_score": full_time.get("away"),
    }


def make_client(api_key: str | None = None, base_url: str | None = None, **kwargs) -> httpx.Client:
    """httpx client with X-Auth-Token from FOOTBALL_API_KEY (or the argument)."""
    api_key = api_key or os.getenv("FOOTBALL_API_KEY")
    if not api_key:
        raise DataPullError("FOOTBALL_API_KEY is not set")
    base_url = base_url or os.getenv("FOOTBALL_API_BASE_URL") or DEFAULT_BASE_URL
    return httpx.Client(base_url=base_url, headers={"X-Auth-Token": api_key}, timeout=30, **kwargs)


def fetch_season(client: httpx.Client, competition: str, season: int, sleep=time.sleep) -> list[dict]:
    """GET /competitions/{code}/matches?season=YYYY&status=FINISHED, normalised."""
    params = {"season": season, "status": "FINISHED"}
    for attempt in range(2):
        response = client.get(f"/competitions/{competition.upper()}/matches", params=params)
        if response.status_code == 429 and attempt == 0:
            sleep(min(float(response.headers.get("Retry-After", 60)), 60))
            continue
        break
    if response.status_code != 200:
        try:
            message = response.json().get("message")
        except ValueError:
            message = None
        raise DataPullError(
            f"{competition} {season}: HTTP {response.status_code} {message or ''}".strip()
        )
    return [normalize_match(m) for m in response.json().get("matches", [])]


def cache_path(competition: str, season: int, data_dir: Path = DATA_DIR) -> Path:
    return Path(data_dir) / f"{competition.upper()}_{season}.csv"


def pull_matches(
    competition: str = "PL",
    seasons: list[int] = (2024,),
    data_dir: Path = DATA_DIR,
    refresh: bool = False,
    client: httpx.Client | None = None,
    sleep=time.sleep,
    min_interval: float = MIN_INTERVAL_SECONDS,
) -> pd.DataFrame:
    """Fetch (or load cached) finished matches for each season; return one DataFrame."""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    frames, own_client, requested = [], client is None, False
    client = client or make_client()
    try:
        for season in seasons:
            path = cache_path(competition, season, data_dir)
            if path.exists() and not refresh:
                frames.append(pd.read_csv(path))
                continue
            if requested:
                sleep(min_interval)  # simple rate limit between API calls
            df = pd.DataFrame(fetch_season(client, competition, season, sleep), columns=API_COLUMNS)
            requested = True
            df.to_csv(path, index=False)
            frames.append(df)
    finally:
        if own_client:
            client.close()
    # TODO: merge with Sriram's static CSV and dedupe by id once its format is agreed.
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=API_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--competition", default="PL")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024])
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    df = pull_matches(args.competition, args.seasons, refresh=args.refresh)
    print(f"{len(df)} matches, {df['date'].min()} -> {df['date'].max()}")


if __name__ == "__main__":
    main()
