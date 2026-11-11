"""Scoreline ingestion step (first stage of the ML pipeline).

Usage:
    python ingest.py historical --start 2018 --end 2025
    python ingest.py fixtures --days 14

Raw data is saved unmodified under data/raw/. Re-running skips files that
already exist (use --force to download again).
API keys are read from environment variables, never from code.
"""
import argparse
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import requests

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))

# football-data.co.uk league codes -> football-data.org competition codes
LEAGUES = {
    "E0": "PL",    # Premier League
    "SP1": "PD",   # La Liga
    "D1": "BL1",   # Bundesliga
    "I1": "SA",    # Serie A
    "F1": "FL1",   # Ligue 1
}

HIST_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
API_URL = "https://api.football-data.org/v4/competitions/{comp}/matches"

log = logging.getLogger("ingest")


def get_with_retry(url, retries=3, backoff=2, **kwargs):
    """GET with simple retry; raises after the last failed attempt."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30, **kwargs)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt == retries:
                raise
            time.sleep(backoff ** attempt)


def season_code(start_year):
    """2023 -> '2324' (the format football-data.co.uk uses)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def ingest_historical(start, end, force=False):
    for year in range(start, end + 1):
        code = season_code(year)
        for league in LEAGUES:
            out = RAW_DIR / "football-data-co-uk" / league / f"{code}.csv"
            if out.exists() and not force:
                log.info("Skipping %s (already downloaded)", out)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            resp = get_with_retry(HIST_URL.format(season=code, league=league))
            out.write_bytes(resp.content)
            log.info("Saved %s (%d bytes)", out, len(resp.content))


def ingest_fixtures(days):
    token = os.getenv("FOOTBALL_API_KEY")
    if not token:
        raise SystemExit("Set FOOTBALL_API_KEY in your environment (.env).")
    today = date.today().isoformat()
    until = (date.today() + timedelta(days=days)).isoformat()
    for comp in LEAGUES.values():
        out = RAW_DIR / "football-data-org" / comp / f"fixtures_{today}.json"
        if out.exists():
            log.info("Skipping %s (already fetched today)", out)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        params = {"dateFrom": today, "dateTo": until}
        resp = get_with_retry(
            API_URL.format(comp=comp),
            headers={"X-Auth-Token": token},
            params=params,
        )
        out.write_text(resp.text, encoding="utf-8")
        log.info("Saved %s", out)
        time.sleep(7)  # free tier is about 10 requests per minute


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Scoreline data ingestion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    hist = sub.add_parser("historical")
    hist.add_argument("--start", type=int, default=2018)
    hist.add_argument("--end", type=int, default=2025)
    hist.add_argument("--force", action="store_true")

    fix = sub.add_parser("fixtures")
    fix.add_argument("--days", type=int, default=14)

    args = parser.parse_args()
    if args.cmd == "historical":
        ingest_historical(args.start, args.end, args.force)
    else:
        ingest_fixtures(args.days)


if __name__ == "__main__":
    main()