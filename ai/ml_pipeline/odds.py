"""Extract bookmaker odds from the raw files and turn them into probabilities.

Usage (from ml/pipeline):
    python odds.py

Reads  data/raw/football-data-co-uk/<league>/<season>.csv   (untouched raw files)
Writes data/clean/odds.csv   (one row per match: market probabilities with the margin removed)

Why: bookmaker odds are the toughest benchmark in football prediction. Converting odds to
probabilities: 1/odds for each outcome, then divide by the sum so they add up to 1
(this removes the bookmaker's built-in profit margin, usually 2-6%).

For each match the first available source is used:
  Pinnacle closing odds -> Bet365 closing -> market-average closing -> Bet365 pre-match
Note: these are mostly CLOSING odds (final prices just before kickoff), so they are a very
hard benchmark and are NOT used as a model feature, because our predictions are made earlier.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from clean import read_csv_safely
from common import CLEAN_DIR

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))

SOURCES = [
    ("Pinnacle closing", ("PSCH", "PSCD", "PSCA")),
    ("Bet365 closing", ("B365CH", "B365CD", "B365CA")),
    ("Market average closing", ("AvgCH", "AvgCD", "AvgCA")),
    ("Bet365 pre-match", ("B365H", "B365D", "B365A")),
]


def load_one(path):
    df = read_csv_safely(path)
    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df.get("Date"), dayfirst=True, format="mixed", errors="coerce")
    df["HomeTeam"] = df.get("HomeTeam").astype("string").str.strip()
    df["AwayTeam"] = df.get("AwayTeam").astype("string").str.strip()
    out = df[["Date", "HomeTeam", "AwayTeam"]].copy()
    out["odds_source"] = pd.NA
    for col in ("odds_h", "odds_d", "odds_a"):
        out[col] = np.nan
    for label, (h, d, a) in SOURCES:
        if not all(c in df.columns for c in (h, d, a)):
            continue
        trip = df[[h, d, a]].apply(pd.to_numeric, errors="coerce")
        usable = out["odds_h"].isna() & trip.notna().all(axis=1) & (trip > 1).all(axis=1)
        out.loc[usable, ["odds_h", "odds_d", "odds_a"]] = trip[usable].to_numpy()
        out.loc[usable, "odds_source"] = label
    out["Season"] = f"20{path.stem[:2]}/{path.stem[2:]}"
    return out.dropna(subset=["Date", "HomeTeam", "AwayTeam"])


def main():
    files = sorted((RAW_DIR / "football-data-co-uk").glob("*/*.csv"))
    if not files:
        raise SystemExit("No raw files found. Run ingest.py first.")
    df = pd.concat([load_one(f) for f in files], ignore_index=True)
    df = df.drop_duplicates(["Date", "HomeTeam", "AwayTeam"])

    inv = 1 / df[["odds_h", "odds_d", "odds_a"]]
    total = inv.sum(axis=1)
    df["market_margin"] = (total - 1).round(4)
    df["market_p_home"] = (inv["odds_h"] / total).round(4)
    df["market_p_draw"] = (inv["odds_d"] / total).round(4)
    df["market_p_away"] = (inv["odds_a"] / total).round(4)

    have = df["odds_source"].notna()
    print(f"Matches read: {len(df)} | with usable odds: {have.sum()} ({have.mean():.1%})")
    print("\nCoverage by season (% of matches with odds) and source used:")
    cov = df.groupby("Season").agg(matches=("Date", "size"), with_odds=("odds_source", "count"))
    cov["pct"] = (cov["with_odds"] / cov["matches"] * 100).round(1)
    cov["main_source"] = df.groupby("Season")["odds_source"].agg(
        lambda s: s.mode().iloc[0] if s.notna().any() else "none")
    print(cov.to_string())
    print(f"\nAverage bookmaker margin: {df['market_margin'].mean():.1%}")

    out = df.loc[have, ["Date", "HomeTeam", "AwayTeam", "odds_source", "market_margin",
                        "market_p_home", "market_p_draw", "market_p_away"]]
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLEAN_DIR / "odds.csv", index=False)
    print(f"\nSaved {CLEAN_DIR / 'odds.csv'}")


if __name__ == "__main__":
    main()
