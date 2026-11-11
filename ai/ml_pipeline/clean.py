"""Scoreline cleaning step: merge raw football-data.co.uk CSVs into one table.

Usage:
    python clean.py

Reads  data/raw/football-data-co-uk/<league>/<season>.csv
Writes data/clean/matches.csv      (one row per match)
       data/clean/team_names.csv   (every team name seen, for the name-mapping file)
Raw files are never modified.
"""
import logging
import os
from pathlib import Path

import pandas as pd

RAW_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
CLEAN_DIR = Path(os.getenv("CLEAN_DATA_DIR", "data/clean"))

CORE = ["Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
OPTIONAL = ["HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR"]
LEAGUE_NAMES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "I1": "Serie A",
    "F1": "Ligue 1",
}

log = logging.getLogger("clean")


def season_label(code):
    """'1819' -> '2018/19'."""
    return f"20{code[:2]}/{code[2:]}"

def read_csv_safely(path):
    """Read a CSV that may start with a UTF-8 BOM or use Latin-1 accents."""
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read {path}")

def load_file(path):
    df = read_csv_safely(path)
    df.columns = df.columns.str.strip()
    df = df.reindex(columns=CORE + OPTIONAL)  # adds missing columns as empty
    df["Season"] = season_label(path.stem)
    df["SourceFile"] = f"{path.parent.name}/{path.name}"
    return df


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    files = sorted((RAW_DIR / "football-data-co-uk").glob("*/*.csv"))
    if not files:
        raise SystemExit(f"No CSV files found under {RAW_DIR / 'football-data-co-uk'}")
    log.info("Found %d raw files", len(files))

    df = pd.concat([load_file(f) for f in files], ignore_index=True)
    total_raw = len(df)

    # Standardize types
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed", errors="coerce")
    for col in ["FTHG", "FTAG"] + OPTIONAL:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["Div", "HomeTeam", "AwayTeam", "FTR"]:
        df[col] = df[col].astype("string").str.strip()

    # Drop rows missing any core field (blank lines, postponed matches with no result)
    before = len(df)
    files_before = set(df["SourceFile"])
    df = df.dropna(subset=CORE).copy()
    log.info("Dropped %d rows with missing core fields", before - len(df))
    for lost in sorted(files_before - set(df["SourceFile"])):
        log.warning("ALL rows lost from %s - check that file's columns and encoding", lost)

    # Drop exact duplicates of the same match
    before = len(df)
    df = df.drop_duplicates(subset=["Div", "Date", "HomeTeam", "AwayTeam"])
    log.info("Dropped %d duplicate matches", before - len(df))

    # Consistency check: FTR must agree with the goals
    expected = df.apply(
        lambda r: "H" if r.FTHG > r.FTAG else ("A" if r.FTHG < r.FTAG else "D"), axis=1
    )
    bad = df[expected != df["FTR"]]
    if len(bad):
        log.warning("%d rows where FTR disagrees with goals (dropping them)", len(bad))
        df = df.drop(bad.index)

    df["League"] = df["Div"].map(LEAGUE_NAMES)
    df = df.sort_values(["Date", "Div", "HomeTeam"]).reset_index(drop=True)

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DIR / "matches.csv", index=False)

    names = (
        pd.concat([df[["Div", "HomeTeam"]].rename(columns={"HomeTeam": "TeamName"}),
                   df[["Div", "AwayTeam"]].rename(columns={"AwayTeam": "TeamName"})])
        .drop_duplicates()
        .sort_values(["Div", "TeamName"])
    )
    names["CanonicalName"] = names["TeamName"]  # edit this column to build the mapping
    names.to_csv(CLEAN_DIR / "team_names.csv", index=False)

    # Summary report
    print("\nRaw rows read:", total_raw, "| clean matches:", len(df))
    print("\nMatches per league and season:")
    print(df.pivot_table(index="Season", columns="League", values="FTR", aggfunc="count").fillna(0).astype(int))
    print("\nResult split (H = home win, D = draw, A = away win):")
    print((df["FTR"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nAverage goals per match: home %.2f, away %.2f" % (df["FTHG"].mean(), df["FTAG"].mean()))
    print("\nMissing optional stats (% of rows):")
    print((df[OPTIONAL].isna().mean() * 100).round(1).to_string())
    print("\nSaved:", CLEAN_DIR / "matches.csv", "and", CLEAN_DIR / "team_names.csv")


if __name__ == "__main__":
    main()