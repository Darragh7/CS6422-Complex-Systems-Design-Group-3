"""Team-name normalisation: one canonical name per club across data sources.

football-data.co.uk writes "Man United", football-data.org writes "Manchester United FC".
Features such as Elo and rolling form are keyed by team name, so the same club under two
spellings would silently be treated as two teams. Every source goes through
`canonical_team` before features are built.

How a name is resolved:
 1. exact match in SOURCE_ALIASES / API_ALIASES,
 2. otherwise a normalised key (accents, case, punctuation, club-type words such as "FC",
    "AFC", "CF" and founding years removed) is looked up, so "Manchester United FC" and
    "Manchester United" meet without an explicit alias,
 3. otherwise the cleaned name is returned unchanged (or UnknownTeamError if strict=True).

API_ALIASES were written from knowledge of football-data.org naming and have NOT been checked
against the live API. Run the check below on cached API pulls and add any missing spelling to
API_ALIASES:

    python -m ai.training.teams --check                     # all ai/training/data/*.csv
    python -m ai.training.teams --check path/to/file.csv    # specific files
"""
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterable

# football-data.co.uk league code -> football-data.org competition code
COMPETITION_CODES = {"E0": "PL", "SP1": "PD", "D1": "BL1", "I1": "SA", "F1": "FL1"}

# Canonical club names per competition (every club seen in 2018/19 onward).
CANONICAL = {
    "PL": [
        "AFC Bournemouth", "Arsenal", "Aston Villa", "Brentford", "Brighton and Hove Albion",
        "Burnley", "Cardiff City", "Chelsea", "Coventry City", "Crystal Palace", "Everton",
        "Fulham", "Huddersfield Town", "Hull City", "Ipswich Town", "Leeds United",
        "Leicester City", "Liverpool", "Luton Town", "Manchester City", "Manchester United",
        "Newcastle United", "Norwich City", "Nottingham Forest", "Sheffield United",
        "Southampton", "Sunderland", "Tottenham Hotspur", "Watford", "West Bromwich Albion",
        "West Ham United", "Wolverhampton Wanderers",
    ],
    "PD": [
        "Alaves", "Almeria", "Athletic Bilbao", "Atletico Madrid", "Barcelona", "Cadiz",
        "Celta Vigo", "Deportivo La Coruna", "Eibar", "Elche", "Espanyol", "Getafe", "Girona",
        "Granada", "Huesca", "Las Palmas", "Leganes", "Levante", "Malaga", "Mallorca",
        "Osasuna", "Racing Santander", "Rayo Vallecano", "Real Betis", "Real Madrid",
        "Real Oviedo", "Real Sociedad", "Real Valladolid", "Sevilla", "Valencia", "Villarreal",
    ],
    "BL1": [
        "Arminia Bielefeld", "Bayer Leverkusen", "Bayern Munich", "Borussia Dortmund",
        "Borussia Monchengladbach", "Darmstadt 98", "Eintracht Frankfurt", "FC Augsburg",
        "FC Heidenheim", "FC Koln", "FC Nurnberg", "Fortuna Dusseldorf", "Greuther Furth",
        "Hamburger SV", "Hannover 96", "Hertha Berlin", "Holstein Kiel", "Mainz 05",
        "RB Leipzig", "SC Freiburg", "SC Paderborn", "SV Elversberg", "Schalke 04", "St Pauli",
        "TSG Hoffenheim", "Union Berlin", "VfB Stuttgart", "VfL Bochum", "VfL Wolfsburg",
        "Werder Bremen",
    ],
    "SA": [
        "AC Milan", "AS Roma", "Atalanta", "Benevento", "Bologna", "Brescia", "Cagliari",
        "Chievo", "Como", "Cremonese", "Crotone", "Empoli", "Fiorentina", "Frosinone", "Genoa",
        "Hellas Verona", "Inter Milan", "Juventus", "Lazio", "Lecce", "Monza", "Napoli",
        "Parma", "Pisa", "SPAL", "Salernitana", "Sampdoria", "Sassuolo", "Spezia", "Torino",
        "Udinese", "Venezia",
    ],
    "FL1": [
        "Ajaccio", "Amiens", "Angers", "Auxerre", "Bordeaux", "Brest", "Caen", "Clermont",
        "Dijon", "Guingamp", "Le Havre", "Le Mans", "Lens", "Lille", "Lorient", "Lyon",
        "Marseille", "Metz", "Monaco", "Montpellier", "Nantes", "Nice", "Nimes", "Paris FC",
        "Paris Saint-Germain", "Reims", "Rennes", "Saint-Etienne", "Strasbourg", "Toulouse",
        "Troyes",
    ],
}
CANONICAL_TEAMS = sorted({team for teams in CANONICAL.values() for team in teams})

# football-data.co.uk spellings that differ from the canonical name.
SOURCE_ALIASES = {
    "Ath Bilbao": "Athletic Bilbao",
    "Ath Madrid": "Atletico Madrid",
    "Augsburg": "FC Augsburg",
    "Betis": "Real Betis",
    "Bielefeld": "Arminia Bielefeld",
    "Bochum": "VfL Bochum",
    "Bournemouth": "AFC Bournemouth",
    "Brighton": "Brighton and Hove Albion",
    "Cardiff": "Cardiff City",
    "Celta": "Celta Vigo",
    "Coventry": "Coventry City",
    "Darmstadt": "Darmstadt 98",
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "Elversberg": "SV Elversberg",
    "Espanol": "Espanyol",
    "Freiburg": "SC Freiburg",
    "Hamburg": "Hamburger SV",
    "Hannover": "Hannover 96",
    "Heidenheim": "FC Heidenheim",
    "Hertha": "Hertha Berlin",
    "Hoffenheim": "TSG Hoffenheim",
    "Huddersfield": "Huddersfield Town",
    "Hull": "Hull City",
    "Inter": "Inter Milan",
    "Ipswich": "Ipswich Town",
    "La Coruna": "Deportivo La Coruna",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Leverkusen": "Bayer Leverkusen",
    "Luton": "Luton Town",
    "M'gladbach": "Borussia Monchengladbach",
    "Mainz": "Mainz 05",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Milan": "AC Milan",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Nott'm Forest": "Nottingham Forest",
    "Nurnberg": "FC Nurnberg",
    "Oviedo": "Real Oviedo",
    "Paderborn": "SC Paderborn",
    "Paris SG": "Paris Saint-Germain",
    "Roma": "AS Roma",
    "Santander": "Racing Santander",
    "Sociedad": "Real Sociedad",
    "Spal": "SPAL",
    "St Etienne": "Saint-Etienne",
    "Stuttgart": "VfB Stuttgart",
    "Tottenham": "Tottenham Hotspur",
    "Valladolid": "Real Valladolid",
    "Vallecano": "Rayo Vallecano",
    "Verona": "Hellas Verona",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolfsburg": "VfL Wolfsburg",
    "Wolves": "Wolverhampton Wanderers",
}

# football-data.org spellings that differ from the canonical name (unverified, see module docstring).
API_ALIASES = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton and Hove Albion",
    "Burnley FC": "Burnley",
    "Cardiff City FC": "Cardiff City",
    "Chelsea FC": "Chelsea",
    "Coventry City FC": "Coventry City",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Huddersfield Town AFC": "Huddersfield Town",
    "Hull City AFC": "Hull City",
    "Ipswich Town FC": "Ipswich Town",
    "Leeds United FC": "Leeds United",
    "Leicester City FC": "Leicester City",
    "Liverpool FC": "Liverpool",
    "Luton Town FC": "Luton Town",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Norwich City FC": "Norwich City",
    "Nottingham Forest FC": "Nottingham Forest",
    "Sheffield United FC": "Sheffield United",
    "Southampton FC": "Southampton",
    "Sunderland AFC": "Sunderland",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Watford FC": "Watford",
    "West Bromwich Albion FC": "West Bromwich Albion",
    "West Ham United FC": "West Ham United",
    "Wolverhampton Wanderers FC": "Wolverhampton Wanderers",
    "Athletic Club": "Athletic Bilbao",
    "Club Atlético de Madrid": "Atletico Madrid",
    "Deportivo Alavés": "Alaves",
    "UD Almería": "Almeria",
    "FC Barcelona": "Barcelona",
    "Cádiz CF": "Cadiz",
    "RC Celta de Vigo": "Celta Vigo",
    "RC Deportivo de La Coruña": "Deportivo La Coruna",
    "SD Eibar": "Eibar",
    "Elche CF": "Elche",
    "RCD Espanyol de Barcelona": "Espanyol",
    "Getafe CF": "Getafe",
    "Girona FC": "Girona",
    "Granada CF": "Granada",
    "SD Huesca": "Huesca",
    "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes",
    "Levante UD": "Levante",
    "Málaga CF": "Malaga",
    "RCD Mallorca": "Mallorca",
    "CA Osasuna": "Osasuna",
    "Real Racing Club de Santander": "Racing Santander",
    "Rayo Vallecano de Madrid": "Rayo Vallecano",
    "Real Betis Balompié": "Real Betis",
    "Real Madrid CF": "Real Madrid",
    "Real Sociedad de Fútbol": "Real Sociedad",
    "Real Valladolid CF": "Real Valladolid",
    "Sevilla FC": "Sevilla",
    "Valencia CF": "Valencia",
    "Villarreal CF": "Villarreal",
    "DSC Arminia Bielefeld": "Arminia Bielefeld",
    "Bayer 04 Leverkusen": "Bayer Leverkusen",
    "FC Bayern München": "Bayern Munich",
    "Borussia Mönchengladbach": "Borussia Monchengladbach",
    "SV Darmstadt 98": "Darmstadt 98",
    "1. FC Heidenheim 1846": "FC Heidenheim",
    "1. FC Köln": "FC Koln",
    "1. FC Nürnberg": "FC Nurnberg",
    "Fortuna Düsseldorf": "Fortuna Dusseldorf",
    "SpVgg Greuther Fürth": "Greuther Furth",
    "Hertha BSC": "Hertha Berlin",
    "1. FSV Mainz 05": "Mainz 05",
    "SC Paderborn 07": "SC Paderborn",
    "FC Schalke 04": "Schalke 04",
    "FC St. Pauli 1910": "St Pauli",
    "TSG 1899 Hoffenheim": "TSG Hoffenheim",
    "1. FC Union Berlin": "Union Berlin",
    "VfL Bochum 1848": "VfL Bochum",
    "SV Werder Bremen": "Werder Bremen",
    "Atalanta BC": "Atalanta",
    "Benevento Calcio": "Benevento",
    "Bologna FC 1909": "Bologna",
    "Brescia Calcio": "Brescia",
    "Cagliari Calcio": "Cagliari",
    "AC ChievoVerona": "Chievo",
    "Como 1907": "Como",
    "US Cremonese": "Cremonese",
    "FC Crotone": "Crotone",
    "Empoli FC": "Empoli",
    "ACF Fiorentina": "Fiorentina",
    "Frosinone Calcio": "Frosinone",
    "Genoa CFC": "Genoa",
    "Hellas Verona FC": "Hellas Verona",
    "FC Internazionale Milano": "Inter Milan",
    "Juventus FC": "Juventus",
    "SS Lazio": "Lazio",
    "US Lecce": "Lecce",
    "AC Monza": "Monza",
    "SSC Napoli": "Napoli",
    "Parma Calcio 1913": "Parma",
    "AC Pisa 1909": "Pisa",
    "US Salernitana 1919": "Salernitana",
    "UC Sampdoria": "Sampdoria",
    "US Sassuolo Calcio": "Sassuolo",
    "Spezia Calcio": "Spezia",
    "Torino FC": "Torino",
    "Udinese Calcio": "Udinese",
    "Venezia FC": "Venezia",
    "AC Ajaccio": "Ajaccio",
    "Amiens SC": "Amiens",
    "Angers SCO": "Angers",
    "AJ Auxerre": "Auxerre",
    "FC Girondins de Bordeaux": "Bordeaux",
    "Stade Brestois 29": "Brest",
    "SM Caen": "Caen",
    "Clermont Foot 63": "Clermont",
    "Dijon FCO": "Dijon",
    "En Avant Guingamp": "Guingamp",
    "Le Havre AC": "Le Havre",
    "Le Mans FC": "Le Mans",
    "Racing Club de Lens": "Lens",
    "Lille OSC": "Lille",
    "FC Lorient": "Lorient",
    "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille",
    "FC Metz": "Metz",
    "AS Monaco FC": "Monaco",
    "Montpellier HSC": "Montpellier",
    "FC Nantes": "Nantes",
    "OGC Nice": "Nice",
    "Nîmes Olympique": "Nimes",
    "Paris Saint-Germain FC": "Paris Saint-Germain",
    "Stade de Reims": "Reims",
    "Stade Rennais FC 1901": "Rennes",
    "AS Saint-Étienne": "Saint-Etienne",
    "Racing Club de Strasbourg Alsace": "Strasbourg",
    "Toulouse FC": "Toulouse",
    "ES Troyes AC": "Troyes",
}

# Words that carry no identity: dropped when building the normalised key.
_NOISE_WORDS = {
    "fc", "afc", "cf", "ac", "as", "ss", "ssc", "sc", "us", "rc", "rcd", "ud", "cd", "ca", "sd",
    "vfb", "vfl", "tsg", "sv", "fsv", "club", "de", "the",
}


class UnknownTeamError(KeyError):
    """Raised by canonical_team(strict=True) for a name that matches no known club."""


def normalise_key(name: str) -> str:
    """Lower-case, accent-free, punctuation-free key without club-type words or numbers."""
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower().replace("&", " and ")
    tokens = re.sub(r"[^a-z0-9]+", " ", text).split()
    kept = [t for t in tokens if t not in _NOISE_WORDS and not t.isdigit()]
    return " ".join(kept or tokens)


def _build_index() -> dict[str, str]:
    """normalised key -> canonical name. Raises if two different clubs would share a key."""
    index: dict[str, str] = {}
    pairs = [(team, team) for team in CANONICAL_TEAMS]
    pairs += list(SOURCE_ALIASES.items()) + list(API_ALIASES.items())
    for variant, canonical in pairs:
        if canonical not in CANONICAL_TEAMS:
            raise ValueError(f"alias {variant!r} points at unknown club {canonical!r}")
        key = normalise_key(variant)
        if index.setdefault(key, canonical) != canonical:
            raise ValueError(f"name clash: {variant!r} and {index[key]!r} both normalise to {key!r}")
    return index


_EXACT = {**SOURCE_ALIASES, **API_ALIASES}
_INDEX = _build_index()


def canonical_team(name: str, strict: bool = False) -> str:
    """Canonical club name for any known spelling (see module docstring)."""
    cleaned = " ".join(str(name).split())
    if cleaned in CANONICAL_TEAMS:
        return cleaned
    if cleaned in _EXACT:
        return _EXACT[cleaned]
    found = _INDEX.get(normalise_key(cleaned))
    if found is not None:
        return found
    if strict:
        raise UnknownTeamError(cleaned)
    return cleaned


def unmatched(names: Iterable[str]) -> list[str]:
    """Sorted distinct names that do not resolve to a known club."""
    return sorted({n for n in names if canonical_team(n) not in CANONICAL_TEAMS})


def _names_in_csv(path: Path) -> set[str]:
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    names: set[str] = set()
    for column in ("home_team", "away_team", "HomeTeam", "AwayTeam"):
        names.update(r[column] for r in rows if r.get(column))
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List team names that do not resolve to a known club.")
    parser.add_argument("--check", nargs="*", metavar="CSV",
                        help="CSV files with home_team/away_team columns (default: ai/training/data/*.csv)")
    args = parser.parse_args(argv)
    paths = [Path(p) for p in args.check] if args.check else sorted((Path(__file__).parent / "data").glob("*.csv"))
    if not paths:
        print("No CSV files to check.")
        return 0
    bad = 0
    for path in paths:
        missing = unmatched(_names_in_csv(path))
        print(f"{path.name}: {'all names resolved' if not missing else 'UNMATCHED ' + ', '.join(missing)}")
        bad += len(missing)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
