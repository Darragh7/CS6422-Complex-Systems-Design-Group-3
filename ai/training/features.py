"""Stage 7b: match features computed strictly from PRIOR matches (no leakage).

Every per-team statistic is shift(1)-ed before rolling, so match k only sees
matches that kicked off before it. Build features on the full chronological
dataset *before* time_split so test matches can use training-period history.
"""
import numpy as np
import pandas as pd

WINDOW = 5
DEFAULT_REST_DAYS = 7.0  # used when a team has no previous match in the data
H2H_NEUTRAL_PPG = 1.5  # midpoint of 0..3 points, used when teams haven't met
REST_DAYS_CAP = 14.0  # summer/winter breaks otherwise dominate the scale
ELO_START, ELO_K, ELO_HOME_ADV = 1500.0, 20.0, 60.0

TEAM_STATS = ["ppg", "gf", "ga", "rest_days"]
FEATURE_COLUMNS = (
    [f"{side}_{s}" for side in ("home", "away") for s in TEAM_STATS]
    + ["home_no_history", "away_no_history", "h2h_home_ppg", "h2h_n", "elo_diff"]
)


def _points(gf: pd.Series, ga: pd.Series) -> pd.Series:
    return np.select([gf > ga, gf == ga], [3, 1], 0).astype(float)


def _team_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) with that team's goals for/against and points."""
    home = pd.DataFrame({"match": df.index, "date": df["date"], "team": df["home_team"],
                         "side": "home", "gf": df["home_score"], "ga": df["away_score"]})
    away = pd.DataFrame({"match": df.index, "date": df["date"], "team": df["away_team"],
                         "side": "away", "gf": df["away_score"], "ga": df["home_score"]})
    long = pd.concat([home, away], ignore_index=True).sort_values(["date", "match"], kind="stable")
    long["pts"] = _points(long["gf"], long["ga"])
    return long


def team_form(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Per (match, side): rolling ppg/gf/ga over previous `window` matches + rest days."""
    long = _team_long(df)
    g = long.groupby("team", sort=False)
    for col, out in (("pts", "ppg"), ("gf", "gf"), ("ga", "ga")):
        # shift(1) BEFORE rolling: exclude the current match.
        long[out] = g[col].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    long["rest_days"] = g["date"].diff().dt.total_seconds() / 86400
    long["no_history"] = long["ppg"].isna().astype(int)
    return long


def head_to_head(df: pd.DataFrame) -> pd.DataFrame:
    """Home team's points per game in previous meetings (either venue) + count."""
    ordered = df.sort_values("date", kind="stable")
    history: dict[frozenset, list[tuple[str, float]]] = {}
    ppg, n = {}, {}
    for idx, row in ordered.iterrows():
        key = frozenset((row["home_team"], row["away_team"]))
        past = history.setdefault(key, [])
        mine = [p for team, p in past if team == row["home_team"]]
        n[idx] = len(mine)
        ppg[idx] = float(np.mean(mine)) if mine else H2H_NEUTRAL_PPG
        # record AFTER computing features for this match
        hp = _points(pd.Series([row["home_score"]]), pd.Series([row["away_score"]]))[0]
        past += [(row["home_team"], hp), (row["away_team"], {3: 0.0, 1: 1.0, 0: 3.0}[int(hp)])]
    return pd.DataFrame({"h2h_home_ppg": pd.Series(ppg), "h2h_n": pd.Series(n)})


def elo_diff(df: pd.DataFrame, k: float = ELO_K, home_adv: float = ELO_HOME_ADV) -> pd.Series:
    """Pre-match Elo difference (home - away), updated AFTER each match.

    Ratings carry across seasons (no regression to the mean yet; TODO maybe).
    New teams (e.g. promoted) start at ELO_START.
    """
    ratings: dict[str, float] = {}
    diff = {}
    for idx, row in df.sort_values("date", kind="stable").iterrows():
        h = ratings.get(row["home_team"], ELO_START)
        a = ratings.get(row["away_team"], ELO_START)
        diff[idx] = h - a
        expected_home = 1 / (1 + 10 ** ((a - h - home_adv) / 400))
        actual = 1.0 if row["home_score"] > row["away_score"] else 0.5 if row["home_score"] == row["away_score"] else 0.0
        ratings[row["home_team"]] = h + k * (actual - expected_home)
        ratings[row["away_team"]] = a - k * (actual - expected_home)
    return pd.Series(diff, name="elo_diff")


def build_features(df: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Feature table indexed like df. Expects clean() output (finished, parsed dates).

    Early-season gaps: rolling stats -> 0 with *_no_history=1; rest days -> 7
    (and capped at REST_DAYS_CAP).
    """
    # TODO: when load_static_csv is implemented, concatenate the static history
    # with API matches (dedupe by id) BEFORE calling this, so form/h2h use it.
    long = team_form(df, window)
    out = pd.DataFrame(index=df.index)
    for side in ("home", "away"):
        part = long[long["side"] == side].set_index("match")
        for s in TEAM_STATS:
            out[f"{side}_{s}"] = part[s]
        out[f"{side}_no_history"] = part["no_history"]
    for side in ("home", "away"):
        out[[f"{side}_ppg", f"{side}_gf", f"{side}_ga"]] = out[[f"{side}_ppg", f"{side}_gf", f"{side}_ga"]].fillna(0.0)
        out[f"{side}_rest_days"] = out[f"{side}_rest_days"].fillna(DEFAULT_REST_DAYS).clip(upper=REST_DAYS_CAP)
    out = out.join(head_to_head(df)).join(elo_diff(df))
    return out[FEATURE_COLUMNS]


# ---------------------------------------------------------------------------
# Extended feature groups (ablation candidates). FEATURE_COLUMNS above stays
# the original 13 = BASELINE. Everything below is prior-only too: a match only
# sees matches with a strictly earlier kickoff (same-kickoff games excluded).
# ---------------------------------------------------------------------------
from scipy.optimize import minimize  # noqa: E402 (sklearn dependency, no new deps)

BASELINE = list(FEATURE_COLUMNS)
FORM_COLUMNS = [f"{side}_{s}" for side in ("home", "away") for s in ("ppg", "gf", "ga")] + [
    "home_rest_days", "away_rest_days"]  # the 8 rolling form columns
WEAK_COLUMNS = ["home_rest_days", "away_rest_days", "home_no_history", "away_no_history", "h2h_n"]
EWM_SPAN = 10
VENUE_WINDOW, VENUE_MIN_GAMES, VENUE_PRIOR_PPG = 5, 3, 1.5
DC_XI = 0.0065 / 3.5  # Dixon-Coles time decay: 0.0065 per half-week, in per-day units
DC_RIDGE, DC_MAX_GOALS, DC_MIN_MATCHES = 1.0, 7, 30

FEATURE_GROUPS = {
    "elo_close": ["abs_elo_diff", "elo_balance"],
    "ewm_form": ["ewm_gd_diff"],
    "exp_goals": ["exp_home_goals", "exp_away_goals", "exp_total_goals"],
    "venue_form": ["home_venue_ppg", "away_venue_ppg", "venue_low_history"],
    "dixon_coles": ["dc_p_draw", "dc_p_home", "dc_exp_total"],
    "table_gap": ["ppg_gap", "pos_gap", "abs_ppg_gap", "table_mw1"],
}


def _strictly_before(values: pd.Series, dates: pd.Series) -> pd.Series:
    """Value of a running total as it stood before the first row at each kickoff time."""
    return values.shift(1).groupby(dates.to_numpy()).transform("first")


def _season(dates: pd.Series) -> pd.Series:
    return dates.dt.year - (dates.dt.month < 7)  # season start year (Aug..May)


def elo_closeness(elo: pd.Series, home_adv: float = ELO_HOME_ADV) -> pd.DataFrame:
    """|elo_diff| and (We - 0.5)^2, We = expected home score incl. home advantage."""
    we = 1 / (1 + 10 ** (-(elo + home_adv) / 400))
    return pd.DataFrame({"abs_elo_diff": elo.abs(), "elo_balance": (we - 0.5) ** 2})


def ewm_and_venue_form(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(home, away) per-match frames with shifted ewm gf/ga/gd and venue-only ppg."""
    long = _team_long(df)
    long["gd"] = long["gf"] - long["ga"]
    g = long.groupby("team", sort=False)
    for col in ("gf", "ga", "gd"):
        long[f"ewm_{col}"] = g[col].transform(lambda s: s.shift(1).ewm(span=EWM_SPAN, min_periods=1).mean())
    # venue form: previous VENUE_WINDOW home (or away) games only, shrunk to a prior if < VENUE_MIN_GAMES
    gv = long.groupby(["team", "side"], sort=False)["pts"]
    pts_sum = gv.transform(lambda s: s.shift(1).rolling(VENUE_WINDOW, min_periods=1).sum()).fillna(0.0)
    n = gv.transform(lambda s: s.shift(1).rolling(VENUE_WINDOW, min_periods=1).count()).fillna(0.0)
    k = (VENUE_MIN_GAMES - n).clip(lower=0)
    long["venue_ppg"] = (pts_sum + k * VENUE_PRIOR_PPG) / (n + k)
    long["venue_low"] = (n < VENUE_MIN_GAMES).astype(int)
    h = long[long["side"] == "home"].set_index("match").reindex(df.index)
    a = long[long["side"] == "away"].set_index("match").reindex(df.index)
    return h, a


def expected_goals(df: pd.DataFrame, h: pd.DataFrame, a: pd.DataFrame) -> pd.DataFrame:
    """League prior average x attack strength x opponent defence weakness (all prior-only)."""
    ordered = df.sort_values("date", kind="stable")
    n = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index)
    lg = {}
    for side, default in (("home", 1.5), ("away", 1.2)):
        mean = ordered[f"{side}_score"].cumsum() / n
        lg[side] = _strictly_before(mean, ordered["date"]).reindex(df.index).fillna(default)
    lg_team = (lg["home"] + lg["away"]) / 2
    out = pd.DataFrame(index=df.index)
    out["exp_home_goals"] = (lg["home"] * (h["ewm_gf"] / lg_team) * (a["ewm_ga"] / lg_team)).fillna(lg["home"])
    out["exp_away_goals"] = (lg["away"] * (a["ewm_gf"] / lg_team) * (h["ewm_ga"] / lg_team)).fillna(lg["away"])
    out["exp_total_goals"] = out["exp_home_goals"] + out["exp_away_goals"]
    return out


def dc_score_matrix(lam: float, mu: float, rho: float, max_goals: int = DC_MAX_GOALS) -> np.ndarray:
    """Dixon-Coles P(home=x, away=y) for x, y <= max_goals, renormalised to sum to 1."""
    from scipy.stats import poisson
    k = np.arange(max_goals + 1)
    p = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))
    p[0, 0] *= 1 - lam * mu * rho
    p[1, 0] *= 1 + mu * rho
    p[0, 1] *= 1 + lam * rho
    p[1, 1] *= 1 - rho
    p = np.clip(p, 0, None)
    return p / p.sum()


def _dc_fit(hi, ai, hg, ag, w, n_teams, x0=None, ridge=DC_RIDGE):
    """Weighted DC likelihood. params: [c, home, rho, att(n), def(n)]; ridge on att/def."""
    def unpack(p):
        return p[0], p[1], p[2], p[3:3 + n_teams], p[3 + n_teams:]

    def nll(p):
        c, home, rho, att, dfn = unpack(p)
        lam = np.exp(c + home + att[hi] + dfn[ai])
        mu = np.exp(c + att[ai] + dfn[hi])
        tau = np.ones_like(lam)
        m00, m10, m01, m11 = (hg == 0) & (ag == 0), (hg == 1) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 1)
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m11] = 1 - rho
        ll = np.log(np.clip(tau, 1e-10, None)) + hg * np.log(lam) - lam + ag * np.log(mu) - mu
        return -(w * ll).sum() + ridge * ((att ** 2).sum() + (dfn ** 2).sum())

    x0 = np.r_[np.log(1.3), 0.2, -0.05, np.zeros(2 * n_teams)] if x0 is None else x0
    bounds = [(None, None), (None, None), (-0.2, 0.2)] + [(None, None)] * (2 * n_teams)
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds)
    return unpack(res.x), res.x


def dixon_coles(df: pd.DataFrame, xi: float = DC_XI, ridge: float = DC_RIDGE) -> pd.DataFrame:
    """Pre-match Dixon-Coles (1997) probabilities, refit once per calendar week.

    Matches in week W use a fit on all matches that kicked off before W starts
    (so never their own or same-week results), time-weighted exp(-xi * age_days).
    Unseen teams sit at league average (att = def = 0) through the ridge penalty.
    Before DC_MIN_MATCHES of history, falls back to independent Poisson(1.5, 1.2).
    """
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    tix = {t: i for i, t in enumerate(teams)}
    d = df.sort_values("date", kind="stable")
    hi_all, ai_all = d["home_team"].map(tix).to_numpy(), d["away_team"].map(tix).to_numpy()
    hg_all, ag_all = d["home_score"].to_numpy(float), d["away_score"].to_numpy(float)
    dates = d["date"]
    week = (dates.dt.tz_convert(None).dt.normalize() - pd.to_timedelta(dates.dt.weekday, unit="D"))
    rows, x0 = {}, None
    for wk in week.unique():
        past = (dates < pd.Timestamp(wk, tz="UTC")).to_numpy()
        cur = (week == wk).to_numpy()
        if past.sum() >= DC_MIN_MATCHES:
            age = (pd.Timestamp(wk, tz="UTC") - dates[past]).dt.total_seconds().to_numpy() / 86400
            (c, home, rho, att, dfn), x0 = _dc_fit(hi_all[past], ai_all[past], hg_all[past], ag_all[past],
                                                   np.exp(-xi * age), len(teams), x0, ridge)
        else:
            c, home, rho, att, dfn = np.log(1.35), np.log(1.5 / 1.35), 0.0, np.zeros(len(teams)), np.zeros(len(teams))
        for idx, h, a in zip(d.index[cur], hi_all[cur], ai_all[cur]):
            lam, mu = np.exp(c + home + att[h] + dfn[a]), np.exp(c + att[a] + dfn[h])
            p = dc_score_matrix(lam, mu, rho)
            rows[idx] = (np.tril(p, -1).sum(), np.trace(p), np.triu(p, 1).sum(), lam + mu)
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["dc_p_home", "dc_p_draw", "dc_p_away", "dc_exp_total"])
    return out.reindex(df.index)


def table_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-match league table of the same season from matches with an earlier kickoff.

    ppg_gap = home season ppg - away season ppg, pos_gap = away_pos - home_pos
    (positive = home higher), abs_ppg_gap. Ranked by points, then GD, then GF.
    table_mw1 = 1 (gaps 0) when either team has no season game yet.
    """
    d = df.assign(season=_season(df["date"])).sort_values("date", kind="stable")
    season_teams = {s: sorted(set(g["home_team"]) | set(g["away_team"])) for s, g in d.groupby("season")}
    tables: dict[int, pd.DataFrame] = {}
    rows = {}
    for (season, _), grp in d.groupby(["season", "date"], sort=True):
        tab = tables.setdefault(season, pd.DataFrame(0.0, index=season_teams[season], columns=["pts", "gd", "gf", "n"]))
        order = tab.sort_values(["pts", "gd", "gf"], ascending=False).index
        pos = pd.Series(np.arange(1, len(order) + 1), index=order)
        for idx, r in grp.iterrows():  # features first (from the table as it stood) ...
            h, a = tab.loc[r["home_team"]], tab.loc[r["away_team"]]
            if h["n"] == 0 or a["n"] == 0:
                rows[idx] = (0.0, 0.0, 0.0, 1)
            else:
                gap = h["pts"] / h["n"] - a["pts"] / a["n"]
                rows[idx] = (gap, float(pos[r["away_team"]] - pos[r["home_team"]]), abs(gap), 0)
        for _, r in grp.iterrows():  # ... then update with this kickoff's results
            hp = 3 if r["home_score"] > r["away_score"] else 1 if r["home_score"] == r["away_score"] else 0
            ap = {3: 0, 1: 1, 0: 3}[hp]
            for team, gf, ga, p in ((r["home_team"], r["home_score"], r["away_score"], hp),
                                    (r["away_team"], r["away_score"], r["home_score"], ap)):
                tab.loc[team] += [p, gf - ga, gf, 1]
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["ppg_gap", "pos_gap", "abs_ppg_gap", "table_mw1"])
    out["table_mw1"] = out["table_mw1"].astype(int)
    return out.reindex(df.index)


def build_extended_features(df: pd.DataFrame, base: pd.DataFrame | None = None, dc: bool = True) -> pd.DataFrame:
    """BASELINE 13 plus every column in FEATURE_GROUPS, indexed like df.

    dc=False skips the (slow, ~weekly refit) Dixon-Coles group.
    """
    base = build_features(df) if base is None else base
    h, a = ewm_and_venue_form(df)
    out = base.join(elo_closeness(base["elo_diff"]))
    out["ewm_gd_diff"] = (h["ewm_gd"] - a["ewm_gd"]).fillna(0.0)
    out = out.join(expected_goals(df, h, a))
    out["home_venue_ppg"], out["away_venue_ppg"] = h["venue_ppg"], a["venue_ppg"]
    out["venue_low_history"] = ((h["venue_low"] + a["venue_low"]) > 0).astype(int)
    if dc:
        out = out.join(dixon_coles(df))
    return out.join(table_gap(df))


def feature_set(*groups: str, base=BASELINE, drop=(), replace_form: bool = False) -> list[str]:
    """Column list: base minus `drop` (minus FORM_COLUMNS if replace_form) plus groups."""
    cols = [c for c in base if c not in set(drop) and not (replace_form and c in FORM_COLUMNS)]
    for g in groups:
        cols += [c for c in FEATURE_GROUPS[g] if c not in cols]
    return cols

# Chosen by train-only time-ordered CV ablation (see eda.ipynb "Feature Ablation").
SELECTED_FEATURES = ["elo_diff", "ewm_gd_diff", "dc_p_home", "dc_p_draw"]
