# Scoreline model card (scoreline_v1)

## 1. Purpose

Scoreline estimates the probability of a **home win, draw and away win** for matches in five leagues: Premier League, La Liga, Bundesliga, Serie A and Ligue 1.

- **Intended use:** information and entertainment on the Scoreline website.
- **Not for:** betting or gambling decisions. The probabilities are estimates, not guarantees, and the site shows a "not betting advice" disclaimer.
- **Output:** three probabilities in the order `[home, draw, away]` that sum to 1, plus the model version (the exact JSON format is still to be agreed with the backend and frontend).

## 2. The model

The final prediction is the **simple average of four models**:

| Member | What it does | Key settings |
|---|---|---|
| Elo | A strength rating per team, updated after every match | K=20, home advantage 60 points, ratings pulled 25% toward average each new season |
| Poisson | Expected goals from team attack and defence strength plus home advantage, re-fitted weekly on the last 3 years | half-life 365 days, penalty 0.001 |
| Logistic regression | Multinomial classifier on the match features | C=0.003 |
| XGBoost | Gradient-boosted trees on the match features | 171 trees, depth 3 |

- **Version:** scoreline_v1, trained on 14,535 matches (seasons 2018/19 to 2026/27 so far). Exact settings and metrics are in `ai/ml_pipeline/models/scoreline_v1/metadata.json`.
- Elo and Poisson are recomputed from the match history at prediction time. Logistic regression and XGBoost are saved as `.joblib` files by `export_model.py`; these files are git-ignored, so regenerate them or get them from the ML team.

## 3. Data

- **Results, shots and odds:** football-data.co.uk (CSV files), 5 leagues, 2018/19 onward.
- **Upcoming fixtures:** football-data.org API (v4). Fixtures ingestion is written but not yet tested with the team's API key.
- **Bookmaker odds** (mostly Pinnacle closing prices, average margin 2.9%) are used **only as a benchmark**, never as a model input.
- **Cleaning:** one row per match, dates parsed into a standard format, incomplete or inconsistent rows dropped (none in the final data), team names kept as in the source (`team_mapping.csv` is a reference list for matching names across sources and is not yet applied in the pipeline).
- **Overall result split:** 43.3% home wins, 25.3% draws, 31.4% away wins. Average goals per match: 1.54 home, 1.26 away.
- **Licensing:** football-data.org is covered in `docs/security/licensing-and-disclaimers.md`. The football-data.co.uk historical data and the bookmaker odds it provides are listed there as not yet reviewed, so their terms still need to be checked and recorded before any public or commercial use.

## 4. Features (45)

Every feature uses only matches played before the one being predicted (checked by an automated test).

- **Form:** points, goals scored and goals conceded over the last 5 and 10 matches
- **Shots:** shots and shots on target, for and against, over the last 5 matches
- **Home/away split:** recent results at home for the home team and away for the away team
- **Schedule:** days of rest (league matches only) and amount of history
- **Strength:** Elo ratings and the Elo gap
- **League:** one column per league
- Differences between the two teams for the main form features

## 5. How it was evaluated

- **Time-based splits, never random:** training on seasons up to 2023/24, validation on 2024/25 (model comparison), and a final test on 2025/26 that was used **once**, after the model was fixed.
- Settings for logistic regression and XGBoost were chosen on the last training season, not on the validation or test seasons.
- **Metrics:** log loss and Brier score (lower is better), ranked probability score (RPS, lower is better) and accuracy (the share of matches where the most likely outcome happened).

## 6. Results

**Validation season 2024/25 (1,752 matches)**

| Model | Log loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Historical frequencies (benchmark) | 1.0774 | 0.6526 | 0.2327 | 42.0% |
| Elo | 0.9880 | 0.5894 | 0.2014 | 53.2% |
| Poisson | 0.9848 | 0.5867 | 0.2008 | 52.8% |
| Logistic regression | 0.9869 | 0.5882 | 0.2002 | 53.5% |
| Random forest | 0.9877 | 0.5887 | 0.2012 | 52.7% |
| XGBoost | 0.9867 | 0.5877 | 0.2007 | 53.9% |
| **Ensemble (final model)** | **0.9819** | **0.5851** | **0.1996** | **53.4%** |
| Bookmakers (margin removed) | 0.9606 | 0.5706 | 0.1935 | 54.7% |

**Final test season 2025/26 (1,752 matches, used once)**

| Model | Log loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Elo | 0.9971 | 0.5948 | 0.2036 | 51.1% |
| Poisson | 0.9981 | 0.5952 | 0.2035 | 51.8% |
| Logistic regression | 0.9916 | 0.5910 | 0.2018 | 52.2% |
| XGBoost | 0.9964 | 0.5941 | 0.2031 | 51.8% |
| **Ensemble (final model)** | **0.9922** | **0.5914** | **0.2020** | **51.4%** |
| Bookmakers (margin removed) | 0.9783 | 0.5826 | 0.1978 | 53.4% |

**What the results mean**

- Every model clearly beats the benchmark that ignores team strength (log loss 1.077, accuracy 42%).
- **The ensemble beat Elo by a small margin on both seasons** (validation: -0.0061, 95% interval [-0.0104, -0.0018]; final test: -0.0049, [-0.0085, -0.0014]). The individual models were statistically tied with Elo on the validation season, so adding complexity gave little by itself.
- **Bookmaker closing odds were better than the ensemble on both seasons** (by 0.021 log loss on validation and 0.014 on the final test). The odds include information we do not have, such as injuries, lineups and sharp bettors' knowledge.
- Scores are lower on the test season than on validation for every model, including the bookmakers, so 2025/26 was simply harder to predict. One season moves log loss by roughly 0.01 to 0.02.

## 7. Calibration

On the validation season the ensemble's predicted probabilities were close to observed frequencies: average gaps of 2.3 points (home win), 3.4 (draw) and 2.2 (away win), which is within what we would expect from noise with about 220 matches per bin. Home wins were slightly over-predicted and strong away wins slightly under-predicted, consistent with 2024/25 having fewer home wins (42.0%) than the training average (43.4%). No recalibration was applied.

## 8. Limitations

- **No lineup, injury, transfer or expected-goals data**, which is the main reason bookmakers do better.
- **Football is highly random.** Expect about 50 to 55% accuracy. Draws are about 25% of matches, but the model almost never picks a draw as the single most likely result, so accuracy cannot reward draw predictions.
- **League matches only:** cup and European fixtures are not in the data, so rest days understate fatigue for teams with extra games.
- **Newly promoted teams** have little history and their predictions are less reliable.
- **Training data includes the 2019/20 and 2020/21 seasons**, when many matches were played without fans, so home advantage was lower.
- **Bundesliga was the hardest league** to predict on the validation season. Per-league results rest on only about 300 to 380 matches each.
- The model describes the past and can drift as teams, rules and styles change.

## 9. How to reproduce

From `ai/ml_pipeline`, with the Python packages in `requirements.txt` installed:

```
python ingest.py historical --start 2018 --end 2026
python clean.py
python elo_baseline.py
python poisson_baseline.py
python features.py
python train_models.py
python odds.py
python evaluate.py
python final_test.py --confirm      # one time only
python export_model.py --version v1
```

Raw and cleaned data are not stored in Git. Re-run the scripts to regenerate them.

## 10. Monitoring and updates

- After matches finish, predictions are compared with results (ML-16): weekly log loss and accuracy.
- **Retraining trigger:** proposed: weekly log loss worse than the Elo baseline for four weeks in a row (to be agreed with the team).
- New versions get a new name (scoreline_v2, ...) and are compared on a fresh, untouched period before replacing this one.

## 11. Owners

ML: Sriram Krishnamurthi (add teammates if applicable). Questions: via the team chat or a GitHub issue.
