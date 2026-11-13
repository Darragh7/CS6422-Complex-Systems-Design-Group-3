# Data licensing and prediction disclaimer

Last checked: 2026-10-12 (football-data.org review). ML pipeline sources added 2026-11-11, terms not yet reviewed.

## Data sources and terms review

| Source | Use found in this repository | Public terms checked | Attribution and restrictions |
|---|---|---|---|
| [football-data.org](https://www.football-data.org/) | The backend calls the v4 API for competition matches and normalizes team names, kickoff dates, statuses, and full-time scores in `backend/app/services/football_api.py`. The configured default competition is the Premier League (`PL`). | Reviewed the provider's [API Quickstart](https://www.football-data.org/documentation/quickstart), [API Reference](https://www.football-data.org/documentation/api), [Pricing](https://www.football-data.org/pricing), [Football Coverage](https://www.football-data.org/coverage), and [About](https://www.football-data.org/about) pages on 2026-10-12. The guessed `/terms` and `/terms-of-use` URLs returned 404, and no standalone public terms or data licence was located from the site's navigation. | Pricing lists the Free plan as limited to 12 competitions and 10 calls/minute, with delayed scores and schedules; check current coverage and the account's actual plan before use. The API Reference asks clients to make smart, non-excessive requests and not poll too frequently. No explicit attribution wording or general permission to redistribute API data was found on the public pages reviewed. |
| [football-data.co.uk](https://www.football-data.co.uk/) (historical results CSVs) | The ML pipeline downloads season CSV files (results, shots and bookmaker odds for the top five leagues) in `ai/ml_pipeline/ingest.py` and uses them to train and evaluate models. The raw files are not committed to the repository (the `ai/ml_pipeline/data/` folder is git-ignored). A team-name reference list derived from them is committed as `ai/ml_pipeline/team_mapping.csv`. | **Not yet reviewed.** The site's terms and notes pages still need to be read and the outcome recorded here. | Redistribution, caching, commercial-use and attribution requirements are **unconfirmed**. Do not republish the raw data. |
| Bookmaker odds (columns in the football-data.co.uk files) | Used only inside `ai/ml_pipeline/odds.py` and `evaluate.py` as a benchmark to compare model accuracy. Odds are not model inputs, are not stored in the repository and are not shown on the site. Only aggregate scores are reported, in `docs/model_card.md`. | **Not yet reviewed** (same source as above). | Do not display or redistribute odds. Confirm that reporting aggregate comparison scores is acceptable. |
| Trained model artefacts (`scoreline_v1`) | Trained by the team on the data above using `ai/ml_pipeline/export_model.py`. The `.joblib` files are git-ignored; only `metadata.json` and a README are committed under `ai/ml_pipeline/models/`. The backend does not load them yet. | Depends on the terms of the training data above. | Confirm that models trained on this data may be used on the public site before deployment. |
| Other datasets, feeds, or model artefacts (backend and website) | None are imported or called by the backend or website. `backend/app/services/prediction_service.py` returns temporary fixed probabilities; no trained model or training dataset is used. | Not applicable to the current implementation. Recheck and record terms before adding any source or artefact. | No additional source attribution is currently identified. |

### Required follow-up before public or commercial use

- Treat football-data.org's redistribution, caching, commercial-use, and attribution requirements as **unconfirmed** until checked against the account-specific terms or confirmed by the provider. Public plan/pricing information is not a substitute for a licence.
- Do not republish or redistribute raw or substantial portions of provider data until the applicable terms are confirmed.
- Check the selected plan's competition coverage, freshness, and request limits against the app's actual use. The public Free plan page currently lists 10 calls/minute; limits and availability may change.
- Review and record the terms of football-data.co.uk (results, shots and odds data used by the ML pipeline) and decide whether models trained on it may be used on the public site. Owner: Security, with the ML team. Until then, treat that data and the trained models as for development use only.
- Keep the API token secret. Continue using efficient requests and avoid unnecessary polling, as advised by the API Reference.
- No mandatory attribution text was discoverable in the public pages reviewed. As a transparent courtesy, display this wherever provider match data is shown, subject to provider confirmation:

  > Football data provided by [football-data.org](https://www.football-data.org/).

  Do not represent this courtesy attribution as a confirmed licence requirement.

## Disclaimer text for the site

> These predictions are for information and education only. They are estimates, may be wrong or out of date, and are not betting or financial advice. Do not use them to decide whether to place a bet or make a financial decision. No result or profit is guaranteed. Make your own decisions.

## Disclaimer text for project documentation

> Match predictions and related information are provided for information and education only. Predictions are estimates based on available data and may be wrong, incomplete, or out of date. They are not betting or financial advice, and are not a recommendation to place a bet. Do not rely on them when betting or making financial decisions. No match result or profit is guaranteed. You are responsible for your decisions and for following the laws that apply to you.

## Approval status

The disclaimer text above is a draft for team review. It has not yet been agreed by the team.
