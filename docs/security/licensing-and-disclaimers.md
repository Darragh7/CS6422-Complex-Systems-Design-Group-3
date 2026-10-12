# Data licensing and prediction disclaimer

Last checked: 2026-10-12

## Data sources and terms review

| Source | Use found in this repository | Public terms checked | Attribution and restrictions |
|---|---|---|---|
| [football-data.org](https://www.football-data.org/) | The backend calls the v4 API for competition matches and normalizes team names, kickoff dates, statuses, and full-time scores in `backend/app/services/football_api.py`. The configured default competition is the Premier League (`PL`). | Reviewed the provider's [API Quickstart](https://www.football-data.org/documentation/quickstart), [API Reference](https://www.football-data.org/documentation/api), [Pricing](https://www.football-data.org/pricing), [Football Coverage](https://www.football-data.org/coverage), and [About](https://www.football-data.org/about) pages on 2026-10-12. The guessed `/terms` and `/terms-of-use` URLs returned 404, and no standalone public terms or data licence was located from the site's navigation. | Pricing lists the Free plan as limited to 12 competitions and 10 calls/minute, with delayed scores and schedules; check current coverage and the account's actual plan before use. The API Reference asks clients to make smart, non-excessive requests and not poll too frequently. No explicit attribution wording or general permission to redistribute API data was found on the public pages reviewed. |
| Other datasets, feeds, or model artefacts | None are imported or called by the current application. `backend/app/services/prediction_service.py` returns temporary fixed probabilities; no trained model or training dataset is used. | Not applicable to the current implementation. Recheck and record terms before adding any source or artefact. | No additional source attribution is currently identified. |

### Required follow-up before public or commercial use

- Treat football-data.org's redistribution, caching, commercial-use, and attribution requirements as **unconfirmed** until checked against the account-specific terms or confirmed by the provider. Public plan/pricing information is not a substitute for a licence.
- Do not republish or redistribute raw or substantial portions of provider data until the applicable terms are confirmed.
- Check the selected plan's competition coverage, freshness, and request limits against the app's actual use. The public Free plan page currently lists 10 calls/minute; limits and availability may change.
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
