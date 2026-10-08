import asyncio
from datetime import date

import httpx

from app.config.settings import settings


# How many times we try a request before giving up
MAX_ATTEMPTS = 3

# Never wait longer than this between retries, even if the API asks us to
MAX_RETRY_WAIT_SECONDS = 10.0


class FootballAPIError(Exception):
    """Raised when we can't get data from football-data.org.

    status_code is the HTTP status returned by the provider, or None if we
    never got a response (missing key, timeout, connection error).
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class MissingAPIKeyError(FootballAPIError):
    pass


class FootballAPI:
    """Small async client for the football-data.org v4 API."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep=asyncio.sleep,
    ):
        self.api_key = api_key if api_key is not None else settings.football_api_key
        self.base_url = base_url or settings.football_api_base_url
        self.timeout = timeout or settings.football_api_timeout
        # transport and sleep can be swapped out in tests (no network, no waiting)
        self.transport = transport
        self.sleep = sleep

    async def get_matches(
        self,
        competition: str = "PL",
        date_from: date | None = None,
        date_to: date | None = None,
        status: str | None = None,
    ) -> list[dict]:
        params = {}
        if date_from:
            params["dateFrom"] = date_from.isoformat()
        if date_to:
            params["dateTo"] = date_to.isoformat()
        if status:
            params["status"] = status.upper()

        data = await self._get(f"/competitions/{competition.upper()}/matches", params)

        return [normalize_match(match) for match in data.get("matches", [])]

    async def _get(self, path: str, params: dict) -> dict:
        if not self.api_key:
            raise MissingAPIKeyError(
                "Football API key is not configured. Set FOOTBALL_API_KEY in your .env file."
            )

        headers = {"X-Auth-Token": self.api_key}
        last_error = None

        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self.transport,
        ) as client:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                wait = backoff_seconds(attempt)

                try:
                    response = await client.get(path, params=params)
                except httpx.TimeoutException:
                    last_error = FootballAPIError("Football API timed out")
                except httpx.TransportError:
                    last_error = FootballAPIError("Could not connect to the football API")
                else:
                    if response.status_code == 429:
                        last_error = FootballAPIError("Football API rate limit reached", 429)
                        wait = retry_after_seconds(response, default=wait)
                    elif response.status_code >= 500:
                        last_error = FootballAPIError(
                            "Football API is unavailable", response.status_code
                        )
                    elif response.status_code >= 400:
                        # 400/403/404 etc. won't fix themselves, so don't retry
                        raise FootballAPIError(
                            provider_message(response), response.status_code
                        )
                    else:
                        return response.json()

                if attempt < MAX_ATTEMPTS:
                    await self.sleep(wait)

        raise last_error


def backoff_seconds(attempt: int) -> float:
    # 0.5s, 1s, 2s, ...
    return 0.5 * 2 ** (attempt - 1)


def retry_after_seconds(response: httpx.Response, default: float) -> float:
    try:
        seconds = float(response.headers.get("Retry-After", default))
    except ValueError:
        seconds = default
    return max(0.0, min(seconds, MAX_RETRY_WAIT_SECONDS))


def provider_message(response: httpx.Response) -> str:
    try:
        message = response.json().get("message")
    except ValueError:
        message = None
    return message or f"Football API returned HTTP {response.status_code}"


def normalize_match(match: dict) -> dict:
    """Turn a football-data.org match into the shape our API returns."""
    home_team = match.get("homeTeam") or {}
    away_team = match.get("awayTeam") or {}
    full_time = (match.get("score") or {}).get("fullTime") or {}

    return {
        "id": match["id"],
        "home_team": home_team.get("name") or "TBD",
        "away_team": away_team.get("name") or "TBD",
        "date": match.get("utcDate"),
        "status": match.get("status"),
        "home_score": full_time.get("home"),
        "away_score": full_time.get("away"),
    }
