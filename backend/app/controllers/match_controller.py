from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.match_schema import MatchListResponse
from app.services.football_api import (
    FootballAPI,
    FootballAPIError,
    MissingAPIKeyError,
)


router = APIRouter(
    prefix="/matches",
    tags=["Matches"]
)


def get_football_api() -> FootballAPI:
    # Separate function so tests can swap in a fake client
    return FootballAPI()


@router.get("/", response_model=MatchListResponse)
async def get_matches(
    competition: str = Query("PL", description="Competition code, e.g. PL, CL, BL1"),
    date_from: date | None = Query(None, description="YYYY-MM-DD"),
    date_to: date | None = Query(None, description="YYYY-MM-DD"),
    status: str | None = Query(None, description="e.g. SCHEDULED, FINISHED"),
    football_api: FootballAPI = Depends(get_football_api),
):
    if date_from and date_to and date_from > date_to:
        raise HTTPException(
            status_code=400,
            detail="date_from must be on or before date_to"
        )

    try:
        matches = await football_api.get_matches(
            competition=competition,
            date_from=date_from,
            date_to=date_to,
            status=status,
        )
    except FootballAPIError as error:
        raise to_http_exception(error, competition)

    return {
        "matches": matches
    }


def to_http_exception(error: FootballAPIError, competition: str) -> HTTPException:
    if isinstance(error, MissingAPIKeyError):
        return HTTPException(status_code=503, detail=error.message)

    if error.status_code == 404:
        return HTTPException(
            status_code=404,
            detail=f"Competition '{competition}' not found"
        )

    if error.status_code == 400:
        # usually a bad filter (e.g. unknown status), so pass the reason on
        return HTTPException(status_code=400, detail=error.message)

    if error.status_code is None or error.status_code == 429:
        # timeouts, connection problems, or still rate limited after retries
        return HTTPException(status_code=503, detail=error.message)

    # 403 (bad key / plan limits) or 5xx from the provider
    return HTTPException(
        status_code=502,
        detail=f"Football API error: {error.message}"
    )
