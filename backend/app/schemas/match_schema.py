from datetime import datetime

from pydantic import BaseModel


class MatchResponse(BaseModel):

    id: int

    home_team: str

    away_team: str

    date: datetime | None = None

    status: str | None = None

    home_score: int | None = None

    away_score: int | None = None


class MatchListResponse(BaseModel):

    matches: list[MatchResponse]
