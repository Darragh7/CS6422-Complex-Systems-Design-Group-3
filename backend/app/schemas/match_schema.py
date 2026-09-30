from datetime import datetime

from pydantic import BaseModel


class MatchResponse(BaseModel):

    id: int

    home_team: str

    away_team: str

    date: datetime | None = None