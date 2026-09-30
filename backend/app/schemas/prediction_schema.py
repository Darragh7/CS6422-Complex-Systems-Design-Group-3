from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    home_win: float = Field(ge=0, le=1)

    draw: float = Field(ge=0, le=1)

    away_win: float = Field(ge=0, le=1)