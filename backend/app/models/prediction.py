from sqlalchemy import Column, Integer, String, DateTime
from app.database.database import Base


class Match(Base):

    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, index=True)

    home_team = Column(String, nullable=False)

    away_team = Column(String, nullable=False)

    match_date = Column(DateTime, nullable=False)

    home_score = Column(Integer, nullable=True)

    away_score = Column(Integer, nullable=True)