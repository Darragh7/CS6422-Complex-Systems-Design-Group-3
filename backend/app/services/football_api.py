class FootballAPI:

    async def get_matches(self):

#TEMPORARY MOCK DATA
        return [
            {
                "id": 1,
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "date": "2026-10-01"
            },
            {
                "id": 2,
                "home_team": "Liverpool",
                "away_team": "Manchester City",
                "date": "2026-10-02"
            }
        ]