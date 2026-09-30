class PredictionService:

    def predict(self, match_id: int):

        # Temporary prediction.
        # This will eventually call the trained AI model.

        return {
            "home_win": 0.45,
            "draw": 0.30,
            "away_win": 0.25
        }