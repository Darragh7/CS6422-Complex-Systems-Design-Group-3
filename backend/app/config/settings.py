from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    football_api_key: str = ""

    # football-data.org v4 API
    football_api_base_url: str = "https://api.football-data.org/v4"

    # seconds to wait for the football API before giving up
    football_api_timeout: float = 10.0

    database_url: str = Field(min_length=1)

    secret_key: str = Field(min_length=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
