from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    football_api_key: str = ""

    database_url: str = (
        "postgresql://sportsuser:sportspassword"
        "@localhost:5432/sportsdb"
    )

    secret_key: str = "development-secret"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()