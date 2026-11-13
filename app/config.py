from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openrouter_api_key: str | None = None
    openrouter_model: str = "openrouter/free"

    github_token: str | None = None
    qdrant_url: str | None = None
    qdrant_collection: str = "mcp_forge_memory"

    allow_github_writes: bool = False
    log_level: str = "INFO"

    model_config = SettingsConfigDict(extra="ignore")


settings = Settings()
