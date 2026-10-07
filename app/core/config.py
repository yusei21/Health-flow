from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self
from uuid import UUID

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HEALTHFLOW_", env_file=".env", extra="ignore", frozen=True
    )

    app_env: AppEnv = AppEnv.DEVELOPMENT
    log_level: str = "INFO"

    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen3:4b"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_retries: int = Field(default=1, ge=0, le=3)

    ml_model_dir: Path = Path("models")
    ml_low_confidence_threshold: float = Field(default=0.55, ge=0, le=1)

    facility_search_radius_km: float = Field(default=25.0, gt=0)

    demo_auth_token: SecretStr | None = None
    demo_user_id: UUID | None = None

    @model_validator(mode="after")
    def _forbid_demo_auth_in_production(self) -> Self:
        # The static demo token exists only to exercise the flow locally.
        if self.app_env is AppEnv.PRODUCTION and self.demo_auth_token is not None:
            raise ValueError("demo authentication must not be enabled in production")
        return self

    @property
    def demo_auth_enabled(self) -> bool:
        return bool(self.demo_auth_token and self.demo_auth_token.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
