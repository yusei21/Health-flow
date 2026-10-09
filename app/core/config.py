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


class HarnessPlannerKind(StrEnum):
    DETERMINISTIC = "deterministic"
    JEV = "jev"


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

    ml_model_dir: Path = Path("models/synthetic-v1")
    ml_low_confidence_threshold: float = Field(default=0.55, ge=0, le=1)

    facility_search_radius_km: float = Field(default=25.0, gt=0)
    cnes_database: Path | None = None

    demo_auth_token: SecretStr | None = None
    demo_user_id: UUID | None = None

    transcription_url: str | None = None
    transcription_api_key: SecretStr | None = None
    transcription_model: str = "whisper-1"

    # Browser origins allowed to call the API (e.g. the Vite dev server). Empty = CORS off.
    cors_allowed_origins: list[str] = Field(default_factory=list)

    harness_planner: HarnessPlannerKind = HarnessPlannerKind.DETERMINISTIC
    jev_enabled: bool = False
    jev_base_url: str = "https://jevmodel.org"
    jev_model: str = "jev-latest"
    jev_api_key: SecretStr | None = None
    jev_timeout_seconds: float = Field(default=5.0, gt=0, le=30)
    jev_min_probability: float = Field(default=0.70, ge=0, le=1)

    @model_validator(mode="after")
    def _forbid_demo_auth_in_production(self) -> Self:
        # The static demo token exists only to exercise the flow locally.
        if self.app_env is AppEnv.PRODUCTION and self.demo_auth_token is not None:
            raise ValueError("demo authentication must not be enabled in production")
        return self

    @model_validator(mode="after")
    def _forbid_wildcard_cors_in_production(self) -> Self:
        if self.app_env is AppEnv.PRODUCTION and "*" in self.cors_allowed_origins:
            raise ValueError("wildcard CORS origin must not be used in production")
        return self

    @property
    def demo_auth_enabled(self) -> bool:
        return bool(self.demo_auth_token and self.demo_auth_token.get_secret_value())


@lru_cache
def get_settings() -> Settings:
    return Settings()
