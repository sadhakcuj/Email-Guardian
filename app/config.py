"""Runtime configuration, read from environment variables (12-factor)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SPAM_", env_file=".env", extra="ignore")

    # Comma-separated API keys. Empty = auth disabled (local dev only; a warning is logged).
    api_keys: str = ""
    model_path: Path = ROOT / "models" / "spam_model.joblib"
    metrics_path: Path = ROOT / "models" / "metrics.json"
    feedback_path: Path = ROOT / "data" / "feedback.jsonl"

    # Scores in [review_floor, threshold) are returned as verdict="review".
    review_floor: float = Field(default=0.35, ge=0.0, le=1.0)
    max_text_chars: int = Field(default=20_000, ge=100)
    max_batch_size: int = Field(default=100, ge=1, le=1000)

    # slowapi limit strings, per client IP.
    rate_limit_demo: str = "20/minute"
    rate_limit_api: str = "300/minute"

    demo_enabled: bool = True
    log_level: str = "INFO"
    cors_origins: str = ""  # comma-separated; empty = no CORS

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
