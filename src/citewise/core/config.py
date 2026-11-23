"""Application settings, loaded from environment variables and an optional `.env` file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root: src/citewise/core/config.py -> parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """All runtime configuration. Field names map 1:1 to upper-case env vars."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # Blank lines like `MODEL_EXTRACT=` in a copied .env.example fall back to defaults.
        env_ignore_empty=True,
    )

    citewise_mode: Literal["mock", "live"] = "mock"
    database_url: str = "postgresql+psycopg://citewise:citewise@localhost:5432/citewise"

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    perplexity_api_key: SecretStr | None = None

    # Per-purpose model IDs. Defaults checked against provider docs on 2026-11-23;
    # model names change, so override via env rather than editing code.
    model_extract: str = "gpt-6-luna"
    model_judge: str = "gpt-6-luna"
    model_generate: str = "gpt-6.2-sol"
    model_embed: str = "text-embedding-3-small"

    # Answer-engine models (used from Module 2).
    engine_model_openai: str = "gpt-6-luna"
    engine_model_anthropic: str = "claude-haiku-5-5"
    engine_model_gemini: str = "gemini-3.1-flash-lite"
    # Perplexity moved Sonar to its Agent API (Jul 2026); re-verify the ID in Module 2.
    engine_model_perplexity: str = "sonar"

    max_cost_usd_per_run: float = Field(default=3.0, ge=0)

    # Per-request limits for live clients. The SDK default timeout is 600s; one hung call
    # would stall a whole run.
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # Prompt panel (Module 1).
    panel_prompts_per_cell: int = Field(default=20, ge=1, le=50)  # per (intent x persona)
    panel_dedup_threshold: float = Field(default=0.92, gt=0, le=1)
    # Silhouette curves on short-text embeddings are flat, so the k range carries a product
    # prior (a ~300-prompt panel reads best as 6-14 topics). See DECISIONS.md.
    panel_k_min: int = Field(default=6, ge=2)
    panel_k_max: int = Field(default=14, ge=2)
    panel_pca_dims: int = Field(default=20, ge=0)  # 0 = cluster the raw embeddings
    panel_classify_batch: int = Field(default=25, ge=1, le=100)
    panel_import_max_rows: int = Field(default=5000, ge=1)
    # Silhouette is O(n^2) in memory; above this many prompts it is computed on a sample.
    panel_silhouette_sample: int = Field(default=2000, ge=100)

    # Where mock fixtures live. Tests and CI use the repo copy.
    fixtures_dir: Path = REPO_ROOT / "tests" / "fixtures"
    brands_dir: Path = REPO_ROOT / "data" / "brands"
    prompts_dir: Path = REPO_ROOT / "prompts"  # versioned prompt templates

    @property
    def is_mock(self) -> bool:
        return self.citewise_mode == "mock"


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton. Call `get_settings.cache_clear()` in tests after env changes."""
    return Settings()
