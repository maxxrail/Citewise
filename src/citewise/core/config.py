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
    # Checked 2026-11-23: gemini-3.1-flash-lite is missing from Google's list of models
    # supported by Search grounding, so the default is 3.5 Flash-Lite (listed on both pages).
    engine_model_gemini: str = "gemini-3.5-flash-lite"
    # Sonar Chat Completions ended 2026-09-27; the Agent API ID is "perplexity/sonar".
    engine_model_perplexity: str = "perplexity/sonar"

    max_cost_usd_per_run: float = Field(default=3.0, ge=0)

    # Scan runner (Module 2). Per-engine limits are deliberately conservative defaults:
    # Perplexity Tier 0 allows 1 query/second, so its rate is the binding one.
    scan_concurrency: int = Field(default=4, ge=1, le=64)  # in-flight requests per engine
    scan_rate_per_s: dict[str, float] = Field(
        default_factory=lambda: {"openai": 5.0, "anthropic": 3.0, "gemini": 3.0, "perplexity": 1.0}
    )
    scan_max_attempts: int = Field(default=4, ge=1, le=10)  # first try + retries
    scan_backoff_base_s: float = Field(default=1.0, gt=0)  # 1s, 2s, 4s ... plus jitter
    scan_backoff_max_s: float = Field(default=30.0, gt=0)
    scan_max_searches: int = Field(default=3, ge=1, le=10)  # per answer, where supported
    # Output-token cap per answer, by engine. Without one, a single call's cost is bounded
    # only by the model's own limit (Perplexity: 128K), so the budget guard's per-call
    # reservation could be overshot without limit. Checked 2026-11-24: reasoning/thinking
    # tokens count against the cap on OpenAI (gpt-6-luna reasons at "medium" by default),
    # Gemini and Anthropic, hence the headroom over a ~700-word answer (~1K tokens).
    scan_max_output_tokens: dict[str, int] = Field(
        default_factory=lambda: {
            "openai": 6000,
            "anthropic": 4096,
            "gemini": 3000,
            "perplexity": 2500,
        }
    )
    # Budget estimate per answer: answers with search run long, and search results are
    # billed as input tokens.
    scan_est_tokens_in: int = Field(default=3000, ge=0)
    scan_est_tokens_out: int = Field(default=700, ge=0)
    # Expected searches per answer, by engine: OpenAI and Perplexity usually make one web
    # search call per answer; Anthropic (up to max_uses) and Gemini (every grounding query is
    # billed) often make more.
    scan_est_searches: dict[str, float] = Field(
        default_factory=lambda: {"openai": 1.0, "anthropic": 1.5, "gemini": 1.5, "perplexity": 1.0}
    )

    # Per-request limits for live clients. The SDK default timeout is 600s; one hung call
    # would stall a whole run.
    llm_timeout_s: float = Field(default=60.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0)

    # Extraction (Module 3): parallel Stage B calls in live mode (mock runs serially).
    extract_concurrency: int = Field(default=4, ge=1, le=32)
    # Bootstrap resamples for metric confidence intervals.
    metrics_bootstrap: int = Field(default=1000, ge=100, le=20000)

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
