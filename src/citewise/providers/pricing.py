"""Token prices used to estimate the cost of every LLM call.

Prices are USD per 1M tokens: (input, output). Embedding models have output = 0.
Last checked against provider pricing pages: 2026-11-23. Prices change; update this table
(and the date) rather than hardcoding costs elsewhere.
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

PRICES_LAST_CHECKED = "2026-11-23"

# model id -> (usd per 1M input tokens, usd per 1M output tokens)
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    # OpenAI (Standard tier, uncached input) - developers.openai.com/api/docs/pricing
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6.2-sol": (2.00, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    # Anthropic - platform.claude.com/docs/en/about-claude/pricing
    # Haiku 5.5 is tiered: prompts >100K tokens cost $0.50/$2.50; we only record the base tier.
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),  # checked 2026-11-24 (claude-api reference)
    # Local embedding model (fastembed, runs on CPU): no per-token charge.
    "BAAI/bge-small-en-v1.5": (0.0, 0.0),
    # Google - aistudio.google.com/docs/pricing (text)
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Perplexity Agent API (docs.perplexity.ai/docs/getting-started/pricing, 2026-11-23)
    "perplexity/sonar": (0.25, 2.50),
    "sonar": (0.25, 2.50),  # legacy Sonar ID, kept so older .env files still price
}

# Web-search fee per search, USD, by engine (checked 2026-11-23). Search results are also
# billed as input tokens at the model's rate, which token usage already covers.
# - OpenAI web_search tool: $10 / 1k calls.
# - Anthropic web_search: $10 / 1k searches (failed searches not billed).
# - Gemini Search grounding (Gemini 3+): 5,000 free queries/month, then $14 / 1k queries.
#   We always charge the paid rate: over-estimating keeps the budget guard safe.
# - Perplexity web_search tool: $2.50 / 1k calls (Perplexity also reports exact cost).
SEARCH_FEE_USD: dict[str, float] = {
    "openai": 0.010,
    "anthropic": 0.010,
    "gemini": 0.014,
    "perplexity": 0.0025,
}


def search_cost(engine: str, searches: int) -> float:
    return round(SEARCH_FEE_USD.get(engine, 0.0) * max(searches, 0), 8)


_warned: set[str] = set()


# Dated snapshot suffixes: `-2026-11-01` (OpenAI style) or `-20261101` (Anthropic style).
_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{8})$")


class UnpricedModelError(RuntimeError):
    """Raised in live mode for a model with no price, so spend is never recorded as $0."""


def price_for(model: str) -> tuple[float, float] | None:
    """Exact match, else the alias of a dated snapshot (`gpt-6-luna-2026-11-01`).

    Only date suffixes are stripped: a generic prefix match would price a different model
    (`gpt-6-luna-mini`) as its namesake and silently under- or over-count spend.
    """
    if model in PRICES_PER_MTOK:
        return PRICES_PER_MTOK[model]
    alias = _SNAPSHOT_SUFFIX.sub("", model)
    return PRICES_PER_MTOK.get(alias) if alias != model else None


def require_price(model: str) -> None:
    """Fail closed before a live call whose cost we couldn't record."""
    if price_for(model) is None:
        raise UnpricedModelError(
            f"No price for model {model!r} in providers/pricing.py. Add its per-1M-token "
            "prices (and update PRICES_LAST_CHECKED) before using it in live mode."
        )


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimated USD cost of one call. Unknown models cost 0 and log a warning once."""
    price = price_for(model)
    if price is None:
        if model not in _warned:
            log.warning("No price for model %r; recording cost 0. Update pricing.py.", model)
            _warned.add(model)
        return 0.0
    in_price, out_price = price
    return round((tokens_in * in_price + tokens_out * out_price) / 1_000_000, 8)
