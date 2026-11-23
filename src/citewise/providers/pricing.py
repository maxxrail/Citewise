"""Token prices used to estimate the cost of every LLM call.

Prices are USD per 1M tokens: (input, output). Embedding models have output = 0.
Last checked against provider pricing pages: 2026-11-23. Prices change; update this table
(and the date) rather than hardcoding costs elsewhere.
"""

from __future__ import annotations

import logging

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
    # Google - aistudio.google.com/docs/pricing (text)
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    # Perplexity Agent API "perplexity/sonar" (search fees not included; re-verify in Module 2)
    "sonar": (0.25, 2.50),
    "perplexity/sonar": (0.25, 2.50),
}

_warned: set[str] = set()


def estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    """Estimated USD cost of one call. Unknown models cost 0 and log a warning once."""
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        if model not in _warned:
            log.warning("No price for model %r; recording cost 0. Update pricing.py.", model)
            _warned.add(model)
        return 0.0
    in_price, out_price = price
    return round((tokens_in * in_price + tokens_out * out_price) / 1_000_000, 8)
