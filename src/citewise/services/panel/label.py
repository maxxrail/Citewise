"""Name a cluster from its most central prompts (structured output, <= 5-word label)."""

from __future__ import annotations

import re
from collections import Counter

from citewise.providers.base import ChatProvider
from citewise.schemas.panel import TopicLabel
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

LABEL_TEMPLATE = "panel/label_topic"
CENTRAL_PROMPTS = 8

_WORD = re.compile(r"[a-z][a-z0-9'-]+")
_STOP = frozenset(
    {
        *("about", "and", "are", "best", "can", "does", "for", "from", "good", "how", "into"),
        *("is", "it", "its", "my", "need", "or", "should", "that", "the", "their", "this"),
        *("what", "whats", "when", "which", "who", "why", "will", "with", "you", "your", "vs"),
        *("any", "get", "have", "want", "worth", "really", "there", "than", "one", "use"),
    }
)


def fallback_label(texts: list[str]) -> TopicLabel:
    """Top 3 content words: a readable label when the LLM call fails twice."""
    counts = Counter(w for t in texts for w in set(_WORD.findall(t.lower())) if w not in _STOP)
    words = [w[:15] for w, _ in counts.most_common(3)] or ["misc"]
    return TopicLabel(
        label=" ".join(w.capitalize() for w in words)[:60].strip(),
        description=f"Questions mentioning {', '.join(words)}.",
    )


def label_cluster(
    chat: ChatProvider, category: str, central_texts: list[str], model: str, version: str = "v1"
) -> tuple[TopicLabel, str]:
    """Returns (label, source) where source is "llm" or "fallback"."""
    prompt = render(LABEL_TEMPLATE, version, category=category, prompts=central_texts)
    label = complete_structured(
        chat, prompt.messages, TopicLabel, model=model, purpose="panel.label_topic"
    )
    if label is None:
        return fallback_label(central_texts), "fallback"
    return label, "llm"
