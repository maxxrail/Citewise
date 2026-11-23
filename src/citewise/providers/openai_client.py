"""Live OpenAI backend: chat via the Responses API, embeddings via /embeddings.

API shape checked against developers.openai.com docs on 2026-11-23 (openai SDK 3.58):
- Responses is recommended over Chat Completions for new projects.
- Structured output: `text={"format": {"type": "json_schema", "name", "schema", "strict"}}`.
- Usage: `response.usage.input_tokens` / `output_tokens`; embeddings: `usage.prompt_tokens`.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from citewise.providers.base import (
    ChatResult,
    Citation,
    EmbedResult,
    JsonSchema,
    Message,
    Usage,
)

log = logging.getLogger(__name__)


def _client(api_key: str) -> Any:
    from openai import OpenAI  # imported lazily so mock mode never needs the SDK configured

    return OpenAI(api_key=api_key)


def extract_url_citations(raw: dict[str, Any]) -> list[Citation]:
    """Pull `url_citation` annotations out of a Responses API payload, in order, de-duplicated."""
    seen: set[str] = set()
    citations: list[Citation] = []
    for item in raw.get("output", []) or []:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []) or []:
            for ann in part.get("annotations", []) or []:
                url = ann.get("url")
                if ann.get("type") != "url_citation" or not url or url in seen:
                    continue
                seen.add(url)
                citations.append(
                    Citation(url=url, title=ann.get("title"), position=len(citations) + 1)
                )
    return citations


class OpenAIChatProvider:
    name = "openai"

    def __init__(self, api_key: str) -> None:
        self.client = _client(api_key)

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {"model": model, "input": list(messages)}
        if json_schema is not None:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema.name,
                    "schema": json_schema.schema,
                    "strict": True,
                }
            }
        if tools:
            kwargs["tools"] = tools

        response = self.client.responses.create(**kwargs)
        raw: dict[str, Any] = response.model_dump(mode="json")
        text: str = response.output_text or ""

        parsed: dict[str, Any] | None = None
        if json_schema is not None:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                # Caller validates `parsed` with Pydantic and owns the retry policy.
                log.warning("OpenAI returned non-JSON for schema %s", json_schema.name)

        usage = response.usage
        return ChatResult(
            text=text,
            model=raw.get("model", model),
            provider=self.name,
            parsed=parsed,
            citations=extract_url_citations(raw),
            usage=Usage(
                tokens_in=getattr(usage, "input_tokens", 0) or 0,
                tokens_out=getattr(usage, "output_tokens", 0) or 0,
            ),
            raw=raw,
        )


class OpenAIEmbedder:
    name = "openai"

    def __init__(self, api_key: str, model: str) -> None:
        self.client = _client(api_key)
        self.model = model

    def embed_with_usage(self, texts: list[str]) -> EmbedResult:
        response = self.client.embeddings.create(input=texts, model=self.model)
        vectors = [d.embedding for d in sorted(response.data, key=lambda d: d.index)]
        return EmbedResult(
            vectors=vectors,
            model=self.model,
            provider=self.name,
            usage=Usage(tokens_in=getattr(response.usage, "prompt_tokens", 0) or 0),
        )
