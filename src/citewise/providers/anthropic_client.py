"""Live Anthropic backend for Citewise's own LLM calls (generation, labelling, extraction,
judging): the Messages API with structured outputs.

API shape checked against the Claude API reference (claude-api skill, models cached
2026-11-03) on 2026-11-24, anthropic SDK 2.13:
- Structured output: `output_config={"format": {"type": "json_schema", "schema": ...}}` on
  `messages.create`; the first text block is then valid JSON. `$ref`/`$defs`, `anyOf` and
  `enum` are supported; numeric and string-length constraints (`minimum`, `maxLength`, ...)
  are not, so they are stripped here and Pydantic re-checks them (services/structured.py).
- System prompts go in the top-level `system` parameter, not as a message.
- No assistant prefill on current models: a conversation must end with a user turn, which
  `complete_structured`'s retry (assistant output + user correction) already does.
- `stop_reason` "max_tokens" or "refusal" means the output may not match the schema: the
  result is marked not completed so tracking records it as not ok.
- Thinking is adaptive by default on Claude Haiku 5.5 / Opus 5.5 and counts against
  `max_tokens`, hence the generous default.

This is separate from the *answer-engine* adapter (providers/engines/live.py), which asks
Claude questions with web search on, as one of the engines being measured.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from citewise.providers.base import ChatResult, JsonSchema, Message, Usage

log = logging.getLogger(__name__)

# Not supported by structured outputs; Pydantic validates them after the call instead.
UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
        "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems",
    }
)  # fmt: skip
COMPLETED_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})


def anthropic_schema(schema: Any) -> Any:
    """A copy of `schema` without the constraints structured outputs rejects. Keys of a
    `properties` / `$defs` mapping are names, not keywords, so they are always kept (a
    field called "pattern" stays)."""
    if isinstance(schema, list):
        return [anthropic_schema(v) for v in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("properties", "$defs", "definitions") and isinstance(v, dict):
            out[k] = {name: anthropic_schema(sub) for name, sub in v.items()}
        elif k not in UNSUPPORTED_SCHEMA_KEYS:
            out[k] = anthropic_schema(v)
    return out


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Citewise templates render `[system, user, ...]`; the Messages API takes the system
    prompt as its own parameter."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return (system or None), [m for m in messages if m["role"] != "system"]


class AnthropicChatProvider:
    name = "anthropic"

    def __init__(
        self,
        api_key: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        max_tokens: int = 16000,
    ) -> None:
        from anthropic import Anthropic  # lazy: mock mode never needs the SDK configured

        self.client = Anthropic(api_key=api_key, timeout=timeout_s, max_retries=max_retries)
        self.max_tokens = max_tokens

    def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        purpose: str = "general",
        json_schema: JsonSchema | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResult:
        system, convo = split_system(messages)
        kwargs: dict[str, Any] = {"model": model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            kwargs["system"] = system
        if json_schema is not None:
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": anthropic_schema(json_schema.schema)}
            }
        if tools:
            kwargs["tools"] = tools

        response = self.client.messages.create(**kwargs)
        return parse_response(
            response.model_dump(mode="json", exclude_none=True), model, json_schema
        )


def parse_response(raw: dict[str, Any], model: str, json_schema: JsonSchema | None) -> ChatResult:
    """Pure: Messages API payload -> ChatResult (tested offline)."""
    text = "".join(b.get("text", "") for b in raw.get("content") or [] if b.get("type") == "text")
    parsed: dict[str, Any] | None = None
    if json_schema is not None:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # The caller validates `parsed` with Pydantic and owns the retry policy.
            log.warning("Anthropic returned non-JSON for schema %s", json_schema.name)
    stop = raw.get("stop_reason")
    usage = raw.get("usage") or {}
    tokens_in = sum(
        int(usage.get(k, 0) or 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    return ChatResult(
        text=text,
        model=model,  # price by the requested ID; the served model stays in raw["model"]
        provider="anthropic",
        status="completed" if stop in COMPLETED_STOP_REASONS or stop is None else str(stop),
        parsed=parsed,
        usage=Usage(tokens_in=tokens_in, tokens_out=int(usage.get("output_tokens", 0) or 0)),
        raw=raw,
    )
