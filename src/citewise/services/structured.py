"""Structured-output calls: JSON schema in, validated Pydantic model out.

Policy (CLAUDE.md): validate with Pydantic, retry once on failure (telling the model what
was wrong), then give up *without* raising, so one bad call can't crash a whole run. The
caller decides the fallback. Both attempts are cost-tracked by the provider layer.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from citewise.providers.base import ChatProvider, JsonSchema, Message
from citewise.providers.budget import BudgetExceededError
from citewise.providers.pricing import UnpricedModelError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@lru_cache
def schema_for(model: type[BaseModel]) -> JsonSchema:
    """Strict JSON schema for a Pydantic model (raises if it isn't strict-compatible)."""
    return JsonSchema(name=model.__name__, schema=model.model_json_schema())


def _parse(model: type[T], parsed: dict | None, text: str) -> T:
    if parsed is not None:
        return model.model_validate(parsed)
    return model.model_validate_json(text)


def complete_structured(
    chat: ChatProvider,
    messages: list[Message],
    output: type[T],
    *,
    model: str,
    purpose: str,
) -> T | None:
    """Return a validated `output`, or None after two failed attempts (logged as a warning)."""
    schema = schema_for(output)
    attempt_messages = list(messages)
    error = ""
    for attempt in (1, 2):
        try:
            result = chat.complete(
                attempt_messages, model=model, purpose=purpose, json_schema=schema
            )
        except UnpricedModelError:
            raise  # configuration error: fail loudly before anything is spent
        except BudgetExceededError as exc:
            log.error("%s: skipped, %s", purpose, exc)
            return None  # no retry: the next call would be refused too
        except Exception as exc:  # provider error after the SDK's own retries
            # The tracker already wrote an ok=False llm_calls row. Returning None lets the
            # caller's fallback run instead of rolling back everything paid for so far.
            error = f"{type(exc).__name__}: {exc}"[:1000]
            log.warning("%s: provider error (attempt %d): %s", purpose, attempt, error)
            continue
        try:
            return _parse(output, result.parsed, result.text)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            error = str(exc)[:1000]
            log.warning("%s: invalid structured output (attempt %d): %s", purpose, attempt, error)
            attempt_messages = [
                *messages,
                {"role": "assistant", "content": result.text[:4000]},
                {
                    "role": "user",
                    "content": f"That output failed validation: {error}\n"
                    "Return corrected JSON that matches the schema exactly.",
                },
            ]
    log.error("%s: giving up after 2 invalid outputs: %s", purpose, error)
    return None
