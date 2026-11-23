"""Versioned prompt templates (`prompts/<family>/<name>.<version>.jinja`).

A template has a system part and a user part, split by a line containing only
`=== user ===`. The version string (e.g. "panel/generate.v1") is stored on whatever the
prompt produced, so results can be traced to the exact wording and A/B tested (Module 6).

Values interpolated into templates are data, not instructions: callers pass user- or
brand-supplied text through `as_data()` (JSON-encoded), never raw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Undefined

from citewise.core.config import get_settings
from citewise.providers.base import Message

USER_MARKER = "=== user ==="


def as_data(value: Any) -> str:
    """JSON-encode a value for a template so quotes/newlines can't break out of it."""
    if isinstance(value, Undefined):
        # Raise jinja's own "'x' is undefined" instead of an opaque JSON serialization error.
        value._fail_with_undefined_error()
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True)
class RenderedPrompt:
    version: str  # "panel/generate.v1"
    messages: list[Message]


@lru_cache
def _env(prompts_dir: Path) -> Environment:
    # Not HTML, so no autoescape; StrictUndefined turns a missing variable into an error
    # instead of silently rendering an empty string into a live prompt.
    env = Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["data"] = as_data
    return env


def render(name: str, version: str, prompts_dir: Path | None = None, **ctx: Any) -> RenderedPrompt:
    """Render `prompts/<name>.<version>.jinja`, e.g. render("panel/generate", "v1", ...)."""
    prompts_dir = prompts_dir or get_settings().prompts_dir
    text = _env(prompts_dir).get_template(f"{name}.{version}.jinja").render(**ctx)
    # Mock fixture keys hash the rendered messages: a CRLF checkout must not change them.
    text = text.replace("\r\n", "\n")
    if USER_MARKER not in text:
        raise ValueError(f"Template {name}.{version} is missing the {USER_MARKER!r} line")
    system, user = (part.strip() for part in text.split(USER_MARKER, 1))
    messages: list[Message] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return RenderedPrompt(version=f"{name}.{version}", messages=messages)
