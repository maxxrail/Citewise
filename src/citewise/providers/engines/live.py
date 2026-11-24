"""Live answer-engine adapters, one per provider, with web search on.

Each adapter has two halves: `answer()` makes the SDK call (async client with
`max_retries=0`, because the runner owns retries), and a pure `parse_*()` that turns the
provider payload into an `EngineAnswer`. The parsers are tested offline against payloads
shaped like the providers' documentation.

API shapes checked against official docs on 2026-11-23 (via the api-docs-checker agent),
and the async surfaces checked against the installed SDKs (openai 3.58, anthropic 2.13,
google-genai 3.6):
- OpenAI: Responses API, tools=[{"type": "web_search"}] (web_search_preview is legacy);
  citations are `url_citation` annotations on message output.
- Anthropic: Messages API, tool "web_search_20250305". Newer versions run search inside
  code execution by default, which changes the content blocks. Citations are
  `web_search_result_location` entries on text blocks. `stop_reason: "pause_turn"` means
  "send the content back to continue".
- Gemini: generateContent + Tool(google_search=GoogleSearch()). Sources are
  grounding_metadata.grounding_chunks[].web (uri is a vertexaisearch redirect; the title is
  the source domain). No location parameter exists, so a persona goes in the system
  instruction.
- Perplexity: Agent API (Sonar Chat Completions ended 2026-09-27), OpenAI-compatible
  `/v1/responses`, model "perplexity/sonar", tools=[{"type": "web_search"}]. Sources are
  the `search_results` output item; inline markers are `[web:N]` / `[N]`, mapping to
  results[].id. The response reports its own cost.
"""

from __future__ import annotations

import re
import time
from typing import Any

from citewise.providers.base import Citation, Usage
from citewise.providers.engines.base import EngineAnswer, EngineError, get_path
from citewise.providers.openai_client import extract_url_citations

GROUNDING_HOST = "vertexaisearch.cloud.google.com"
MAX_PAUSE_TURNS = 3  # Anthropic pause_turn continuations before giving up


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _dedupe(pairs: list[tuple[str, str | None]]) -> list[Citation]:
    seen: set[str] = set()
    out: list[Citation] = []
    for url, title in pairs:
        if url and url not in seen:
            seen.add(url)
            out.append(Citation(url=url, title=title, position=len(out) + 1))
    return out


def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
    msgs = [{"role": "system", "content": system}] if system else []
    return [*msgs, {"role": "user", "content": prompt}]


# ---------------------------------------------------------------- OpenAI


def parse_openai(raw: dict[str, Any], model: str) -> EngineAnswer:
    texts = [
        part.get("text", "")
        for item in raw.get("output") or []
        if item.get("type") == "message"
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    ]
    searches = sum(1 for item in raw.get("output") or [] if item.get("type") == "web_search_call")
    usage = raw.get("usage") or {}
    answer = EngineAnswer(
        text="".join(texts).strip(),
        citations=extract_url_citations(raw),
        model=model,
        usage=Usage(usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0),
        searches=searches,
        raw=raw,
    )
    if raw.get("status") not in (None, "completed"):
        # A partial answer would skew mention metrics. Hitting the token limit or the
        # content filter happens again on retry, so only other reasons are retried.
        reason = get_path(raw, "incomplete_details", "reason")
        raise EngineError(
            f"OpenAI response status {raw.get('status')} ({reason})",
            retryable=reason not in ("max_output_tokens", "content_filter"),
            answer=answer,
        )
    return answer


class OpenAIEngine:
    name = "openai"

    def __init__(self, api_key: str, model: str, *, timeout_s: float) -> None:
        from openai import AsyncOpenAI

        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        start = time.perf_counter()
        response = await self.client.responses.create(
            model=self.model,
            input=_messages(prompt, system),  # type: ignore[arg-type]
            tools=[{"type": "web_search"}],  # the model decides whether to search, like ChatGPT
        )
        answer = parse_openai(response.model_dump(mode="json"), self.model)
        answer.latency_ms = _elapsed_ms(start)
        return answer

    async def aclose(self) -> None:
        await self.client.close()


# ---------------------------------------------------------------- Anthropic

_ANTHROPIC_RETRYABLE_SEARCH_ERRORS = {"too_many_requests", "unavailable"}


def parse_anthropic(raw: dict[str, Any], model: str) -> EngineAnswer:
    texts: list[str] = []
    pairs: list[tuple[str, str | None]] = []
    search_error: str | None = None
    for block in raw.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
            for c in block.get("citations") or []:
                if c.get("type") == "web_search_result_location":
                    pairs.append((c.get("url", ""), c.get("title")))
        elif block.get("type") == "web_search_tool_result":
            content = block.get("content")
            if isinstance(content, dict) and content.get("type") == "web_search_tool_result_error":
                search_error = content.get("error_code")
    text = "".join(texts).strip()
    usage = raw.get("usage") or {}
    tokens_in = sum(
        usage.get(k, 0) or 0
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    answer = EngineAnswer(
        text=text,
        citations=_dedupe(pairs),
        model=model,
        usage=Usage(tokens_in, usage.get("output_tokens", 0) or 0),
        searches=int(get_path(usage, "server_tool_use", "web_search_requests", default=0) or 0),
        raw=raw,
    )
    if not text and search_error:
        raise EngineError(
            f"Anthropic web search error: {search_error}",
            retryable=search_error in _ANTHROPIC_RETRYABLE_SEARCH_ERRORS,
            answer=answer,
        )
    stop = raw.get("stop_reason")
    if stop in ("max_tokens", "refusal", "pause_turn"):
        # Truncated, refused, or still paused after MAX_PAUSE_TURNS: all repeat on retry.
        raise EngineError(f"Anthropic stop_reason {stop}", retryable=False, answer=answer)
    return answer


class AnthropicEngine:
    name = "anthropic"

    def __init__(
        self, api_key: str, model: str, *, timeout_s: float, max_searches: int, max_tokens: int
    ) -> None:
        from anthropic import AsyncAnthropic

        self.model = model
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout_s, max_retries=0)
        self.max_searches = max_searches
        self.max_tokens = max_tokens

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        start = time.perf_counter()
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "tools": [
                {"type": "web_search_20250305", "name": "web_search", "max_uses": self.max_searches}
            ],
        }
        if system:
            kwargs["system"] = system
        blocks: list[dict[str, Any]] = []
        usage: dict[str, Any] = {}
        raw: dict[str, Any] = {}
        for _ in range(MAX_PAUSE_TURNS):
            response = await self.client.messages.create(messages=messages, **kwargs)  # type: ignore[arg-type]
            raw = response.model_dump(mode="json", exclude_none=True)
            blocks += raw.get("content") or []
            for k, v in (raw.get("usage") or {}).items():  # accumulate billable usage
                if isinstance(v, int):
                    usage[k] = usage.get(k, 0) + v
                elif isinstance(v, dict):
                    usage.setdefault(k, {})
                    for k2, v2 in v.items():
                        if isinstance(v2, int):
                            usage[k][k2] = usage[k].get(k2, 0) + v2
            if raw.get("stop_reason") != "pause_turn":
                break
            # Long-running search turn: send the content back unchanged to let it finish.
            messages = [*messages, {"role": "assistant", "content": raw.get("content") or []}]
        answer = parse_anthropic({**raw, "content": blocks, "usage": usage}, self.model)
        answer.latency_ms = _elapsed_ms(start)
        return answer

    async def aclose(self) -> None:
        await self.client.close()


# ---------------------------------------------------------------- Gemini


def parse_gemini(raw: dict[str, Any], model: str) -> EngineAnswer:
    """Accepts REST camelCase or the SDK's snake_case `model_dump`."""
    cand = get_path(raw, "candidates", 0, default={}) or {}
    parts = get_path(cand, "content", "parts", default=[]) or []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
    meta = get_path(cand, "grounding_metadata|groundingMetadata", default={}) or {}
    chunks = get_path(meta, "grounding_chunks|groundingChunks", default=[]) or []
    supports = get_path(meta, "grounding_supports|groundingSupports", default=[]) or []

    # Order sources by where they first support the text, then any never-cited chunks.
    first_use: dict[int, int] = {}
    for s in supports:
        start = get_path(s, "segment", "start_index|startIndex", default=0) or 0
        for idx in get_path(s, "grounding_chunk_indices|groundingChunkIndices", default=[]) or []:
            first_use[idx] = min(first_use.get(idx, start), start)
    order = sorted(range(len(chunks)), key=lambda i: (i not in first_use, first_use.get(i, 0), i))
    pairs = [
        (get_path(chunks[i], "web", "uri", default=""), get_path(chunks[i], "web", "title"))
        for i in order
    ]
    um = get_path(raw, "usage_metadata|usageMetadata", default={}) or {}

    def n(*keys: str) -> int:
        return sum(int(get_path(um, k, default=0) or 0) for k in keys)

    queries = get_path(meta, "web_search_queries|webSearchQueries", default=[]) or []
    answer = EngineAnswer(
        text=text,
        citations=_dedupe(pairs),
        model=model,
        usage=Usage(
            n(
                "prompt_token_count|promptTokenCount",
                "tool_use_prompt_token_count|toolUsePromptTokenCount",
            ),
            n(
                "candidates_token_count|candidatesTokenCount",
                "thoughts_token_count|thoughtsTokenCount",
            ),
        ),
        searches=len(queries),
        raw=raw,
    )
    finish = get_path(cand, "finish_reason|finishReason")
    if finish == "MAX_TOKENS" or (finish not in (None, "STOP") and not text):
        # Truncated, or blocked (SAFETY, RECITATION, ...): the same thing happens on retry.
        raise EngineError(f"Gemini finish_reason {finish}", retryable=False, answer=answer)
    return answer


class GeminiEngine:
    name = "gemini"

    def __init__(self, api_key: str, model: str, *, timeout_s: float) -> None:
        from google import genai
        from google.genai import types

        self.model = model
        self.types = types
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)
            ),
        )

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        t = self.types
        start = time.perf_counter()
        response = await self.client.aio.models.generate_content(
            model=self.model,
            contents=prompt,
            config=t.GenerateContentConfig(
                tools=[t.Tool(google_search=t.GoogleSearch())], system_instruction=system
            ),
        )
        answer = parse_gemini(response.model_dump(mode="json", exclude_none=True), self.model)
        answer.latency_ms = _elapsed_ms(start)
        return answer

    async def aclose(self) -> None:
        await self.client.aio.aclose()


# ---------------------------------------------------------------- Perplexity

_MARKER = re.compile(r"\[(?:web:)?(\d+)\]")
# With an explicit model (not an `effort` preset) Perplexity only emits citation markers
# when asked to.
PERPLEXITY_CITE_INSTRUCTION = "Cite sources inline as [web:N] using the search result ids."


def parse_perplexity(raw: dict[str, Any], model: str) -> EngineAnswer:
    results: dict[int, dict[str, Any]] = {}
    ordered_ids: list[int] = []
    texts: list[str] = []
    for item in raw.get("output") or []:
        if item.get("type") == "search_results":
            for r in item.get("results") or []:
                if isinstance(r.get("id"), int) and r["id"] not in results:
                    results[r["id"]] = r
                    ordered_ids.append(r["id"])
        elif item.get("type") == "message":
            texts += [
                p.get("text", "")
                for p in item.get("content") or []
                if p.get("type") == "output_text"
            ]
    text = "".join(texts).strip()
    cited = [int(m) for m in _MARKER.findall(text) if int(m) in results]
    # Perplexity shows its sources even when the text has no markers; keep them in order.
    ids = cited or ordered_ids
    pairs = [(results[i].get("url", ""), results[i].get("title")) for i in ids]
    usage = raw.get("usage") or {}
    searches = get_path(usage, "tool_calls_details", "search_web", "invocation")
    reported = get_path(usage, "cost", "total_cost")
    return EngineAnswer(
        text=text,
        citations=_dedupe(pairs),
        model=model,
        usage=Usage(usage.get("input_tokens", 0) or 0, usage.get("output_tokens", 0) or 0),
        searches=int(searches) if isinstance(searches, int) else int(bool(results)),
        raw=raw,
        reported_cost_usd=float(reported) if isinstance(reported, int | float) else None,
    )


class PerplexityEngine:
    name = "perplexity"
    BASE_URL = "https://api.perplexity.ai/v1"

    def __init__(self, api_key: str, model: str, *, timeout_s: float) -> None:
        from openai import AsyncOpenAI  # Perplexity's Agent API is OpenAI-compatible

        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key, base_url=self.BASE_URL, timeout=timeout_s, max_retries=0
        )

    async def answer(self, prompt: str, *, system: str | None = None) -> EngineAnswer:
        start = time.perf_counter()
        sys_msg = (
            f"{system}\n{PERPLEXITY_CITE_INSTRUCTION}" if system else PERPLEXITY_CITE_INSTRUCTION
        )
        response = await self.client.responses.create(
            model=self.model,
            input=_messages(prompt, sys_msg),  # type: ignore[arg-type]
            tools=[{"type": "web_search"}],  # type: ignore[list-item]
        )
        answer = parse_perplexity(response.model_dump(mode="json"), self.model)
        answer.latency_ms = _elapsed_ms(start)
        return answer

    async def aclose(self) -> None:
        await self.client.close()


def is_grounding_redirect(url: str) -> bool:
    return GROUNDING_HOST in url
