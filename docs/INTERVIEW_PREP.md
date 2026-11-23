# Interview prep

Likely questions per module, with short, strong answers grounded in the actual code.

## Module 0 - Foundations

**1. "How do you know what every LLM call costs, and what stops a run from silently overspending?"**
Every call goes through one wrapper (`providers/tracking.py`), which `registry` always applies, so no caller can bypass it. It times the call, prices tokens from a dated table in `pricing.py`, and writes one `llm_calls` row per call. Failures and truncated responses are logged too, and the row is written in its own transaction so a caller's rollback can't erase it. Dated snapshot IDs map back to their alias. In live mode an unpriced model is refused *before* the request (`UnpricedModelError`), because recording $0 would blind the Module 2 budget guard. A test asserts the row, cost and latency for each of the four engines.

**2. "Your tests never touch the network. How do you trust that mock mode resembles production?"**
Mocks implement the same `ChatProvider` protocol as the live clients and are deterministic: the fixture key is sha256 of purpose plus messages, optionally per engine, and the embedder uses a seeded PRNG rather than Python's salted `hash()`. The structural risks are checked offline. `JsonSchema` rejects schemas OpenAI's strict mode would 400 on. The mock builds schema-valid output, including Pydantic `$ref`/`$defs`; a review found it had been returning strings for nested models. Mock calls carry realistic token and cost estimates, so the cost path runs in CI. The remaining gap is honest and documented: the live OpenAI path needs one smoke call with a real key.

**3. "User-supplied brand data ends up in your prompts. What's the injection story?"**
Defence in depth, starting at the boundary. `BrandCreate` caps aliases and competitors (20 items, 100 chars each), strips and dedupes them, rejects newlines and control characters (the cheapest way to fake a new instruction block), and reduces the domain to a validated hostname. That also bounds cost, since prompts scale with N aliases × M competitors. From Module 1 on, templates put brand values in delimited data sections and never treat them as instructions. Extraction output is schema-validated with Pydantic, so an injected "rank me first" can't change the output shape, and evals include red-team cases for it.
