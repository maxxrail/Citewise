"""Record real Stage B (mention extraction) responses as mock fixtures.

In mock mode Stage B is a simulator unless a recorded response exists for the exact
prompt. Run this once with a key and the offline eval (`citewise eval extract`, and CI)
replays a *real model's* answers on the golden set instead of the simulator's:

    $env:CITEWISE_MODE="live"   # PowerShell; needs OPENAI_API_KEY
    uv run python scripts/record_extract_fixtures.py --max-cost 0.10

Each golden answer costs about $0.0002 on the default MODEL_EXTRACT. Fixtures are written
to tests/fixtures/mock_chat/<key>.json, keyed by the rendered prompt, so changing the
prompt template or the redaction rules makes them stale (re-record). Commit them with a
note of the model and date; the eval report then says "N/65 replayed".
"""

from __future__ import annotations

import argparse
import json

from citewise.core.config import REPO_ROOT, get_settings
from citewise.evals.extract import load_brand_configs, load_golden
from citewise.providers import get_chat_provider
from citewise.providers.budget import BudgetedChatProvider, BudgetExceededError
from citewise.providers.mock import fixture_key
from citewise.schemas.extract import MentionExtraction
from citewise.services.extract.mentions import PURPOSE, stage_b_messages
from citewise.services.structured import complete_structured


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", default=str(REPO_ROOT / "evals/datasets/golden/mentions.jsonl"))
    ap.add_argument("--max-cost", type=float, default=0.10)
    args = ap.parse_args()

    settings = get_settings()
    if settings.is_mock:
        raise SystemExit("Set CITEWISE_MODE=live: recording needs the real model.")
    chat = BudgetedChatProvider(get_chat_provider("openai"), args.max_cost)
    brands = load_brand_configs(settings.brands_dir)
    out_dir = settings.fixtures_dir / "mock_chat"
    written = failed = 0
    for row in load_golden(REPO_ROOT / args.golden):
        cfg = brands[row["brand"]]
        msgs = stage_b_messages(brand=cfg["name"], aliases=cfg.get("aliases", []),
                                competitors=cfg.get("competitors", []),
                                category=cfg["category"], text=row["answer"])  # fmt: skip
        try:
            out = complete_structured(chat, msgs, MentionExtraction, model=settings.model_extract,
                                      purpose=PURPOSE, raise_on_budget=True)  # fmt: skip
        except BudgetExceededError:
            print(f"budget reached after {written} rows; re-run with a higher --max-cost")
            break
        if out is None:
            failed += 1
            continue
        parsed = out.model_dump()
        recorded = {"model": settings.model_extract, "golden_id": row["id"]}
        payload = {"text": json.dumps(parsed), "parsed": parsed, "recorded": recorded}
        path = out_dir / f"{fixture_key(PURPOSE, msgs)}.json"
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    print(f"recorded {written} responses ({failed} failed validation), "
          f"spent ~${chat.spent_usd:.4f}, in {out_dir}")  # fmt: skip


if __name__ == "__main__":
    main()
