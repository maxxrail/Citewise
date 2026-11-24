from __future__ import annotations

import pytest

from citewise.core.config import Settings


def test_defaults_are_mock_and_have_model_ids() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.is_mock
    # Claude for Citewise's own calls, a local embedder (Anthropic has no embeddings API).
    assert (s.llm_provider, s.embed_provider) == ("anthropic", "local")
    assert s.model_embed == "BAAI/bge-small-en-v1.5"
    assert all(m.startswith("claude-") for m in (s.model_extract, s.model_judge, s.model_generate))
    assert all([s.model_extract, s.model_judge, s.model_generate])
    assert s.max_cost_usd_per_run == 3.0


def test_blank_env_values_fall_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # A freshly copied .env.example has `MODEL_EXTRACT=` and `OPENAI_API_KEY=`.
    monkeypatch.setenv("MODEL_EXTRACT", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.model_extract == Settings.model_fields["model_extract"].default
    assert s.openai_api_key is None


def test_secrets_are_not_printed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert "sk-test-not-real" not in repr(s)
    assert s.openai_api_key is not None
    assert s.openai_api_key.get_secret_value() == "sk-test-not-real"


def test_invalid_mode_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CITEWISE_MODE", "prod")
    with pytest.raises(ValueError):
        Settings(_env_file=None)  # type: ignore[call-arg]
