"""Module 1 review fixes: embedding provenance, panel reset, UTF-8 console output."""

from __future__ import annotations

import os
import subprocess
import sys

import pytest
from sqlalchemy.orm import Session, sessionmaker
from typer.testing import CliRunner

from citewise.core.config import Settings
from citewise.models.panel import Prompt
from citewise.providers import get_chat_provider, get_embedder
from citewise.services.brands import seed_brands
from citewise.services.panel.cluster import cluster_brand
from citewise.services.panel.common import brand_prompts, resolve_brand
from citewise.services.panel.generate import Candidate, refresh_embeddings, store_candidates


class OtherSpaceEmbedder:
    """Stands in for a different vector space (e.g. live OpenAI vs the mock)."""

    ident = "other:some-model"

    def embed(self, texts: list[str], *, purpose: str = "embed") -> list[list[float]]:
        # 3 dims (would crash a 1536-dim matmul); one axis per text so they aren't duplicates.
        return [[float(i == j) for j in range(3)] for i in range(len(texts))]


def _cands(*texts: str) -> list[Candidate]:
    return [Candidate(text=t, source="import") for t in texts]


def test_tracked_embedder_ident_names_the_vector_space(
    settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    embedder = get_embedder(settings=settings, session_factory=session_factory)
    assert embedder.ident == f"mock:{settings.model_embed}"


def test_store_re_embeds_prompts_from_another_vector_space(
    db: Session, settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed_brands(db, settings.brands_dir)
    brand = resolve_brand(db, "Stridelab")
    store_candidates(
        db, brand, _cands("best trail shoes", "how to lace shoes"), OtherSpaceEmbedder(), 0.92
    )

    mock = get_embedder(settings=settings, session_factory=session_factory)
    report = store_candidates(db, brand, _cands("stridelab promo code"), mock, 0.92)

    # Without the refresh this would matmul 3-dim stored vectors with 1536-dim new ones.
    assert report.reembedded == 2 and report.stored == 1
    prompts = brand_prompts(db, brand.id)
    assert {p.embedding_model for p in prompts} == {mock.ident}
    assert {len(p.embedding or []) for p in prompts} == {1536}
    assert refresh_embeddings(prompts, mock) == 0  # nothing stale any more


def test_rows_without_provenance_are_treated_as_stale(
    db: Session, settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    """Rows from before migration 0003 have embedding_model NULL."""
    seed_brands(db, settings.brands_dir)
    brand = resolve_brand(db, "Stridelab")
    db.add(
        Prompt(brand_id=brand.id, text="old row", text_hash="h", source="import", embedding=[0.0])
    )
    db.flush()
    mock = get_embedder(settings=settings, session_factory=session_factory)
    assert refresh_embeddings(brand_prompts(db, brand.id), mock) == 1


def test_cluster_refuses_mixed_vector_spaces_unless_given_an_embedder(
    db: Session, settings: Settings, session_factory: sessionmaker[Session]
) -> None:
    seed_brands(db, settings.brands_dir)
    brand = resolve_brand(db, "Stridelab")
    mock = get_embedder(settings=settings, session_factory=session_factory)
    texts = [f"best {w} running shoes" for w in ("trail", "road", "track", "marathon", "budget")]
    store_candidates(db, brand, _cands(*texts), mock, 0.92)
    stray = brand_prompts(db, brand.id)[0]
    stray.embedding_model = "other:some-model"
    chat = get_chat_provider(settings=settings, session_factory=session_factory)

    with pytest.raises(ValueError, match="different embedders"):
        cluster_brand(db, brand, chat, settings)
    report = cluster_brand(db, brand, chat, settings, embedder=mock)
    assert sum(size for _, size, _ in report.topics) == 5
    assert stray.embedding_model == mock.ident


def test_panel_reset_deletes_only_that_brands_panel(cli_db: str) -> None:
    from citewise.cli import app

    runner = CliRunner()
    assert runner.invoke(app, ["seed"]).exit_code == 0
    for brand in ("Stridelab", "Taskflow"):
        assert runner.invoke(app, ["panel", "generate", brand]).exit_code == 0

    declined = runner.invoke(app, ["panel", "reset", "Stridelab"], input="n\n")
    assert declined.exit_code == 0 and "Nothing deleted" in declined.output

    out = runner.invoke(app, ["panel", "reset", "Stridelab", "--yes"])
    assert out.exit_code == 0 and "deleted 272 prompts" in out.output

    from citewise.core.db import get_sessionmaker

    with get_sessionmaker()() as s:
        assert brand_prompts(s, resolve_brand(s, "Stridelab").id) == []
        assert len(brand_prompts(s, resolve_brand(s, "Taskflow").id)) == 276

    regen = runner.invoke(app, ["panel", "generate", "Stridelab"])
    assert "0 already generated" in regen.output and "stored: 272 new" in regen.output


def test_cli_output_survives_non_latin_text_when_piped() -> None:
    """On Windows a pipe defaults to cp1252; printing Japanese or emoji used to crash."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    code = "from citewise.cli import _utf8_console; _utf8_console(); print('ランニング 👟')"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, env=env, timeout=60, check=False
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    assert result.stdout.decode("utf-8").strip() == "ランニング 👟"
