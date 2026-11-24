"""Sentiment narrative: per engine, the most common positive and negative themes in how it
describes the brand.

Input is Module 3's brand mentions (snippet + sentiment). Positive = sentiment > 0.15;
negative = sentiment < -0.15 or labelled mixed (a mixed snippet carries a criticism). Each
side is embedded and clustered with KMeans (k <= 3). Each cluster is labelled by the LLM
from up to 8 of its snippets, and clusters are reported largest first, with the snippet
nearest the centroid as the example.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import KMeans
from sqlalchemy import select
from sqlalchemy.orm import Session

from citewise.models.collect import Answer, Run
from citewise.models.extract import Mention
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.schemas.factcheck import ThemeLabel
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

PURPOSE = "narrative.theme"
POS, NEG = 0.15, -0.15


@dataclass
class Theme:
    label: str
    count: int
    share: float  # of this engine's snippets on this side
    example: str


@dataclass
class EngineNarrative:
    engine: str
    mentions: int
    positive: list[Theme] = field(default_factory=list)
    negative: list[Theme] = field(default_factory=list)


def _themes(
    snippets: list[str],
    polarity: str,
    brand: str,
    embedder: EmbeddingProvider,
    chat: ChatProvider,
    model: str,
    k: int,
) -> list[Theme]:
    if not snippets:
        return []
    vecs = np.array(embedder.embed(snippets, purpose="narrative.embed"), dtype=float)
    n = min(k, len(set(snippets)))
    labels = (
        KMeans(n_clusters=n, n_init=10, random_state=0).fit_predict(vecs)
        if n > 1
        else np.zeros(len(snippets), dtype=int)
    )
    out = []
    for c in range(n):
        idx = [i for i, lab in enumerate(labels) if lab == c]
        centroid = vecs[idx].mean(axis=0)
        nearest = min(idx, key=lambda i: float(np.linalg.norm(vecs[i] - centroid)))
        sample = list(dict.fromkeys(snippets[i] for i in idx))[:8]
        rendered = render("factcheck/theme", "v1", brand=brand, polarity=polarity, snippets=sample)
        lab = complete_structured(chat, rendered.messages, ThemeLabel, model=model, purpose=PURPOSE)
        out.append(Theme(lab.label if lab else sample[0][:60], len(idx), len(idx) / len(snippets),
                         snippets[nearest]))  # fmt: skip
    return sorted(out, key=lambda t: -t.count)


def narratives(
    session: Session,
    brand_id: int,
    brand_name: str,
    run_id: int,
    embedder: EmbeddingProvider,
    chat: ChatProvider,
    model: str,
    *,
    k: int = 3,
) -> list[EngineNarrative]:
    if session.get(Run, run_id) is None:
        raise ValueError(f"No run {run_id}")
    rows = session.execute(
        select(Answer.engine, Mention.snippet, Mention.sentiment, Mention.sentiment_label)
        .join(Mention, Mention.answer_id == Answer.id)
        .where(Answer.run_id == run_id, Mention.is_brand, Mention.snippet != "")
    ).all()
    by_engine: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
    for engine, snippet, sentiment, label in rows:
        by_engine[engine].append((snippet, float(sentiment), label))
    out = []
    for engine in sorted(by_engine):
        items = by_engine[engine]
        pos = [s for s, v, _ in items if v > POS]
        neg = [s for s, v, lab in items if v < NEG or lab == "mixed"]
        out.append(
            EngineNarrative(
                engine, len(items),
                _themes(pos, "positive", brand_name, embedder, chat, model, k),
                _themes(neg, "negative", brand_name, embedder, chat, model, k),
            )
        )  # fmt: skip
    return out
