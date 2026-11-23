"""Intent classification: an LLM zero-shot classifier and a cheap embedding classifier.

The LLM labels are the teacher. The embedding classifier (logistic regression on the
prompt embeddings) is trained on them and is the cheap path for new prompts. `agreement`
measures how often the cheap path matches the teacher, using cross-validated predictions
so no prompt is scored by a model that saw its label.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sqlalchemy.orm import Session

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.panel import Prompt
from citewise.providers.base import ChatProvider, EmbeddingProvider
from citewise.schemas.panel import INTENTS, IntentBatch
from citewise.services.panel.common import as_matrix
from citewise.services.prompt_library import render
from citewise.services.structured import complete_structured

log = logging.getLogger(__name__)

CLASSIFY_TEMPLATE = "panel/classify_intent"
MIN_TRAIN_PER_CLASS = 2


class IntentClassifier(Protocol):
    name: str

    def predict(self, texts: list[str]) -> list[str | None]:
        """One intent per text; None where the classifier couldn't decide."""
        ...


class LLMIntentClassifier:
    """Zero-shot, batched. Prompts are sent as JSON data with local ids, never as text."""

    name = "llm"

    def __init__(
        self,
        chat: ChatProvider,
        *,
        model: str,
        category: str,
        batch_size: int = 25,
        version: str = "v1",
    ) -> None:
        self.chat = chat
        self.model = model
        self.category = category
        self.batch_size = batch_size
        self.version = version

    def _predict_batch(self, texts: list[str]) -> list[str | None]:
        items = [{"id": i + 1, "text": t} for i, t in enumerate(texts)]
        prompt = render(CLASSIFY_TEMPLATE, self.version, category=self.category, items=items)
        batch = complete_structured(
            self.chat, prompt.messages, IntentBatch, model=self.model, purpose="panel.classify"
        )
        if batch is None:
            return [None] * len(texts)
        by_id = {label.id: label.intent for label in batch.labels}
        missing = len(texts) - sum(1 for i in range(1, len(texts) + 1) if i in by_id)
        if missing:
            log.warning("panel.classify: %d of %d items came back unlabeled", missing, len(texts))
        return [by_id.get(i + 1) for i in range(len(texts))]

    def predict(self, texts: list[str]) -> list[str | None]:
        out: list[str | None] = []
        for start in range(0, len(texts), self.batch_size):
            out += self._predict_batch(texts[start : start + self.batch_size])
        return out


def _logreg(seed: int = 0) -> LogisticRegression:
    # Balanced weights: generated panels are roughly even, imported ones often aren't.
    return LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced", random_state=seed)


class EmbeddingIntentClassifier:
    """Logistic regression on embeddings. Costs one embedding call per new prompt."""

    name = "embedding"

    def __init__(self, embedder: EmbeddingProvider | None = None, seed: int = 0) -> None:
        self.embedder = embedder
        self.model = _logreg(seed)
        self.fitted = False

    def fit(self, x: np.ndarray, labels: list[str]) -> EmbeddingIntentClassifier:
        self.model.fit(x, labels)
        self.fitted = True
        return self

    def predict_vectors(self, x: np.ndarray) -> list[str | None]:
        if not self.fitted:
            raise RuntimeError("EmbeddingIntentClassifier.fit() must be called first")
        return [str(v) for v in self.model.predict(x)]

    def predict(self, texts: list[str]) -> list[str | None]:
        if self.embedder is None:
            raise RuntimeError("predict(texts) needs an embedder; use predict_vectors()")
        return self.predict_vectors(as_matrix(self.embedder.embed(texts, purpose="panel.embed")))


@dataclass
class Agreement:
    n: int
    accuracy: float
    macro_f1: float
    labels: list[str]
    confusion: list[list[int]]  # rows = reference, cols = predicted, in `labels` order


def score(reference: list[str], predicted: list[str]) -> Agreement:
    labels = [i for i in INTENTS if i in set(reference) | set(predicted)]
    return Agreement(
        n=len(reference),
        accuracy=float(accuracy_score(reference, predicted)),
        macro_f1=float(f1_score(reference, predicted, labels=labels, average="macro")),
        labels=labels,
        confusion=confusion_matrix(reference, predicted, labels=labels).tolist(),
    )


def cross_validated_agreement(
    x: np.ndarray, teacher: list[str], folds: int = 5, seed: int = 0
) -> Agreement:
    """Score the embedding classifier against teacher labels with stratified k-fold CV."""
    smallest = min(Counter(teacher).values())
    if len(set(teacher)) < 2 or smallest < MIN_TRAIN_PER_CLASS:
        raise ValueError(
            "Need at least 2 intents with 2+ labeled prompts each to evaluate; "
            "run `citewise panel cluster` first."
        )
    cv = StratifiedKFold(n_splits=min(folds, smallest), shuffle=True, random_state=seed)
    predicted = cross_val_predict(_logreg(seed), x, teacher, cv=cv)
    return score(teacher, [str(p) for p in predicted])


def format_agreement(a: Agreement, title: str) -> str:
    abbrev = {i: i[:5] for i in a.labels}
    width = max(len(i) for i in a.labels)
    lines = [
        title,
        f"  n={a.n}  accuracy={a.accuracy:.3f}  macro-F1={a.macro_f1:.3f}",
        "  confusion matrix (rows = reference, cols = predicted):",
        "  " + " " * width + " " + " ".join(f"{abbrev[i]:>6}" for i in a.labels),
    ]
    for label, row in zip(a.labels, a.confusion, strict=True):
        lines.append(f"  {label:>{width}} " + " ".join(f"{v:>6}" for v in row))
    return "\n".join(lines)


def ensure_intents(
    session: Session, brand: Brand, prompts: list[Prompt], chat: ChatProvider, settings: Settings
) -> dict[str, int]:
    """Label every prompt that has no intent yet. Returns counts by intent_source.

    Order of preference: the LLM classifier; for items it couldn't label, the embedding
    classifier trained on everything labeled so far; last resort, the intent the
    generator was asked for. Already-labeled prompts are never re-sent, so re-running
    `panel cluster` costs nothing extra.
    """
    todo = [p for p in prompts if p.intent is None]
    if todo:
        llm = LLMIntentClassifier(
            chat,
            model=settings.model_extract,
            category=brand.category,
            batch_size=settings.panel_classify_batch,
        )
        for p, intent in zip(todo, llm.predict([p.text for p in todo]), strict=True):
            if intent is not None:
                p.intent, p.intent_source = intent, "llm"

    still = [p for p in prompts if p.intent is None]
    labeled = [p for p in prompts if p.intent is not None and p.embedding]
    counts = Counter(p.intent for p in labeled)
    if still and len(counts) >= 2 and min(counts.values()) >= MIN_TRAIN_PER_CLASS:
        clf = EmbeddingIntentClassifier().fit(
            as_matrix([p.embedding for p in labeled if p.embedding]),
            [p.intent for p in labeled if p.intent],
        )
        with_vec = [p for p in still if p.embedding]
        if with_vec:
            preds = clf.predict_vectors(as_matrix([p.embedding for p in with_vec if p.embedding]))
            for p, intent in zip(with_vec, preds, strict=True):
                p.intent, p.intent_source = intent, "embedding"
    for p in prompts:
        if p.intent is None:
            p.intent, p.intent_source = p.requested_intent or "informational", "fallback"
    return dict(Counter(p.intent_source or "none" for p in prompts))
