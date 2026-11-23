"""Cluster a brand's prompts into topics, label them, and make sure every prompt has an intent.

KMeans with k picked by silhouette (cosine) is the default. HDBSCAN (scikit-learn's
built-in implementation) is behind `algorithm="hdbscan"`; its noise points are assigned
to the nearest cluster so every prompt still gets a topic.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sqlalchemy import delete, update
from sqlalchemy.orm import Session

from citewise.core.config import Settings
from citewise.models.brand import Brand
from citewise.models.panel import Prompt, Topic
from citewise.providers.base import ChatProvider
from citewise.services.panel.common import as_matrix, brand_prompts
from citewise.services.panel.intent import ensure_intents
from citewise.services.panel.label import CENTRAL_PROMPTS, label_cluster

log = logging.getLogger(__name__)


@dataclass
class Clustering:
    labels: np.ndarray  # cluster index per row, 0..k-1
    k: int
    algorithm: str
    silhouette_by_k: dict[int, float] = field(default_factory=dict)


@dataclass
class ClusterReport:
    k: int
    algorithm: str
    silhouette_by_k: dict[int, float]
    topics: list[tuple[str, int, str]]  # (label, size, label_source)
    intents: dict[str, int]
    intent_sources: dict[str, int]


def select_k(
    x: np.ndarray, k_min: int, k_max: int, seed: int = 0, sample: int = 2000
) -> tuple[int, dict[int, float]]:
    """Pick k in [k_min, k_max] with the highest cosine silhouette (ties -> smaller k).

    The range is clipped to what the data allows (silhouette needs 2 <= k <= n - 1).
    """
    n = x.shape[0]
    hi = min(k_max, n - 1)
    lo = max(2, min(k_min, hi))
    if n < 3 or hi < 2:
        return 1, {}
    scores: dict[int, float] = {}
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
        if len(set(labels)) < 2:
            continue
        scores[k] = float(
            silhouette_score(
                x, labels, metric="cosine", sample_size=min(n, sample), random_state=seed
            )
        )
    if not scores:
        return 1, {}
    best = max(scores, key=lambda k: (round(scores[k], 6), -k))
    return best, scores


def kmeans_clusters(
    x: np.ndarray, k_min: int, k_max: int, seed: int = 0, sample: int = 2000
) -> Clustering:
    k, scores = select_k(x, k_min, k_max, seed, sample)
    if k == 1:
        return Clustering(np.zeros(x.shape[0], dtype=int), 1, "kmeans", scores)
    labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
    return Clustering(labels.astype(int), k, "kmeans", scores)


def hdbscan_clusters(x: np.ndarray, min_cluster_size: int | None = None) -> Clustering:
    """Density clustering; noise (-1) is attached to the nearest cluster centroid."""
    n = x.shape[0]
    size = min_cluster_size or max(5, n // 40)
    raw = HDBSCAN(min_cluster_size=size, metric="euclidean", copy=True).fit_predict(x)
    ids = sorted(set(raw) - {-1})
    if not ids:  # everything is noise: one topic
        return Clustering(np.zeros(n, dtype=int), 1, "hdbscan")
    remap = {old: new for new, old in enumerate(ids)}
    labels = np.array([remap.get(int(c), -1) for c in raw])
    cents = as_matrix([x[labels == c].mean(axis=0).tolist() for c in range(len(ids))])
    noise = labels == -1
    if noise.any():
        labels[noise] = np.argmax(x[noise] @ cents.T, axis=1)
    return Clustering(labels, len(ids), "hdbscan")


def reduce_dims(x: np.ndarray, dims: int) -> np.ndarray:
    """PCA to `dims` (then unit length) before clustering.

    In 1536 dimensions cosine distances concentrate, so silhouette barely separates k
    values and HDBSCAN finds no density. Full SVD keeps the projection deterministic.
    """
    if dims <= 0 or dims >= min(x.shape):
        return x
    z = PCA(n_components=dims, svd_solver="full").fit_transform(x)
    norms = np.linalg.norm(z, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return z / norms


def centrality(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Cosine of each row to its own cluster centroid (higher = more typical)."""
    out = np.zeros(x.shape[0])
    for c in range(k):
        members = labels == c
        if not members.any():
            continue
        centroid = x[members].mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        out[members] = x[members] @ centroid
    return out


def cluster_brand(
    session: Session,
    brand: Brand,
    chat: ChatProvider,
    settings: Settings,
    algorithm: str = "kmeans",
    seed: int = 0,
) -> ClusterReport:
    """Replace the brand's topics with a fresh clustering and fill in missing intents.

    Prompts keep their ids (Module 2 answers reference them); only topic links change.
    """
    prompts = [p for p in brand_prompts(session, brand.id) if p.embedding]
    if not prompts:
        raise ValueError(f"{brand.name} has no embedded prompts. Run `citewise panel generate`.")
    x = as_matrix([p.embedding for p in prompts if p.embedding])

    z = reduce_dims(x, settings.panel_pca_dims)
    if algorithm == "hdbscan":
        clustering = hdbscan_clusters(z)
    elif algorithm == "kmeans":
        clustering = kmeans_clusters(
            z, settings.panel_k_min, settings.panel_k_max, seed, settings.panel_silhouette_sample
        )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; use kmeans or hdbscan")
    cent = centrality(x, clustering.labels, clustering.k)  # in the original space

    groups: dict[int, list[int]] = {}
    for row, c in enumerate(clustering.labels):
        groups.setdefault(int(c), []).append(row)
    # Biggest topics first; ties broken by the oldest prompt so the order is stable.
    ordered = sorted(groups.values(), key=lambda rows: (-len(rows), prompts[rows[0]].id))

    # All LLM calls happen before any write, so no DB transaction (and lock) stays open
    # across minutes of model latency, and the call tracker's own session never waits on it.
    labels = []
    for rows in ordered:
        central = sorted(rows, key=lambda r: (-round(float(cent[r]), 9), prompts[r].id))
        texts = [prompts[r].text for r in central[:CENTRAL_PROMPTS]]
        labels.append(label_cluster(chat, brand.category, texts, settings.model_extract))
    sources = ensure_intents(session, brand, prompts, chat, settings)

    # Clear links explicitly: SQLite doesn't enforce ON DELETE SET NULL without a PRAGMA.
    # The old panel was stratified over the old topics, so it is cleared: re-run `build`.
    session.execute(
        update(Prompt).where(Prompt.brand_id == brand.id).values(topic_id=None, in_panel=False)
    )
    session.execute(delete(Topic).where(Topic.brand_id == brand.id))

    topics: list[tuple[str, int, str]] = []
    for rows, (label, source) in zip(ordered, labels, strict=True):
        topic = Topic(
            brand_id=brand.id,
            label=label.label,
            description=label.description,
            size=len(rows),
            algorithm=clustering.algorithm,
            label_source=source,
        )
        session.add(topic)
        session.flush()
        for r in rows:
            prompts[r].topic_id = topic.id
            prompts[r].centrality = float(cent[r])
        topics.append((topic.label, topic.size, source))
    session.flush()
    return ClusterReport(
        k=clustering.k,
        algorithm=clustering.algorithm,
        silhouette_by_k=clustering.silhouette_by_k,
        topics=topics,
        intents=dict(Counter(p.intent or "none" for p in prompts)),
        intent_sources=sources,
    )
