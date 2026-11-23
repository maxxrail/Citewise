"""Unit tests for the panel's pure logic: dedup thresholds, k-selection, stratified sampling."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_blobs

from citewise.services.panel.cluster import (
    centrality,
    hdbscan_clusters,
    kmeans_clusters,
    reduce_dims,
    select_k,
)
from citewise.services.panel.common import clean_prompt_text, text_hash
from citewise.services.panel.dedup import dedup
from citewise.services.panel.select import PanelItem, _proportional, stratified_sample


def _unit(*rows: list[float]) -> np.ndarray:
    x = np.asarray(rows, dtype=float)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _pair_with_cos(cos: float) -> np.ndarray:
    """Two unit vectors whose cosine is exactly `cos`."""
    return _unit([1.0, 0.0], [cos, float(np.sqrt(1 - cos**2))])


# ---- dedup ----


@pytest.mark.parametrize(
    ("cos", "threshold", "dropped"),
    [
        (0.95, 0.92, True),  # above threshold -> near duplicate
        (0.90, 0.92, False),  # below -> distinct
        (0.92, 0.92, False),  # strictly greater-than: equal is kept
        (0.95, 0.96, False),  # raising the threshold keeps more
        (0.95, 0.80, True),  # lowering it drops more
    ],
)
def test_dedup_threshold_behaviour(cos: float, threshold: float, dropped: bool) -> None:
    result = dedup(["best shoes", "best shoes for me"], _pair_with_cos(cos), threshold)
    assert (len(result.keep) == 1) is dropped


def test_dedup_keeps_shortest_phrasing_regardless_of_order() -> None:
    texts = ["best trail running shoes for mud?", "best trail shoes for mud"]
    result = dedup(texts, _pair_with_cos(0.99), 0.92)
    assert result.keep == [1]
    assert result.dropped == {0: ("candidate", 1)}


def test_dedup_existing_prompts_always_win() -> None:
    existing = _unit([1.0, 0.0])
    result = dedup(["x"], _unit([0.99, 0.141]), 0.92, existing=existing)
    assert result.keep == []
    assert result.dropped == {0: ("existing", 0)}


def test_dedup_chain_is_greedy_not_transitive() -> None:
    """a~b and b~c but a!~c: keeping a drops b, and c survives (it never meets b)."""
    a, b, c = 0.0, 0.35, 0.7  # angles in radians; cos(0.35)=0.94, cos(0.7)=0.76
    vecs = np.array([[np.cos(t), np.sin(t)] for t in (a, b, c)])
    result = dedup(["aa", "bbb", "cccc"], vecs, 0.92)
    assert result.keep == [0, 2]


def test_dedup_empty_and_mismatched() -> None:
    assert dedup([], np.zeros((0, 2)), 0.9).keep == []
    with pytest.raises(ValueError):
        dedup(["a", "b"], _unit([1.0, 0.0]), 0.9)


def test_text_normalisation() -> None:
    assert text_hash("Best Shoes?") == text_hash("best  shoes")
    assert text_hash("best shoes") != text_hash("best shoe")
    assert clean_prompt_text("  two\nlines\t here ") == "two lines here"
    assert clean_prompt_text("   ") is None
    assert clean_prompt_text("x" * 501) is None


# ---- k-selection and clustering ----


def _blobs(k: int, n: int = 240, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    x, y = make_blobs(n_samples=n, centers=k, n_features=12, cluster_std=0.6, random_state=seed)
    return x / np.linalg.norm(x, axis=1, keepdims=True), y


@pytest.mark.parametrize("true_k", [3, 5, 7])
def test_select_k_recovers_true_number_of_clusters(true_k: int) -> None:
    x, _ = _blobs(true_k)
    k, scores = select_k(x, 2, 10)
    assert k == true_k
    assert set(scores) == set(range(2, 11))


def test_select_k_clips_range_to_data() -> None:
    x, _ = _blobs(2, n=6)
    _, scores = select_k(x, 4, 16)
    assert max(scores) <= 5  # silhouette needs k <= n - 1
    assert select_k(x[:2], 2, 5) == (1, {})


def test_kmeans_is_deterministic_and_labels_every_row() -> None:
    x, _ = _blobs(4)
    a, b = kmeans_clusters(x, 2, 8), kmeans_clusters(x, 2, 8)
    assert a.k == 4
    assert np.array_equal(a.labels, b.labels)
    assert set(a.labels) == set(range(4))


def test_hdbscan_assigns_noise_to_a_cluster() -> None:
    x, _ = _blobs(4)
    noisy = np.vstack([x, _unit(*np.random.default_rng(0).normal(size=(10, 12)).tolist())])
    result = hdbscan_clusters(noisy, min_cluster_size=10)
    assert result.k >= 2
    assert (result.labels >= 0).all()
    assert len(result.labels) == noisy.shape[0]


def test_reduce_dims_and_centrality() -> None:
    x, _ = _blobs(3)
    z = reduce_dims(x, 5)
    assert z.shape == (x.shape[0], 5)
    assert np.allclose(np.linalg.norm(z, axis=1), 1.0)
    assert reduce_dims(x, 0) is x
    labels = kmeans_clusters(z, 2, 6).labels
    cent = centrality(x, labels, int(labels.max()) + 1)
    assert ((cent > 0.5) & (cent <= 1.0 + 1e-9)).all()  # tight blobs: close to centroid


# ---- stratified sampling ----


def _items() -> list[PanelItem]:
    """3 topics of very different sizes (60/30/10) x 5 intents, ids 1..100."""
    intents = ["informational", "commercial", "comparison", "navigational", "transactional"]
    items, next_id = [], 1
    for topic, size in ((1, 60), (2, 30), (3, 10)):
        for i in range(size):
            items.append(PanelItem(next_id, topic, intents[i % 5]))
            next_id += 1
    return items


def test_sample_has_exact_size_and_unique_ids() -> None:
    picked = stratified_sample(_items(), 20)
    assert len(picked) == 20 == len(set(picked))


def test_sample_covers_every_topic_even_small_ones() -> None:
    items = _items()
    topic_of = {i.id: i.topic_id for i in items}
    assert {topic_of[i] for i in stratified_sample(items, 5)} == {1, 2, 3}


def test_sample_is_proportional_to_topic_and_intent_mix() -> None:
    items = _items()
    topic_of = {i.id: i.topic_id for i in items}
    intent_of = {i.id: i.intent for i in items}
    picked = stratified_sample(items, 50)
    per_topic = {t: sum(topic_of[i] == t for i in picked) for t in (1, 2, 3)}
    assert per_topic == {1: 30, 2: 15, 3: 5}
    per_intent = {it: sum(intent_of[i] == it for i in picked) for it in set(intent_of.values())}
    assert set(per_intent.values()) == {10}


def test_sample_is_deterministic_per_seed() -> None:
    items = _items()
    assert stratified_sample(items, 25, seed=1) == stratified_sample(items, 25, seed=1)
    assert stratified_sample(items, 25, seed=1) != stratified_sample(items, 25, seed=2)


def test_sample_edge_sizes() -> None:
    items = _items()
    assert stratified_sample(items, 0) == []
    assert stratified_sample(items, 500) == sorted(i.id for i in items)
    two = stratified_sample(items, 2)  # fewer slots than topics: biggest topics first
    assert {next(i.topic_id for i in items if i.id == p) for p in two} == {1, 2}


def test_proportional_allocation_respects_capacity() -> None:
    alloc = _proportional({("a",): 1, ("b",): 9}, 8)
    assert alloc == {("a",): 1, ("b",): 7}
    assert _proportional({("a",): 2}, 5) == {("a",): 2}
