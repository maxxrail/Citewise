"""Near-duplicate removal by cosine similarity of embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DedupResult:
    keep: list[int]  # indices into the candidate list, ascending
    # dropped candidate index -> what it duplicated: ("candidate", idx) or ("existing", idx)
    dropped: dict[int, tuple[str, int]] = field(default_factory=dict)


def dedup(
    texts: list[str],
    vectors: np.ndarray,
    threshold: float,
    existing: np.ndarray | None = None,
) -> DedupResult:
    """Greedy dedup: drop a candidate whose cosine to anything already kept is > threshold.

    Candidates are visited shortest-first, so of a near-duplicate pair the shorter (usually
    the more natural, typed-by-a-human) phrasing survives. `existing` holds vectors already
    stored for the brand; they always win. Rows must be unit length.
    O(n * kept): fine for panels of thousands.
    """
    n = len(texts)
    if n == 0:
        return DedupResult(keep=[])
    if vectors.shape[0] != n:
        raise ValueError("texts and vectors must have the same length")

    n_existing = 0 if existing is None else existing.shape[0]
    kept = np.empty((n_existing + n, vectors.shape[1]))
    if existing is not None and n_existing:
        kept[:n_existing] = existing
    kept_ids: list[tuple[str, int]] = [("existing", i) for i in range(n_existing)]

    result = DedupResult(keep=[])
    for i in sorted(range(n), key=lambda j: (len(texts[j]), texts[j])):
        count = len(kept_ids)
        if count:
            sims = kept[:count] @ vectors[i]
            best = int(np.argmax(sims))
            if sims[best] > threshold:
                result.dropped[i] = kept_ids[best]
                continue
        kept[count] = vectors[i]
        kept_ids.append(("candidate", i))
        result.keep.append(i)
    result.keep.sort()
    return result
