"""Stratified panel selection across topics x intents.

The scanned panel is a small sample of the full prompt set (scans cost money per prompt
per engine). It should cover as many topics as possible and keep the topic/intent mix
roughly proportional to the full set.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

from sqlalchemy import update
from sqlalchemy.orm import Session

from citewise.models.brand import Brand
from citewise.models.panel import Prompt, Topic
from citewise.services.panel.common import brand_prompts


@dataclass(frozen=True)
class PanelItem:
    id: int
    topic_id: int | None
    intent: str | None


@dataclass
class PanelReport:
    selected: list[int]
    topics_total: int
    topics_covered: int
    intents: dict[str, int]

    @property
    def topic_coverage(self) -> float:
        return self.topics_covered / self.topics_total if self.topics_total else 1.0


def _proportional(sizes: dict[tuple, int], quota: int) -> dict[tuple, int]:
    """Split `quota` across cells in proportion to `sizes` (Sainte-Laguë), capped at each size.

    One seat at a time to the cell with the highest size / (2 * seats + 1); ties go to the
    bigger cell, then the cell key, so the result is deterministic.
    """
    alloc = dict.fromkeys(sizes, 0)
    for _ in range(quota):
        open_cells = [c for c, s in sizes.items() if alloc[c] < s]
        if not open_cells:
            break
        best = max(open_cells, key=lambda c: (sizes[c] / (2 * alloc[c] + 1), sizes[c], str(c)))
        alloc[best] += 1
    return alloc


def stratified_sample(items: list[PanelItem], size: int, seed: int = 0) -> list[int]:
    """Pick `size` ids: first one per topic (biggest topics first, spreading intents),
    then the remainder in proportion to each (topic, intent) cell. Deterministic per seed.
    """
    if size <= 0:
        return []
    if size >= len(items):
        return sorted(i.id for i in items)

    rng = random.Random(seed)
    cells: dict[tuple, list[int]] = {}
    for item in sorted(items, key=lambda i: i.id):
        cells.setdefault((item.topic_id, item.intent), []).append(item.id)
    for ids in cells.values():
        rng.shuffle(ids)

    topic_sizes = Counter(item.topic_id for item in items)
    picked: list[int] = []
    taken: Counter[tuple] = Counter()
    intent_counts: Counter[str | None] = Counter()

    # Phase 1: coverage. One prompt per topic, choosing the intent least represented so far.
    for topic, _ in sorted(topic_sizes.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        if len(picked) >= size:
            break
        options = [c for c in cells if c[0] == topic]
        cell = min(options, key=lambda c: (intent_counts[c[1]], -len(cells[c]), str(c[1])))
        picked.append(cells[cell][0])
        taken[cell] += 1
        intent_counts[cell[1]] += 1

    # Phase 2: proportional fill from what's left in each cell.
    remaining = {c: len(ids) - taken[c] for c, ids in cells.items()}
    alloc = _proportional(remaining, size - len(picked))
    for cell, n in alloc.items():
        start = taken[cell]
        picked += cells[cell][start : start + n]
    return sorted(picked)


def build_panel(session: Session, brand: Brand, size: int, seed: int = 0) -> PanelReport:
    """Mark the selected prompts `in_panel` (and unmark the rest). Re-running replaces it."""
    prompts = brand_prompts(session, brand.id)
    if not prompts:
        raise ValueError(f"{brand.name} has no prompts. Run `citewise panel generate`.")
    unclustered = sum(1 for p in prompts if p.topic_id is None or p.intent is None)
    if unclustered:
        raise ValueError(
            f"{unclustered} {brand.name} prompts have no topic or intent yet "
            "(new imports or generations?). Run `citewise panel cluster` first."
        )
    items = [PanelItem(p.id, p.topic_id, p.intent) for p in prompts]
    selected = stratified_sample(items, size, seed)
    chosen = set(selected)

    session.execute(update(Prompt).where(Prompt.brand_id == brand.id).values(in_panel=False))
    for p in prompts:
        p.in_panel = p.id in chosen
    session.flush()

    topics_total = session.query(Topic).filter(Topic.brand_id == brand.id).count()
    covered = {p.topic_id for p in prompts if p.id in chosen and p.topic_id is not None}
    return PanelReport(
        selected=selected,
        topics_total=topics_total,
        topics_covered=len(covered),
        intents=dict(Counter(p.intent or "none" for p in prompts if p.id in chosen)),
    )
