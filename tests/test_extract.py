"""Module 3 extraction: entity resolution, Stage A, merge, and idempotent `extract run`."""

from __future__ import annotations

from collections import Counter

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from citewise.core.config import Settings
from citewise.models.collect import Answer
from citewise.models.extract import AnswerExtraction
from citewise.models.extract import Mention as MentionRow
from citewise.providers import get_chat_provider
from citewise.providers.budget import BudgetedChatProvider
from citewise.schemas.extract import ExtractedMention
from citewise.services.extract.mentions import merge, stage_a, structural_ranks
from citewise.services.extract.resolve import Resolver, display_name, normalize
from citewise.services.extract.runner import EXTRACTOR_VERSION, extract_run

PM = Resolver(
    "Taskflow",
    ["TaskFlow", "Taskflow PM"],
    ["Asana", "Monday.com", "Trello", "ClickUp", "Notion", "Jira"],
)
SHOES = Resolver(
    "Stridelab",
    ["StrideLab", "Stride Lab"],
    ["Nike", "Hoka", "Brooks", "ASICS", "New Balance", "On"],
)


# ---------------------------------------------------------------- entity resolution


@pytest.mark.parametrize(
    ("surface", "key"),
    [
        ("Asana", "asana"),
        ("Asana Inc.", "asana"),
        ("Asana, Inc.", "asana"),
        ("ASANA LLC", "asana"),
        ("Hoka's", "hoka"),
        ("Hoka’s", "hoka"),
        ("Brooks'", "brooks"),
        ("HOKA", "hoka"),
        ("monday.com", "monday"),
        ("Monday.com Ltd", "monday"),
        ("New Balance Co.", "new balance"),
        ("Stride-Lab", "stride lab"),
        ("Ben & Jerry's", "ben and jerry"),
        ("Inc.", "inc"),  # a lone suffix is kept rather than emptied
    ],
)
def test_normalize(surface: str, key: str) -> None:
    assert normalize(surface) == key


@pytest.mark.parametrize(
    ("surface", "canonical", "typ", "how"),
    [
        ("Asana Inc.", "Asana", "competitor", "exact"),  # legal suffix
        ("asana's", "Asana", "competitor", "exact"),  # possessive + casing
        ("MONDAY.COM", "Monday.com", "competitor", "exact"),  # TLD + casing
        ("monday", "Monday.com", "competitor", "exact"),
        ("Taskflow PM", "Taskflow", "brand", "exact"),  # alias -> brand
        ("taskflow", "Taskflow", "brand", "exact"),
        ("ClickUp Brain", "ClickUp", "competitor", "product"),  # product name -> maker
        ("Clickup", "ClickUp", "competitor", "exact"),
        ("Asanna", "Asana", "competitor", "fuzzy"),  # typo
        ("Taskflw", "Taskflow", "brand", "fuzzy"),
        ("Wrike, Inc.", "Wrike", "unknown", "unknown"),  # emerging competitor
    ],
)
def test_resolve_known_and_unknown(surface: str, canonical: str, typ: str, how: str) -> None:
    r = PM.resolve(surface)
    assert (r.canonical, r.entity_type, r.how) == (canonical, typ, how)


def test_resolve_products_and_ambiguous_names() -> None:
    assert SHOES.resolve("Nike Pegasus 41").canonical == "Nike"
    assert SHOES.resolve("New Balance Fresh Foam 1080").canonical == "New Balance"
    assert SHOES.resolve("On Cloudmonster").canonical == "On"
    assert SHOES.resolve("Stridelab Tempo").entity_type == "brand"
    # Short keys never fuzzy-match: "Hook" is not Hoka.
    assert SHOES.resolve("Hook").entity_type == "unknown"


def test_unknowns_group_by_key() -> None:
    a, b = PM.resolve("Wrike"), PM.resolve("Wrike Inc.")
    assert a.key == b.key == "wrike" and display_name("Wrike, Inc.'s") == "Wrike"


# ---------------------------------------------------------------- Stage A


def test_stage_a_finds_aliases_with_rank_and_snippet() -> None:
    text = (
        "Here are some of the best options:\n"
        "1. **Asana**: polished and widely adopted.\n"
        "2. **TaskFlow Boards**: clean interface. Many teams compare it with Asana.\n"
        "3. **Clickup**: packs many features into one app.\n"
        "Jira is the standard for software teams."
    )
    got = {m.canonical: m for m in stage_a(text, PM)}
    assert set(got) == {"Asana", "Taskflow", "ClickUp", "Jira"}
    assert (got["Asana"].rank, got["Taskflow"].rank, got["ClickUp"].rank) == (1, 2, 3)
    assert got["Jira"].rank is None  # prose, not a list item
    assert got["Taskflow"].entity == "TaskFlow" and got["Taskflow"].entity_type == "brand"
    assert got["Asana"].snippet.startswith("1. **Asana**")
    assert len(got["Asana"].positions) == 2


def test_stage_a_ambiguous_words_are_not_brands() -> None:
    text = (
        "On the other hand, the On Cloudmonster is firm. Runners also like On.\n"
        "We meet on Monday to plan. Monday is also a popular PM tool.\n"
        "I have a notion that Notion works for docs."
    )
    shoes = {m.canonical: len(m.positions) for m in stage_a(text, SHOES)}
    assert shoes == {"On": 2}  # "On Cloudmonster", "like On." but not "On the other hand"
    pm = {m.canonical: len(m.positions) for m in stage_a(text, PM)}
    assert pm == {"Monday.com": 1, "Notion": 1}  # not "on Monday", not "a notion"


def test_stage_a_fuzzy_needs_a_capitalized_long_name() -> None:
    text = "Asanna and Stridelabs are fine. The brook was cold."
    assert {m.canonical for m in stage_a(text, PM)} == {"Asana"}
    assert {m.canonical for m in stage_a(text, SHOES)} == {"Stridelab"}  # not Brooks


def test_structural_rank_counts_only_items_naming_a_company() -> None:
    text = "1. Think about fit first.\n2. Nike Pegasus.\n- Hoka Clifton, or Nike.\n"
    ranks = structural_ranks(text, {"nike": [text.index("Nike")], "hoka": [text.index("Hoka")]})
    assert ranks == {"nike": 1, "hoka": 2}  # item 1 names nobody; Nike is 2nd in item 3


# ---------------------------------------------------------------- merge


def _b(entity: str, canonical: str | None = None, rank: int | None = None) -> ExtractedMention:
    return ExtractedMention(
        entity=entity,
        canonical=canonical or entity,
        rank=rank,
        sentiment=0.7,
        sentiment_label="positive",
        is_recommended=True,
        snippet="",
    )


def test_merge_unions_records_disagreements_and_guards_hallucinations() -> None:
    text = (
        "Here are some of the best options:\n"
        "1. **Wrike**: strong resource planning.\n"
        "2. **Asana**: polished.\n"
        "On the other hand, Trello is simpler."
    )
    shoes_like = Resolver("Taskflow", [], ["Asana", "Trello", "On"])
    a = stage_a(text, shoes_like)
    b = [_b("Wrike", rank=1), _b("Asana", "Asana Inc.", rank=5), _b("Basecamp")]
    merged, dis = merge(text, shoes_like, a, b)
    by = {m.canonical: m for m in merged}
    assert list(by) == ["Wrike", "Asana", "Trello"]  # text order; Basecamp isn't in the text
    assert by["Wrike"].entity_type == "unknown" and by["Wrike"].rank == 1
    assert by["Asana"].source == "both" and by["Asana"].rank == 2  # structural beats B's 5
    assert by["Trello"].source == "a" and by["Trello"].sentiment_label == "neutral"
    types = Counter(d["type"] for d in dis)
    assert types == {"b_not_in_text": 1, "rank": 1, "a_only": 1}


def test_merge_drops_ambiguous_a_only_but_keeps_stage_a_when_llm_failed() -> None:
    text = "Runners also like On."
    a = stage_a(text, SHOES)
    assert [m.canonical for m in a] == ["On"]
    merged, dis = merge(text, SHOES, a, [])  # the model saw no company here
    assert merged == [] and dis == [{"type": "a_only_dropped", "canonical": "On"}]
    merged, dis = merge(text, SHOES, stage_a(text, SHOES), None)  # LLM failed: keep A
    assert [m.canonical for m in merged] == ["On"] and dis == []


# ---------------------------------------------------------------- extract run


def _counts(factory: sessionmaker[Session], run_id: int) -> tuple[int, int, int]:
    with factory() as s:
        answers = s.scalar(select(func.count()).where(Answer.run_id == run_id, Answer.ok))
        ext = s.scalar(
            select(func.count())
            .select_from(AnswerExtraction)
            .join(Answer, Answer.id == AnswerExtraction.answer_id)
            .where(Answer.run_id == run_id)
        )
        men = s.scalar(
            select(func.count())
            .select_from(MentionRow)
            .join(Answer, Answer.id == MentionRow.answer_id)
            .where(Answer.run_id == run_id)
        )
    return int(answers or 0), int(ext or 0), int(men or 0)


def test_extract_run_processes_every_answer_idempotently(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    chat = get_chat_provider(settings=settings, session_factory=scanned_factory)
    first = extract_run(scanned_factory, 1, chat, settings)
    answers, ext, men = _counts(scanned_factory, 1)
    assert answers == 200 and first.extracted == ext == 200 and first.llm_failed == 0
    assert men == first.mentions > 200
    with scanned_factory() as s:
        brand_rows = s.scalar(select(func.count()).where(MentionRow.is_brand))
        unknown = set(
            s.scalars(select(MentionRow.canonical).where(MentionRow.entity_type == "unknown"))
        )
        versions = set(s.scalars(select(AnswerExtraction.version)))
    assert brand_rows and brand_rows > 0
    assert {"Saucony", "Altra"} <= unknown  # emerging competitors from the world file
    assert versions == {EXTRACTOR_VERSION}

    again = extract_run(scanned_factory, 1, chat, settings)
    assert (again.skipped, again.extracted) == (200, 0)
    forced = extract_run(scanned_factory, 1, chat, settings, force=True)
    assert forced.extracted == 200
    assert _counts(scanned_factory, 1) == (answers, ext, men)  # replaced, not duplicated


def test_extract_run_retries_answers_whose_llm_stage_failed(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    class Broken:
        name = "broken"

        def complete(self, *args: object, **kwargs: object) -> object:
            raise ConnectionError("provider down")

    report = extract_run(scanned_factory, 2, Broken(), settings)  # type: ignore[arg-type]
    assert report.extracted == 200 and report.llm_failed == 200
    with scanned_factory() as s:
        assert s.scalar(select(func.count()).where(AnswerExtraction.llm_ok.is_(False))) == 200
    chat = get_chat_provider(settings=settings, session_factory=scanned_factory)
    retry = extract_run(scanned_factory, 2, chat, settings)
    assert (retry.skipped, retry.extracted, retry.llm_failed) == (0, 200, 0)


def test_extract_run_stops_at_the_spend_cap_without_marking_answers_done(
    scanned_factory: sessionmaker[Session], settings: Settings
) -> None:
    inner = get_chat_provider(settings=settings, session_factory=scanned_factory)
    capped = BudgetedChatProvider(inner, max_cost_usd=0.002)
    report = extract_run(scanned_factory, 1, capped, settings)
    assert 0 < report.extracted < 200
    assert report.budget_stopped == 200 - report.extracted
    assert _counts(scanned_factory, 1)[1] == report.extracted  # the rest stay unextracted
    rest = extract_run(scanned_factory, 1, inner, settings)
    assert rest.extracted == report.budget_stopped
