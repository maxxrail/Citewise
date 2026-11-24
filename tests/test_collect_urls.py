"""URL normalization (Module 2 acceptance: UTM, trailing slash, www, fragments)."""

from __future__ import annotations

import pytest

from citewise.services.collect.urls import domain_of, normalize_url


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # UTM and other click-tracking parameters are dropped; real parameters stay.
        ("https://nike.com/shoes?utm_source=openai", "https://nike.com/shoes"),
        (
            "https://nike.com/shoes?utm_source=x&utm_medium=y&utm_campaign=z",
            "https://nike.com/shoes",
        ),
        (
            "https://nike.com/shoes?color=red&UTM_Source=openai&gclid=abc",
            "https://nike.com/shoes?color=red",
        ),
        ("https://nike.com/s?q=trail&fbclid=1&srsltid=2", "https://nike.com/s?q=trail"),
        # Remaining parameters are sorted so order doesn't create two keys.
        ("https://nike.com/s?b=2&a=1", "https://nike.com/s?a=1&b=2"),
        # Trailing slash.
        ("https://nike.com/running/", "https://nike.com/running"),
        ("https://nike.com/", "https://nike.com"),
        ("https://nike.com", "https://nike.com"),
        # www and host case.
        ("https://www.Nike.COM/Running", "https://nike.com/Running"),  # path case is kept
        ("https://WWW.nike.com.", "https://nike.com"),
        # Fragments.
        ("https://nike.com/shoes#reviews", "https://nike.com/shoes"),
        ("https://nike.com/shoes/?utm_source=openai#top", "https://nike.com/shoes"),
        # Default ports dropped, others kept; scheme lowercased; missing scheme -> https.
        ("HTTPS://nike.com:443/a", "https://nike.com/a"),
        ("http://nike.com:80/a", "http://nike.com/a"),
        ("https://nike.com:8443/a", "https://nike.com:8443/a"),
        ("nike.com/running", "https://nike.com/running"),
        # Blank values of real params survive.
        ("https://nike.com/s?q=", "https://nike.com/s?q="),
    ],
)
def test_normalize_url(raw: str, expected: str) -> None:
    assert normalize_url(raw) == expected


def test_normalization_is_idempotent() -> None:
    raw = "https://www.Reddit.com/r/running/comments/abc/?utm_source=openai&sort=top#c1"
    once = normalize_url(raw)
    assert normalize_url(once) == once == "https://reddit.com/r/running/comments/abc?sort=top"


def test_unparseable_input_does_not_raise() -> None:
    assert normalize_url("  ") == ""
    assert normalize_url("https://nike.com:abc/x") == "https://nike.com:abc/x"


@pytest.mark.parametrize(
    ("url", "domain"),
    [
        ("https://www.runnersworld.com/gear/a1/", "runnersworld.com"),
        ("https://Shop.Nike.com:443/x", "shop.nike.com"),
        ("reddit.com/r/running", "reddit.com"),
        ("https://nike.com:abc/x", "nike.com"),  # a bad port still has a usable host
    ],
)
def test_domain_of(url: str, domain: str) -> None:
    assert domain_of(url) == domain
