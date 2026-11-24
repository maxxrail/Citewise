"""Fetch a public web page safely (live mode only): KB URL ingest and, optionally, reading
the pages an engine cited.

A URL can come from a user (`kb ingest <url>`, and from Module 8 a dashboard form) or from
an engine's citations, so the fetcher treats it as hostile:
- **No SSRF.** Every hop's host is resolved and every address must be public
  (`ipaddress.is_global`): loopback, RFC 1918, link-local (169.254.169.254 cloud
  metadata) and unique-local addresses are refused. Redirects are followed by hand, at
  most MAX_REDIRECTS, and each target is re-validated.
- **robots.txt** is checked for every host visited. 401/403 on robots.txt means
  disallow (the stdlib parser's convention); 404 means allow.
- **Bounded.** The body is streamed and cut at MAX_BYTES; only text/html and text/plain
  are accepted; one timeout per request.

A DNS answer can change between our check and the connection (rebinding). Pinning the
resolved IP would close that, at the cost of breaking TLS SNI; for a CLI and a dashboard
that only reads public pages, the per-hop check is the accepted trade-off.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.robotparser
from collections.abc import Callable
from urllib.parse import urljoin, urlsplit

import httpx

USER_AGENT = "CitewiseBot/0.1 (+https://github.com/maxxrail/Citewise)"
MAX_REDIRECTS = 3
MAX_BYTES = 2_000_000
ALLOWED_TYPES = ("text/html", "text/plain")
Resolver = Callable[[str], list[str]]


class FetchError(ValueError):
    pass


def _resolve(host: str) -> list[str]:
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None)]


def check_public(url: str, resolve: Resolver = _resolve) -> None:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FetchError(f"{url}: only http(s) URLs with a host can be fetched")
    try:
        addresses = resolve(parts.hostname)
    except OSError as exc:
        raise FetchError(f"{url}: cannot resolve {parts.hostname} ({exc})") from exc
    for addr in addresses:
        if not ipaddress.ip_address(addr.split("%")[0]).is_global:
            raise FetchError(f"{url}: {parts.hostname} resolves to a non-public address ({addr})")


def _robots_allows(client: httpx.Client, url: str) -> bool:
    parts = urlsplit(url)
    robots = urllib.robotparser.RobotFileParser()
    try:
        resp = client.get(f"{parts.scheme}://{parts.netloc}/robots.txt")
    except httpx.HTTPError:
        return True  # unreachable robots.txt: no rules published
    if resp.status_code in (401, 403):
        return False
    robots.parse(resp.text.splitlines() if resp.status_code == 200 else [])
    return robots.can_fetch(USER_AGENT, url)


def fetch_text(
    url: str,
    *,
    timeout_s: float = 20.0,
    resolve: Resolver = _resolve,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, str]:
    """(final URL, decoded body) of a public text/html or text/plain page."""
    with httpx.Client(
        timeout=timeout_s, follow_redirects=False, headers={"User-Agent": USER_AGENT},
        transport=transport,
    ) as client:  # fmt: skip
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            check_public(current, resolve)
            if not _robots_allows(client, current):
                raise FetchError(f"{current}: disallowed by robots.txt")
            with client.stream("GET", current) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise FetchError(f"{current}: redirect without a Location header")
                    current = urljoin(current, location)
                    continue
                resp.raise_for_status()
                ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if ctype not in ALLOWED_TYPES:
                    raise FetchError(f"{current}: unsupported content type {ctype or 'unknown'}")
                body = bytearray()
                for chunk in resp.iter_bytes():
                    body += chunk
                    if len(body) > MAX_BYTES:
                        raise FetchError(f"{current}: larger than {MAX_BYTES // 1_000_000} MB")
                return current, body.decode(resp.encoding or "utf-8", errors="replace")
        raise FetchError(f"{url}: more than {MAX_REDIRECTS} redirects")
