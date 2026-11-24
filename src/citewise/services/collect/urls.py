"""Citation URL normalization.

Engines cite the same page in many spellings: OpenAI appends `utm_source=openai`, links
carry `#fragments`, `www.` comes and goes, hosts arrive in mixed case. Module 3 counts
"top cited domains" and per-URL citation share, so every spelling must collapse to one key.
"""

from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# Query parameters that only identify the click, never the content.
TRACKING_PARAMS = frozenset(
    {
        "gclid", "gbraid", "wbraid", "fbclid", "msclkid", "yclid", "dclid", "igshid",
        "mc_cid", "mc_eid", "_hsenc", "_hsmi", "mkt_tok", "srsltid", "ref", "ref_src",
    }
)  # fmt: skip
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _is_tracking(key: str) -> bool:
    k = key.lower()
    return k.startswith("utm_") or k in TRACKING_PARAMS


def normalize_url(url: str) -> str:
    """Canonical form: https-or-http scheme, lowercase host without `www.`, no default port,
    no fragment, no tracking params, sorted query, no trailing slash (root becomes no path).

    A URL without a scheme is assumed to be https. Unparseable input is returned stripped.
    """
    raw = url.strip()
    if not raw:
        return raw
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    try:
        parts = urlsplit(raw)
        port = parts.port
    except ValueError:  # e.g. an invalid port such as ":abc"
        return url.strip()
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower().rstrip(".")
    host = host.removeprefix("www.")
    netloc = host if port is None or _DEFAULT_PORTS.get(scheme) == port else f"{host}:{port}"
    path = parts.path.rstrip("/")
    query = urlencode(
        sorted(
            (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if not _is_tracking(k)
        )
    )
    return urlunsplit((scheme, netloc, path, query, ""))


def domain_of(url: str) -> str:
    """Lowercase host without `www.` (and without port), e.g. "nike.com"."""
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw.lstrip("/")
    try:
        host = urlsplit(raw).hostname or ""
    except ValueError:
        return ""
    return host.lower().rstrip(".").removeprefix("www.")
