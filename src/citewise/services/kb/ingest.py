"""Ingest brand source-of-truth documents: parse -> chunk -> embed -> kb_docs / kb_chunks.

Inputs: Markdown or text files (optional YAML front matter: title, category, as_of,
source_url), PDFs (text layer via pypdf), a directory of those, or a URL (live mode only:
fetched with robots.txt respected, HTML reduced to text).

Chunking: paragraphs are packed into chunks of about `chunk_tokens` tokens. A paragraph
that is too long is split at sentence boundaries, and a single sentence longer than a chunk
is hard-wrapped by words. Consecutive chunks share about `overlap_tokens` of trailing
sentences, so a fact that straddles a boundary is retrievable from either side. Every chunk
starts with "<doc title> > <section heading>" so it is self-describing for BM25, the
embedder, and the verifier's evidence quote.

Idempotent: a doc is keyed by (brand, slug); an unchanged doc (same content hash, chunks
embedded in the current vector space) is skipped, and a changed one replaces its chunks.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from citewise.models.brand import Brand
from citewise.models.kb import KB_CATEGORIES, KBChunk, KBDoc
from citewise.providers.base import EmbeddingProvider
from citewise.providers.mock import estimate_tokens

SUPPORTED = (".md", ".markdown", ".txt", ".pdf")
USER_AGENT = "CitewiseBot/0.1 (+https://github.com/maxxrail/Citewise)"
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$")
_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


class IngestError(ValueError):
    pass


@dataclass
class ParsedDoc:
    slug: str
    title: str
    category: str
    source: str
    as_of: str | None
    text: str


@dataclass
class Chunk:
    ord: int
    text: str
    tokens: int


@dataclass
class IngestReport:
    docs_added: int = 0
    docs_updated: int = 0
    docs_unchanged: int = 0
    chunks: int = 0
    slugs: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- parsing


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:200] or "doc"


def _parse_text(raw: str, slug: str, source: str, category: str | None) -> ParsedDoc:
    meta: dict = {}
    if m := _FRONT_MATTER.match(raw):
        meta = yaml.safe_load(m.group(1)) or {}
        raw = raw[m.end() :]
    title = str(meta.get("title") or "")
    if not title:
        first = next((ln for ln in raw.splitlines() if ln.strip()), slug)
        title = _HEADING.sub(r"\1", first).strip() or slug
    cat = str(meta.get("category") or category or "").lower()
    if cat not in KB_CATEGORIES:
        raise IngestError(
            f"{source}: category must be one of {', '.join(KB_CATEGORIES)} "
            "(front matter `category:` or --category)"
        )
    return ParsedDoc(
        slug=slug,
        title=title[:300],
        category=cat,
        source=str(meta.get("source_url") or source),
        as_of=str(meta["as_of"]) if meta.get("as_of") else None,
        text=raw.replace("\r\n", "\n").strip(),
    )


def parse_file(path: Path, category: str | None = None) -> ParsedDoc:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise IngestError(f"{path}: unsupported type (expected {', '.join(SUPPORTED)})")
    if suffix == ".pdf":
        from pypdf import PdfReader

        raw = "\n\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    else:
        raw = path.read_text(encoding="utf-8")
    return _parse_text(raw, slugify(path.stem), str(path.as_posix()), category)


def html_to_text(page: str) -> str:
    page = re.sub(r"(?is)<(script|style|noscript|svg)\b.*?</\1>", " ", page)
    page = re.sub(r"(?i)<br\s*/?>|</(p|div|li|h[1-6]|tr)>", "\n\n", page)
    page = re.sub(r"(?s)<[^>]+>", " ", page)
    page = html.unescape(page)
    lines = [" ".join(ln.split()) for ln in page.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fetch_url(url: str, category: str | None, *, timeout_s: float = 20.0) -> ParsedDoc:
    """Live mode only (the caller checks). Honours robots.txt for our user agent."""
    import httpx

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        raise IngestError(f"{url}: only http(s) URLs can be ingested")
    robots = urllib.robotparser.RobotFileParser()
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        resp = httpx.get(robots_url, timeout=timeout_s, headers={"User-Agent": USER_AGENT})
        robots.parse(resp.text.splitlines() if resp.status_code == 200 else [])
    except httpx.HTTPError:
        robots.parse([])
    if not robots.can_fetch(USER_AGENT, url):
        raise IngestError(f"{url}: disallowed by robots.txt")
    resp = httpx.get(
        url, timeout=timeout_s, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    )
    resp.raise_for_status()
    text = html_to_text(resp.text) if "html" in resp.headers.get("content-type", "") else resp.text
    slug = (
        slugify(parts.netloc + "-" + parts.path) if parts.path.strip("/") else slugify(parts.netloc)
    )
    return _parse_text(text, slug, url, category)


# ---------------------------------------------------------------- chunking


def _sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(paragraph) if s.strip()]


def _hard_wrap(sentence: str, max_tokens: int) -> list[str]:
    words = sentence.split()
    out: list[str] = []
    cur: list[str] = []
    for w in words:
        if cur and estimate_tokens(" ".join([*cur, w])) > max_tokens:
            out.append(" ".join(cur))
            cur = []
        cur.append(w)
    return [*out, " ".join(cur)] if cur else out


def chunk_text(
    text: str, title: str, *, chunk_tokens: int = 400, overlap_tokens: int = 60
) -> list[Chunk]:
    """Pack sentences (grouped by paragraph and section) into ~chunk_tokens chunks."""
    units: list[tuple[str, str]] = []  # (section heading, sentence)
    heading = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        if (m := _HEADING.match(lines[0])) is not None:
            heading = m.group(1).strip()
            block = "\n".join(lines[1:]).strip()
            if not block:
                continue
        for sentence in _sentences(" ".join(ln.strip() for ln in block.splitlines())):
            for piece in (
                _hard_wrap(sentence, chunk_tokens)
                if estimate_tokens(sentence) > chunk_tokens
                else [sentence]
            ):
                units.append((heading, piece))

    chunks: list[Chunk] = []
    cur: list[tuple[str, str]] = []

    def prefix(h: str) -> str:
        return f"{title} > {h}\n" if h else f"{title}\n"

    def emit() -> None:
        body = " ".join(s for _, s in cur)
        txt = prefix(cur[0][0]) + body
        chunks.append(Chunk(ord=len(chunks), text=txt, tokens=estimate_tokens(txt)))

    for unit in units:
        if cur and (
            unit[0] != cur[-1][0]  # a new section starts a new chunk
            or estimate_tokens(" ".join(s for _, s in [*cur, unit])) > chunk_tokens
        ):
            emit()
            # Overlap: carry trailing sentences of the same section into the next chunk.
            carry: list[tuple[str, str]] = []
            if unit[0] == cur[-1][0]:
                for prev in reversed(cur):
                    if estimate_tokens(" ".join(s for _, s in [prev, *carry])) > overlap_tokens:
                        break
                    carry.insert(0, prev)
                if len(carry) == len(cur):
                    carry = carry[1:]  # always advance
            cur = carry
        cur.append(unit)
    if cur:
        emit()
    return chunks


# ---------------------------------------------------------------- ingest


def _content_hash(doc: ParsedDoc, chunk_tokens: int, overlap_tokens: int) -> str:
    parts = [doc.title, doc.category, str(doc.as_of), str(chunk_tokens), str(overlap_tokens)]
    body = "\x1f".join([*parts, doc.text])
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def collect_docs(target: str, category: str | None, *, live: bool) -> list[ParsedDoc]:
    if target.startswith(("http://", "https://")):
        if not live:
            raise IngestError("URL ingest needs CITEWISE_MODE=live (mock mode never fetches)")
        return [fetch_url(target, category)]
    path = Path(target)
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED)
        if not files:
            raise IngestError(f"{path}: no {', '.join(SUPPORTED)} files")
        return [parse_file(p, category) for p in files]
    if not path.exists():
        raise IngestError(f"{path}: not found")
    return [parse_file(path, category)]


def ingest(
    session: Session,
    brand: Brand,
    docs: list[ParsedDoc],
    embedder: EmbeddingProvider,
    *,
    chunk_tokens: int = 400,
    overlap_tokens: int = 60,
) -> IngestReport:
    report = IngestReport()
    for doc in docs:
        h = _content_hash(doc, chunk_tokens, overlap_tokens)
        row = session.scalar(
            select(KBDoc).where(KBDoc.brand_id == brand.id, KBDoc.slug == doc.slug)
        )
        if row is not None and row.content_hash == h:
            stale = session.scalar(
                select(KBChunk.id)
                .where(KBChunk.doc_id == row.id, KBChunk.embedding_model != embedder.ident)
                .limit(1)
            )
            if stale is None:
                report.docs_unchanged += 1
                report.slugs.append(doc.slug)
                continue
        chunks = chunk_text(
            doc.text, doc.title, chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens
        )
        # Embed before writing: a failed embedding call leaves the old version intact.
        vectors = embedder.embed([c.text for c in chunks], purpose="kb.embed") if chunks else []
        if row is None:
            row = KBDoc(brand_id=brand.id, slug=doc.slug)
            session.add(row)
            report.docs_added += 1
        else:
            session.execute(delete(KBChunk).where(KBChunk.doc_id == row.id))
            report.docs_updated += 1
        row.title, row.category, row.source = doc.title, doc.category, doc.source
        row.as_of, row.content_hash, row.text = doc.as_of, h, doc.text
        session.flush()
        for c, v in zip(chunks, vectors, strict=True):
            session.add(
                KBChunk(
                    doc_id=row.id, brand_id=brand.id, chunk_key=f"{doc.slug}#{c.ord}", ord=c.ord,
                    text=c.text, tokens=c.tokens, embedding=v, embedding_model=embedder.ident,
                )
            )  # fmt: skip
        report.chunks += len(chunks)
        report.slugs.append(doc.slug)
    session.flush()
    return report
