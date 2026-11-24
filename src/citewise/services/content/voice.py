"""Brand voice: `data/brands/<brand>/voice.md` (front matter + Markdown).

Front matter sets the machine-checked limits (`reading_level` = max Flesch-Kincaid grade,
`max_words`); the `## Banned words` list is checked word by word; the whole text goes to
the writer and the judge as guidance.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from citewise.core.config import get_settings

_FRONT = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def voice_path(brand_name: str, brands_dir: Path | None = None) -> Path:
    base = brands_dir or get_settings().brands_dir
    return base / re.sub(r"[^a-z0-9]+", "-", brand_name.casefold()).strip("-") / "voice.md"


def parse_voice(text: str) -> dict[str, Any]:
    text = text.replace("\r\n", "\n")
    meta: dict[str, Any] = {}
    m = _FRONT.match(text)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        text = text[m.end() :]
    banned: list[str] = []
    section = re.search(r"^## Banned words\s*\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if section:
        banned = [
            line.strip()[2:].strip().casefold()
            for line in section.group(1).splitlines()
            if line.strip().startswith(("- ", "* "))
        ]
    return {
        "text": text.strip(),
        "reading_level": meta.get("reading_level"),
        "max_words": meta.get("max_words"),
        "banned": [b for b in banned if b],
    }


def load_voice(brand_name: str, brands_dir: Path | None = None) -> dict[str, Any]:
    """The brand's voice, or a neutral default (no limits) if it has no voice.md."""
    path = voice_path(brand_name, brands_dir)
    if not path.exists():
        return {
            "text": "Plain, specific and neutral. No hype.",
            "reading_level": None,
            "max_words": None,
            "banned": [],
            "missing": True,
        }
    return {**parse_voice(path.read_text(encoding="utf-8")), "missing": False}
