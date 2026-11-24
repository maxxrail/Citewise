"""Local embeddings (no API key, no per-token cost) via fastembed (ONNX, CPU).

Anthropic has no embeddings API, so "Claude for everything" still needs vectors from
somewhere. A small local model keeps the whole pipeline on one paid account:
`BAAI/bge-small-en-v1.5` (384 dims, English, ~130 MB) is fastembed's default and is strong
for short-text similarity, which is what prompt dedup, clustering and (Module 4)
retrieval need. The model is downloaded once on first use to the fastembed cache.

Vectors are L2-normalised, so cosine similarity is a dot product, like OpenAI's. Stored
embeddings record their space (`local:<model>`), so switching embedders re-embeds stale
rows automatically (Module 1 review).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from citewise.providers.base import EmbedResult, Usage
from citewise.providers.mock import estimate_tokens

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-en-v1.5"
# Per-user, outside the repo and outside the temp dir. FASTEMBED_CACHE_PATH overrides it
# (e.g. a mounted volume in Docker).
CACHE_DIR = Path(
    os.environ.get("FASTEMBED_CACHE_PATH", Path.home() / ".cache" / "citewise" / "fastembed")
)


class LocalEmbedder:
    name = "local"

    def __init__(self, model: str = DEFAULT_LOCAL_MODEL, *, batch_size: int = 64) -> None:
        self.model = model
        self.batch_size = batch_size
        self._impl: Any = None

    def _load(self) -> Any:
        if self._impl is None:  # lazy: the first call downloads/loads the ONNX model
            # Windows without Developer Mode can't symlink; the HF cache still works, so
            # don't print a warning on every run.
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            from fastembed import TextEmbedding

            # fastembed defaults to the OS temp dir, which Windows cleans: the model would
            # be re-downloaded (~130 MB) after every cleanup.
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            self._impl = TextEmbedding(model_name=self.model, cache_dir=str(CACHE_DIR))
        return self._impl

    def embed_with_usage(self, texts: list[str]) -> EmbedResult:
        vectors = [
            [float(x) for x in v] for v in self._load().embed(texts, batch_size=self.batch_size)
        ]
        return EmbedResult(
            vectors=vectors,
            model=self.model,
            provider=self.name,
            # Not billed; recorded so llm_calls still shows the volume of embedding work.
            usage=Usage(tokens_in=sum(estimate_tokens(t) for t in texts)),
        )
