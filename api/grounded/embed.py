"""Turning text into unit vectors.

Every embedder returns L2-normalised rows, so cosine similarity is a dot product, and the score
PostgreSQL computes (``1 - (a <=> b)``) is the same number the evaluation scripts compute.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from grounded.ingest import Passage

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSIONS = 384

Vectors = npt.NDArray[np.float32]

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingError(Exception):
    """An embedding model was asked for but isn't installed."""


class Embedder(Protocol):
    @property
    def name(self) -> str:
        """Stable identifier, stored with the index so a model change forces a re-ingest."""
        ...

    @property
    def dimensions(self) -> int: ...

    def encode(self, texts: list[str]) -> Vectors:
        """Shape (len(texts), dimensions), L2-normalised, float32."""
        ...


def normalise(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


class HashingEmbedder:
    """Feature hashing over words. No model, no download, the same output everywhere.

    It is lexical, not semantic. It exists so tests and CI never download a model, and it has
    the real model's 384 dimensions so the same database schema serves both.
    """

    def __init__(self, dimensions: int = DIMENSIONS) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing-{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimensions

    def encode(self, texts: list[str]) -> Vectors:
        matrix = np.zeros((len(texts), self._dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                matrix[row, self._bucket(token)] += 1.0
        return normalise(matrix)


class SentenceTransformerEmbedder:
    """A real semantic model. Needs the ``ml`` extra; loaded on first use."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise EmbeddingError(
                    f'{self._model_name!r} needs sentence-transformers: pip install -e "api[ml]"'
                ) from error
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    @property
    def dimensions(self) -> int:
        return int(self._load().get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> Vectors:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        vectors = self._load().encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return normalise(np.asarray(vectors, dtype=np.float32))


def get_embedder(name: str) -> Embedder:
    """``hashing`` or ``hashing-<dims>`` for the stub; otherwise a sentence-transformers model."""
    if name == "hashing":
        return HashingEmbedder()
    if name.startswith("hashing-"):
        return HashingEmbedder(int(name.removeprefix("hashing-")))
    return SentenceTransformerEmbedder(name)


def passage_input(title: str, passage: Passage, *, with_headings: bool = True) -> str:
    """What gets embedded for a passage: its headings, then its text.

    A passage like "Sunday, 10am to 4pm." doesn't say it's about opening hours; its heading
    does. The evaluation measures retrieval both ways (``eval/retrieval.py``).
    """
    if not with_headings:
        return passage.text
    path = " > ".join((title, *passage.heading_path)) if title else " > ".join(passage.heading_path)
    return f"{path}\n{passage.text}" if path else passage.text
