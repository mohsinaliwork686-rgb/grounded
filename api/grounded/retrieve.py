"""Finding passages for a question, and deciding whether they are good enough to answer from.

The decision is one comparison: the best passage's cosine similarity against a threshold. Below
it, grounded refuses without calling the model. The threshold isn't guessed; it's measured on
labelled questions by ``eval/refusal.py`` and its result is the default here.
"""

from __future__ import annotations

from dataclasses import dataclass

from grounded.embed import Embedder
from grounded.store import Hit, Store

# Measured on 25 Nov 2026 by eval/refusal.py on the demo documents (see eval/results/).
DEFAULT_THRESHOLD = 0.55
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class Retrieval:
    question: str
    hits: tuple[Hit, ...]
    threshold: float

    @property
    def top_score(self) -> float:
        return self.hits[0].score if self.hits else 0.0

    @property
    def confident(self) -> bool:
        """True when the best passage clears the threshold. No passages is never confident."""
        return bool(self.hits) and self.top_score >= self.threshold


def retrieve(
    question: str,
    store: Store,
    embedder: Embedder,
    *,
    k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> Retrieval:
    if not question.strip():
        raise ValueError("the question is empty")
    vector = embedder.encode([question.strip()])[0]
    return Retrieval(question=question, hits=tuple(store.search(vector, k)), threshold=threshold)
