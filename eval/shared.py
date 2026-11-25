"""What the evaluation scripts share: the labelled questions and an index built in memory.

The index is built with grounded's own ingest, embedder and in-memory store, so the scores
measured here are the scores the app computes (the PostgreSQL store is tested to rank and score
identically).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from grounded.embed import Embedder, passage_input
from grounded.ingest import find_documents, read_document
from grounded.store import MemoryStore

EVAL = Path(__file__).resolve().parent
ROOT = EVAL.parent
DOCS = ROOT / "demo" / "docs"
RESULTS = EVAL / "results"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    source: str | None
    evidence: str | None
    kind: str | None


def load_questions(path: Path = EVAL / "questions.jsonl") -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        questions.append(
            Question(
                id=row["id"],
                question=row["question"],
                answerable=row["answerable"],
                source=row.get("source"),
                evidence=row.get("evidence"),
                kind=row.get("kind"),
            )
        )
    return questions


def build_index(
    embedder: Embedder, *, chunk_size: int = 256, with_headings: bool = True
) -> MemoryStore:
    store = MemoryStore()
    for path in find_documents(DOCS):
        document = read_document(path, DOCS, chunk_size)
        texts = [
            passage_input(document.title, p, with_headings=with_headings) for p in document.passages
        ]
        store.replace_document(document, "eval", embedder.encode(texts))
    return store


_SPACE = re.compile(r"\s+")


def squash(text: str) -> str:
    """Lower case, one space between words, curly apostrophes made straight."""
    return _SPACE.sub(" ", text.replace("’", "'")).strip().lower()


def contains_evidence(passage_text: str, evidence: str) -> bool:
    return squash(evidence) in squash(passage_text)


def write_json(name: str, data: object) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
