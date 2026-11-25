"""Where passages and their vectors live: PostgreSQL with pgvector, or memory for tests.

Both stores rank by cosine similarity of unit vectors and return the same ``Hit`` shape, so the
retrieval and refusal code above them can't tell which one it's talking to. The PostgreSQL one
is what the app runs; the in-memory one is what the unit tests and the evaluation scripts use,
and an integration test checks the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from grounded.embed import Vectors
from grounded.ingest import Document, Passage


@dataclass(frozen=True)
class StoredPassage:
    id: int
    document_path: str
    document_title: str
    text: str
    start: int
    end: int
    page: int | None
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class Hit:
    passage: StoredPassage
    score: float


class Store(Protocol):
    def indexed_with(self, path: str) -> tuple[str, str] | None:
        """The content hash and index signature a document was stored with, if it is stored."""
        ...

    def replace_document(self, document: Document, signature: str, vectors: Vectors) -> None:
        """Store a document and its passages, replacing any earlier version of the same path."""
        ...

    def search(self, vector: Vectors, k: int) -> list[Hit]:
        """The k passages most similar to ``vector``, best first."""
        ...

    def counts(self) -> tuple[int, int]:
        """(documents, passages)."""
        ...


def _check(document: Document, vectors: Vectors) -> None:
    if len(vectors) != len(document.passages):
        raise ValueError(
            f"{document.path}: {len(document.passages)} passages but {len(vectors)} vectors"
        )


# ----------------------------------------------------------------------------- memory


class MemoryStore:
    def __init__(self) -> None:
        self._documents: dict[str, tuple[Document, str]] = {}
        self._passages: list[StoredPassage] = []
        self._vectors: list[Vectors] = []
        self._next_id = 1

    def indexed_with(self, path: str) -> tuple[str, str] | None:
        entry = self._documents.get(path)
        return (entry[0].content_hash, entry[1]) if entry else None

    def replace_document(self, document: Document, signature: str, vectors: Vectors) -> None:
        _check(document, vectors)
        keep = [i for i, p in enumerate(self._passages) if p.document_path != document.path]
        self._passages = [self._passages[i] for i in keep]
        self._vectors = [self._vectors[i] for i in keep]
        self._documents[document.path] = (document, signature)
        for passage, vector in zip(document.passages, vectors, strict=True):
            self._passages.append(self._stored(document, passage))
            self._vectors.append(vector)

    def _stored(self, document: Document, passage: Passage) -> StoredPassage:
        stored = StoredPassage(
            id=self._next_id,
            document_path=document.path,
            document_title=document.title,
            text=passage.text,
            start=passage.start,
            end=passage.end,
            page=passage.page,
            heading_path=passage.heading_path,
        )
        self._next_id += 1
        return stored

    def search(self, vector: Vectors, k: int) -> list[Hit]:
        if not self._passages or k <= 0:
            return []
        scores = np.vstack(self._vectors) @ np.asarray(vector, dtype=np.float32)
        # Stable sort on the negated score: ties keep insertion order, like the SQL's id tiebreak.
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(self._passages[i], float(scores[i])) for i in order]

    def counts(self) -> tuple[int, int]:
        return len(self._documents), len(self._passages)


# ----------------------------------------------------------------------------- PostgreSQL

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           bigserial PRIMARY KEY,
    path         text NOT NULL UNIQUE,
    kind         text NOT NULL CHECK (kind IN ('markdown', 'pdf')),
    title        text NOT NULL,
    content_hash text NOT NULL,
    signature    text NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS passages (
    id           bigserial PRIMARY KEY,
    document_id  bigint NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    position     integer NOT NULL,
    text         text NOT NULL,
    start_offset integer NOT NULL,
    end_offset   integer NOT NULL,
    page         integer,
    heading_path text[] NOT NULL,
    embedding    vector({dimensions}) NOT NULL,
    UNIQUE (document_id, position)
);
"""

_SEARCH = """
SELECT p.id, d.path, d.title, p.text, p.start_offset, p.end_offset, p.page, p.heading_path,
       1 - (p.embedding <=> %(vector)s) AS score
FROM passages p
JOIN documents d ON d.id = p.document_id
ORDER BY p.embedding <=> %(vector)s, p.id
LIMIT %(k)s
"""


class PostgresStore:
    """pgvector, searched exactly.

    No approximate index: a few thousand passages are scanned in milliseconds, and an exact scan
    means the scores the refusal threshold was measured on are the scores the app sees. An HNSW
    index is the change to make past roughly a hundred thousand passages.
    """

    def __init__(self, url: str, dimensions: int) -> None:
        try:
            import psycopg
            from pgvector.psycopg import register_vector
        except ImportError as error:  # pragma: no cover - core dependencies
            raise RuntimeError("the PostgreSQL store needs psycopg and pgvector") from error
        self._connection: Any = psycopg.connect(url, autocommit=True)
        self._connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self._connection)
        self._connection.execute(SCHEMA.format(dimensions=int(dimensions)))

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PostgresStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def indexed_with(self, path: str) -> tuple[str, str] | None:
        row = self._connection.execute(
            "SELECT content_hash, signature FROM documents WHERE path = %s", (path,)
        ).fetchone()
        return (row[0], row[1]) if row else None

    def replace_document(self, document: Document, signature: str, vectors: Vectors) -> None:
        _check(document, vectors)
        with self._connection.transaction():
            self._connection.execute("DELETE FROM documents WHERE path = %s", (document.path,))
            row = self._connection.execute(
                "INSERT INTO documents (path, kind, title, content_hash, signature) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (document.path, document.kind, document.title, document.content_hash, signature),
            ).fetchone()
            with self._connection.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO passages (document_id, position, text, start_offset, "
                    "end_offset, page, heading_path, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            row[0],
                            position,
                            passage.text,
                            passage.start,
                            passage.end,
                            passage.page,
                            list(passage.heading_path),
                            np.asarray(vector, dtype=np.float32),
                        )
                        for position, (passage, vector) in enumerate(
                            zip(document.passages, vectors, strict=True)
                        )
                    ],
                )

    def search(self, vector: Vectors, k: int) -> list[Hit]:
        if k <= 0:
            return []
        rows = self._connection.execute(
            _SEARCH, {"vector": np.asarray(vector, dtype=np.float32), "k": k}
        ).fetchall()
        return [
            Hit(
                StoredPassage(
                    id=row[0],
                    document_path=row[1],
                    document_title=row[2],
                    text=row[3],
                    start=row[4],
                    end=row[5],
                    page=row[6],
                    heading_path=tuple(row[7]),
                ),
                float(row[8]),
            )
            for row in rows
        ]

    def counts(self) -> tuple[int, int]:
        row = self._connection.execute(
            "SELECT (SELECT count(*) FROM documents), (SELECT count(*) FROM passages)"
        ).fetchone()
        return int(row[0]), int(row[1])
