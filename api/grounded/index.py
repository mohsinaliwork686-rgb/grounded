"""Ingest a folder: read, chunk, embed and store, skipping files that haven't changed."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from grounded.embed import Embedder, passage_input
from grounded.ingest import DEFAULT_CHUNK_SIZE, IngestError, find_documents, read_document
from grounded.store import Store

Outcome = Literal["added", "updated", "unchanged", "failed"]


@dataclass
class IngestReport:
    files: list[tuple[str, Outcome, str]] = field(default_factory=list)

    def count(self, outcome: Outcome) -> int:
        return sum(1 for _, result, _ in self.files if result == outcome)


def signature(embedder: Embedder, chunk_size: int) -> str:
    """What the stored vectors depend on besides the file. A change re-indexes every file.

    ``pdf-joined``: PDF words split across lines with a hyphen are joined (26 November). Without
    it in the signature, an index built before that change would keep the split words.
    """
    return f"{embedder.name}|sentences-{chunk_size}|headings|pdf-joined"


def ingest_folder(
    root: Path, store: Store, embedder: Embedder, *, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> IngestReport:
    """Index every Markdown and PDF file under ``root``.

    A file that fails to read is reported and skipped; the rest still go in. An unchanged file
    (same content hash, same embedding model and chunking) isn't read twice.
    """
    report = IngestReport()
    wanted = signature(embedder, chunk_size)
    for path in find_documents(root):
        relative = path.relative_to(root).as_posix()
        try:
            document = read_document(path, root, chunk_size)
        except IngestError as error:
            report.files.append((relative, "failed", str(error)))
            continue
        previous = store.indexed_with(relative)
        if previous == (document.content_hash, wanted):
            report.files.append((relative, "unchanged", ""))
            continue
        texts = [passage_input(document.title, p) for p in document.passages]
        vectors = embedder.encode(texts)
        store.replace_document(document, wanted, vectors)
        outcome: Outcome = "added" if previous is None else "updated"
        report.files.append((relative, outcome, f"{len(document.passages)} passages"))
    return report
