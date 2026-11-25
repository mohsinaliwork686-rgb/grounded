"""Turn grounded's demo set into ragsweep's input: a text corpus and document-level labels.

ragsweep (week 2) reads Markdown and plain text, not PDF, so the warranty booklet goes in as the
text grounded's own ingest extracts from it. Labels are per document, which is how ragsweep
scores; only the 28 answerable questions have a document to find.

    python eval/ragsweep/export.py && cd eval/ragsweep && ragsweep run
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from shared import DOCS, load_questions  # noqa: E402

from grounded.ingest import find_documents, read_pdf_pages  # noqa: E402


def text_name(path: Path) -> str:
    return path.with_suffix(".txt").name if path.suffix.lower() == ".pdf" else path.name


def main() -> None:
    corpus = HERE / "corpus"
    shutil.rmtree(corpus, ignore_errors=True)
    corpus.mkdir()
    names = {}
    for path in find_documents(DOCS):
        names[path.name] = text_name(path)
        if path.suffix.lower() == ".pdf":
            pages, _ = read_pdf_pages(path)
            (corpus / names[path.name]).write_text("\n\n".join(pages), encoding="utf-8")
        else:
            shutil.copy(path, corpus / names[path.name])
    with (HERE / "labels.jsonl").open("w", encoding="utf-8") as out:
        for q in load_questions():
            if q.answerable and q.source:
                row = {"id": q.id, "question": q.question, "relevant": [names[q.source]]}
                out.write(json.dumps(row) + "\n")
    print(f"wrote {len(names)} documents to corpus/ and labels.jsonl")


if __name__ == "__main__":
    main()
