"""``grounded ingest ./docs``: index a folder of Markdown and PDF files."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

from grounded.config import Settings
from grounded.embed import EmbeddingError, get_embedder
from grounded.index import ingest_folder
from grounded.ingest import IngestError
from grounded.store import PostgresStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="grounded")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="index a folder of Markdown and PDF files")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--database-url", help="default: $GROUNDED_DATABASE_URL")
    ingest.add_argument("--model", help="embedding model; default: $GROUNDED_EMBEDDING_MODEL")
    ingest.add_argument("--chunk-size", type=int, help="characters per passage (default 256)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    settings = replace(
        settings,
        database_url=args.database_url or settings.database_url,
        embedding_model=args.model or settings.embedding_model,
        chunk_size=args.chunk_size or settings.chunk_size,
    )
    embedder = get_embedder(settings.embedding_model)
    try:
        with PostgresStore(settings.database_url, embedder.dimensions) as store:
            report = ingest_folder(args.folder, store, embedder, chunk_size=settings.chunk_size)
            documents, passages = store.counts()
    except (IngestError, EmbeddingError) as error:
        print(f"grounded: {error}", file=sys.stderr)
        return 2

    for path, outcome, detail in report.files:
        print(f"{outcome:9}  {path}" + (f"  ({detail})" if detail else ""))
    print(
        f"{report.count('added')} added, {report.count('updated')} updated, "
        f"{report.count('unchanged')} unchanged, {report.count('failed')} failed. "
        f"The index holds {documents} documents and {passages} passages."
    )
    return 1 if report.count("failed") else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
