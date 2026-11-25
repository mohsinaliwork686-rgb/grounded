"""The pgvector store against a real PostgreSQL in a container."""

from __future__ import annotations

from pathlib import Path

import pytest

from grounded.cli import main
from grounded.embed import HashingEmbedder
from grounded.index import ingest_folder
from grounded.ingest import read_document
from grounded.store import MemoryStore, PostgresStore


@pytest.fixture
def store(postgres_url: str) -> PostgresStore:
    pg = PostgresStore(postgres_url, 384)
    pg._connection.execute("TRUNCATE documents CASCADE")
    yield pg
    pg.close()


QUESTIONS = [
    "How long are carbon frames covered for?",
    "How long does a refund take to reach my card?",
    "When is the shop open on Sunday?",
    "What is the capital of Australia?",
]


def test_postgres_ranks_and_scores_exactly_like_the_memory_store(
    store: PostgresStore, demo_docs: Path
) -> None:
    embedder = HashingEmbedder()
    memory = MemoryStore()
    ingest_folder(demo_docs, store, embedder)
    ingest_folder(demo_docs, memory, embedder)
    assert store.counts() == memory.counts()

    for question in QUESTIONS:
        vector = embedder.encode([question])[0]
        pg_hits, memory_hits = store.search(vector, 5), memory.search(vector, 5)
        assert [h.passage.text for h in pg_hits] == [h.passage.text for h in memory_hits]
        for pg_hit, memory_hit in zip(pg_hits, memory_hits, strict=True):
            assert pg_hit.score == pytest.approx(memory_hit.score, abs=1e-5)


def test_stored_passages_keep_their_source_positions(store: PostgresStore, demo_docs: Path) -> None:
    ingest_folder(demo_docs, store, HashingEmbedder())
    vector = HashingEmbedder().encode(["carbon frames covered five years"])[0]
    hit = store.search(vector, 1)[0]
    assert hit.passage.document_path == "warranty.pdf" and hit.passage.page == 1
    assert hit.passage.heading_path == ("Frames",)
    document = read_document(demo_docs / "warranty.pdf", demo_docs)
    source = document.sources[0]
    assert source[hit.passage.start : hit.passage.end] == hit.passage.text


def test_replacing_a_document_replaces_its_passages(store: PostgresStore, tmp_path: Path) -> None:
    embedder = HashingEmbedder()
    (tmp_path / "a.md").write_text("# A\n\nReturns within 30 days.")
    ingest_folder(tmp_path, store, embedder)
    (tmp_path / "a.md").write_text("# A\n\nReturns within 14 days. A second sentence here.")
    report = ingest_folder(tmp_path, store, embedder)
    assert report.count("updated") == 1
    assert store.counts() == (1, 1)
    texts = [h.passage.text for h in store.search(embedder.encode(["returns"])[0], 5)]
    assert texts == ["Returns within 14 days. A second sentence here."]
    assert ingest_folder(tmp_path, store, embedder).count("unchanged") == 1


def test_the_cli_ingests_a_folder(
    postgres_url: str, store: PostgresStore, demo_docs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["ingest", str(demo_docs), "--database-url", postgres_url, "--model", "hashing"]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert "6 added" in first and "6 documents" in first
    assert main(args) == 0
    assert "6 unchanged" in capsys.readouterr().out


def test_the_cli_exits_non_zero_when_a_file_fails(
    postgres_url: str, store: PostgresStore, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    assert (
        main(["ingest", str(tmp_path), "--database-url", postgres_url, "--model", "hashing"]) == 1
    )
    assert "failed" in capsys.readouterr().out


def test_the_cli_reports_a_missing_folder(postgres_url: str, store: PostgresStore) -> None:
    assert (
        main(["ingest", "/no/such/folder", "--database-url", postgres_url, "--model", "hashing"])
        == 2
    )
