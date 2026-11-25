from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from grounded.embed import HashingEmbedder, get_embedder, passage_input
from grounded.index import ingest_folder, signature
from grounded.ingest import Passage
from grounded.retrieve import Retrieval, retrieve
from grounded.store import Hit, MemoryStore, StoredPassage


@pytest.fixture
def indexed(demo_docs: Path) -> MemoryStore:
    store = MemoryStore()
    ingest_folder(demo_docs, store, HashingEmbedder())
    return store


def test_the_hashing_embedder_returns_unit_rows_of_the_right_shape() -> None:
    vectors = HashingEmbedder().encode(["frame warranty", "", "returns within 30 days"])
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    norms = np.linalg.norm(vectors, axis=1)
    assert norms[0] == pytest.approx(1.0) and norms[1] == 0.0


def test_get_embedder_names() -> None:
    assert get_embedder("hashing").name == "hashing-384"
    assert get_embedder("hashing-64").dimensions == 64


def test_passage_input_puts_the_headings_first() -> None:
    passage = Passage("Sunday, 10am to 4pm.", 0, 20, None, ("Visiting the shop", "Opening hours"))
    assert passage_input("Handbook", passage) == (
        "Handbook > Visiting the shop > Opening hours\nSunday, 10am to 4pm."
    )
    assert passage_input("Handbook", passage, with_headings=False) == "Sunday, 10am to 4pm."


def test_a_known_question_finds_its_passage(indexed: MemoryStore) -> None:
    result = retrieve(
        "How long are carbon frames covered for?", indexed, HashingEmbedder(), threshold=0.1
    )
    assert result.hits[0].passage.document_path == "warranty.pdf"
    assert "Carbon frames are covered for 5 years" in result.hits[0].passage.text
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True) and len(scores) == 5


def test_refusal_is_decided_by_the_top_score_against_the_threshold(indexed: MemoryStore) -> None:
    question = "How long are carbon frames covered for?"
    top = retrieve(question, indexed, HashingEmbedder(), threshold=0.0).top_score
    assert retrieve(question, indexed, HashingEmbedder(), threshold=top).confident
    assert not retrieve(question, indexed, HashingEmbedder(), threshold=top + 1e-6).confident


def test_a_question_with_nothing_in_common_is_not_confident(indexed: MemoryStore) -> None:
    result = retrieve("photosynthesis chlorophyll", indexed, HashingEmbedder(), threshold=0.2)
    assert result.top_score < 0.2 and not result.confident


def test_an_empty_store_is_never_confident() -> None:
    result = retrieve("anything", MemoryStore(), HashingEmbedder(), threshold=0.0)
    assert result.hits == () and result.top_score == 0.0 and not result.confident


def test_an_empty_question_is_rejected(indexed: MemoryStore) -> None:
    with pytest.raises(ValueError):
        retrieve("   ", indexed, HashingEmbedder())


def test_retrieval_without_hits_is_not_confident_even_at_zero_threshold() -> None:
    assert not Retrieval("q", (), threshold=0.0).confident
    hit = Hit(StoredPassage(1, "a.md", "A", "text", 0, 4, None, ()), 0.0)
    assert Retrieval("q", (hit,), threshold=0.0).confident


def test_reingesting_skips_unchanged_files_and_replaces_changed_ones(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nThe first version says 30 days.")
    (tmp_path / "b.md").write_text("# B\n\nUnrelated text.")
    store, embedder = MemoryStore(), HashingEmbedder()

    first = ingest_folder(tmp_path, store, embedder)
    assert first.count("added") == 2 and store.counts() == (2, 2)

    second = ingest_folder(tmp_path, store, embedder)
    assert second.count("unchanged") == 2

    (tmp_path / "a.md").write_text("# A\n\nThe second version says 14 days. It has two sentences.")
    third = ingest_folder(tmp_path, store, embedder)
    assert [(path, outcome) for path, outcome, _ in third.files] == [
        ("a.md", "updated"),
        ("b.md", "unchanged"),
    ]
    texts = [hit.passage.text for hit in store.search(embedder.encode(["days"])[0], 10)]
    assert not any("30 days" in text for text in texts)
    assert any("14 days" in text for text in texts)


def test_a_new_chunk_size_reindexes_everything(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n\nOne sentence. Another sentence.")
    store, embedder = MemoryStore(), HashingEmbedder()
    ingest_folder(tmp_path, store, embedder, chunk_size=256)
    report = ingest_folder(tmp_path, store, embedder, chunk_size=16)
    assert report.count("updated") == 1
    stored = store.indexed_with("a.md")
    assert stored is not None and stored[1] == signature(embedder, 16)


def test_a_broken_file_is_reported_and_the_rest_still_go_in(tmp_path: Path) -> None:
    (tmp_path / "good.md").write_text("# Good\n\nFine.")
    (tmp_path / "bad.pdf").write_bytes(b"not a pdf")
    report = ingest_folder(tmp_path, MemoryStore(), HashingEmbedder())
    assert report.count("added") == 1 and report.count("failed") == 1
    assert next(detail for path, _, detail in report.files if path == "bad.pdf")


def test_the_store_rejects_vectors_that_do_not_match_the_passages(demo_docs: Path) -> None:
    from grounded.ingest import read_document

    document = read_document(demo_docs / "store.md", demo_docs)
    with pytest.raises(ValueError, match="passages but"):
        MemoryStore().replace_document(document, "sig", np.zeros((1, 384), dtype=np.float32))
