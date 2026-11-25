"""The SSE stream, with the scripted model, the hashing embedder and the overlap stand-in."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from grounded.app import Components, create_app
from grounded.config import Settings
from grounded.embed import HashingEmbedder
from grounded.generate import AnswerEvent, Finished, ScriptedAnswerer, TextPiece
from grounded.index import ingest_folder
from grounded.store import MemoryStore, StoredPassage
from grounded.verify import OverlapNli

QUESTION = "How long are carbon frames covered for?"


@pytest.fixture(scope="module")
def store() -> MemoryStore:
    store = MemoryStore()
    ingest_folder(Path(__file__).resolve().parents[2] / "demo" / "docs", store, HashingEmbedder())
    return store


def client_for(store: MemoryStore, answerer: object, threshold: float = 0.0) -> TestClient:
    parts = Components(
        store=store,
        embedder=HashingEmbedder(),
        answerer=answerer,  # type: ignore[arg-type]
        nli=OverlapNli(),
        settings=Settings(refusal_threshold=threshold, entailment_threshold=0.9),
    )
    return TestClient(create_app(parts))


def events(client: TestClient, question: str = QUESTION) -> list[tuple[str, dict]]:
    with client.stream("POST", "/api/ask", json={"question": question}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    parsed = []
    for chunk in body.strip().split("\n\n"):
        name_line, data_line = chunk.split("\n")
        parsed.append(
            (name_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: ")))
        )
    return parsed


def test_below_the_threshold_it_refuses_without_calling_the_model(store: MemoryStore) -> None:
    answerer = ScriptedAnswerer([("Should never be said.", [0])])
    out = events(client_for(store, answerer, threshold=0.99))
    assert [name for name, _ in out] == ["retrieval", "refused", "done"]
    assert answerer.calls == 0
    assert out[0][1]["confident"] is False and out[0][1]["threshold"] == 0.99
    assert out[1][1]["reason"] == "retrieval"
    assert out[1][1]["message"] == "Your documents don't cover this."
    assert out[2][1]["model"] is None


def test_an_answer_streams_text_then_a_verdict_per_sentence(store: MemoryStore) -> None:
    script = [
        ("Here's what the booklet says: ", [0]),
        ("Carbon frames are covered for 5 years. ", [0]),
        ("Carbon frames are covered for 10 years. ", [0]),
        ("You can also post it to us.", []),
    ]
    out = events(client_for(store, ScriptedAnswerer(script)))
    names = [name for name, _ in out]
    assert names == ["retrieval"] + ["text"] * 4 + ["verdict"] * 4 + ["done"]

    retrieval = out[0][1]
    assert retrieval["confident"] is True and len(retrieval["passages"]) == 5
    top = retrieval["passages"][0]
    assert top["document"] == "warranty.pdf" and top["page"] == 1
    assert "Carbon frames are covered for 5 years" in top["text"]

    first_text = out[1][1]
    assert first_text["block"] == 0 and first_text["citations"][0]["passage_id"] == top["id"]

    verdicts = [data for name, data in out if name == "verdict"]
    assert [v["verdict"] for v in verdicts] == [
        "connective",
        "supported",
        "not_supported",
        "no_citation",
    ]
    assert verdicts[1]["checks"][0]["passage_id"] == top["id"]
    assert verdicts[3]["entailment"] is None

    done = out[-1][1]
    assert done["model"] == "scripted"
    assert set(done["timings"]) == {"retrieval_ms", "generation_ms", "verification_ms", "total_ms"}


def test_when_the_model_says_the_documents_do_not_cover_it(store: MemoryStore) -> None:
    out = events(client_for(store, ScriptedAnswerer([("Your documents don’t cover this.", [])])))
    assert [name for name, _ in out] == ["retrieval", "text", "refused", "done"]
    assert out[2][1]["reason"] == "answer"


class Declining:
    name = "declining"

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        yield Finished("claude-opus-5-5", "refusal", 10, 0)


class Failing:
    name = "failing"

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        yield TextPiece(0, "Frames are")
        raise ConnectionError("network down")


def test_a_model_refusal_is_reported(store: MemoryStore) -> None:
    out = events(client_for(store, Declining()))
    assert [name for name, _ in out] == ["retrieval", "refused", "done"]
    assert out[1][1]["reason"] == "model"
    assert out[2][1]["input_tokens"] == 10


def test_a_failing_model_ends_the_stream_with_an_error(store: MemoryStore) -> None:
    out = events(client_for(store, Failing()))
    assert [name for name, _ in out] == ["retrieval", "text", "error", "done"]
    assert "ConnectionError" in out[2][1]["message"]


@pytest.mark.parametrize("body", [{"question": ""}, {}, {"question": "x" * 2001}])
def test_a_bad_request_is_rejected(store: MemoryStore, body: dict) -> None:
    response = client_for(store, ScriptedAnswerer([])).post("/api/ask", json=body)
    assert response.status_code == 422
