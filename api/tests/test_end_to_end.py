"""The whole path once, over the demo documents: the CLI indexes them into a real PostgreSQL,
then the API answers one question, refuses one and flags a sentence in a third.

The models are the test stand-ins (hashing embedder, overlap checker, scripted answers), so this
checks that the pieces fit together, not how good the real models are; eval/ measures that.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from grounded.app import Components, create_app
from grounded.cli import main
from grounded.config import Settings
from grounded.embed import HashingEmbedder
from grounded.generate import AnswerEvent, Finished, TextPiece
from grounded.store import PostgresStore, StoredPassage
from grounded.verify import Citation, OverlapNli

# With the hashing embedder, the answerable questions below score 0.33 and above and the
# unanswerable one 0.20 (the real threshold, 0.55, is measured for the real embedding model).
THRESHOLD = 0.3

# For each question, the answer as (sentence, text of the passage it cites).
ANSWERS = {
    "How long are carbon frames covered for?": [
        ("Carbon frames are covered for 5 years. ", "Carbon frames are covered for 5 years"),
    ],
    "Can I return a bike I've ridden 30 km?": [
        (
            "A bike can be returned within 30 days if it has been ridden less than 50 km. ",
            "ridden less than 50 km",
        ),
        ("Bikes can be returned within 60 days.", "ridden less than 50 km"),
    ],
}


class DemoAnswerer:
    """Answers each demo question from the script above, citing the passage that holds the text."""

    def __init__(self) -> None:
        self.questions: list[str] = []

    @property
    def name(self) -> str:
        return "scripted"

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        self.questions.append(question)
        for block, (text, evidence) in enumerate(ANSWERS[question]):
            cited = next(p for p in passages if evidence in p.text)
            yield TextPiece(block, text, (Citation(cited.id, cited.text),))
        yield Finished(model=self.name, stop_reason="end_turn")


def ask(client: TestClient, question: str) -> list[tuple[str, dict]]:
    response = client.post("/api/ask", json={"question": question})
    assert response.status_code == 200
    events = []
    for chunk in response.text.strip().split("\n\n"):
        name, data = chunk.split("\n")
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def verdicts(events: list[tuple[str, dict]]) -> list[tuple[str, str]]:
    return [(data["text"], data["verdict"]) for name, data in events if name == "verdict"]


def test_answer_refuse_and_flag_over_the_demo_documents(postgres_url: str, demo_docs: Path) -> None:
    with PostgresStore(postgres_url, 384) as store:
        store._connection.execute("TRUNCATE documents CASCADE")

    command = ["ingest", str(demo_docs), "--database-url", postgres_url, "--model", "hashing"]
    assert main(command) == 0

    answerer = DemoAnswerer()
    with PostgresStore(postgres_url, 384) as store:
        assert store.counts() == (6, 29)
        parts = Components(
            store=store,
            embedder=HashingEmbedder(),
            answerer=answerer,
            nli=OverlapNli(),
            settings=Settings(refusal_threshold=THRESHOLD, entailment_threshold=0.9),
        )
        client = TestClient(create_app(parts))

        answered = ask(client, "How long are carbon frames covered for?")
        assert answered[0][1]["confident"] is True
        assert answered[0][1]["passages"][0]["document"] == "warranty.pdf"
        assert verdicts(answered) == [("Carbon frames are covered for 5 years.", "supported")]

        refused = ask(client, "Do you match online prices?")
        assert [name for name, _ in refused] == ["retrieval", "refused", "done"]
        assert refused[1][1]["reason"] == "retrieval"

        flagged = ask(client, "Can I return a bike I've ridden 30 km?")
        assert verdicts(flagged) == [
            (
                "A bike can be returned within 30 days if it has been ridden less than 50 km.",
                "supported",
            ),
            ("Bikes can be returned within 60 days.", "not_supported"),
        ]

    # The refused question never reached the model.
    assert answerer.questions == [
        "How long are carbon frames covered for?",
        "Can I return a bike I've ridden 30 km?",
    ]
