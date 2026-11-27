"""The answer models, with the Anthropic client replaced by a recording fake. No network."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from grounded.generate import (
    REFUSAL_SENTENCE,
    SYSTEM_PROMPT,
    ClaudeAnswerer,
    ExtractiveAnswerer,
    Finished,
    ScriptedAnswerer,
    TextPiece,
    get_answerer,
    to_citation,
)
from grounded.store import StoredPassage

PASSAGES = [
    StoredPassage(
        11, "warranty.pdf", "Warranty", "Frames are covered for 2 years.", 0, 31, 1, ("Frames",)
    ),
    StoredPassage(
        12, "returns.md", "Returns", "Returns within 30 days.", 5, 28, None, ("Returns",)
    ),
]


def ns(**kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(**kwargs)


class FakeStream:
    def __init__(self, events: list[Any], final: Any) -> None:
        self._events, self._final = events, final

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._events)

    def get_final_message(self) -> Any:
        return self._final


class FakeClient:
    def __init__(self, events: list[Any], stop_reason: str = "end_turn") -> None:
        final = ns(
            model="claude-opus-5-5",
            stop_reason=stop_reason,
            usage=ns(input_tokens=812, output_tokens=64),
        )
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Messages:
            def stream(self, **request: Any) -> FakeStream:
                outer.requests.append(request)
                return FakeStream(events, final)

        self.beta = ns(messages=Messages())


def delta(index: int, **delta_fields: Any) -> SimpleNamespace:
    return ns(type="content_block_delta", index=index, delta=ns(**delta_fields))


def test_the_request_sends_each_passage_as_a_cited_document() -> None:
    request = ClaudeAnswerer().request("How long are frames covered?", PASSAGES)
    assert request["model"] == "claude-opus-5-5"
    assert request["output_config"] == {"effort": "medium"}
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["system"] == SYSTEM_PROMPT and REFUSAL_SENTENCE in SYSTEM_PROMPT
    content = request["messages"][0]["content"]
    assert [block["type"] for block in content] == ["document", "document", "text"]
    assert content[0]["source"] == {
        "type": "text",
        "media_type": "text/plain",
        "data": "Frames are covered for 2 years.",
    }
    assert content[0]["citations"] == {"enabled": True}
    assert content[0]["title"] == "warranty.pdf, page 1: Frames"
    assert content[1]["title"] == "returns.md: Returns"
    assert content[2] == {"type": "text", "text": "How long are frames covered?"}
    assert "temperature" not in request and "thinking" not in request


def test_streamed_citations_map_to_the_retrieved_passages() -> None:
    events = [
        ns(type="content_block_start", index=0, content_block=ns(type="thinking")),
        delta(
            1,
            type="citations_delta",
            citation=ns(
                type="char_location", document_index=0, cited_text="Frames are covered for 2 years."
            ),
        ),
        delta(1, type="text_delta", text="Frames are covered for 2 years."),
        delta(2, type="text_delta", text=" Ask us about parts."),
        delta(3, type="citations_delta", citation=ns(document_index=7, cited_text="nope")),
    ]
    client = FakeClient(events)
    out = list(ClaudeAnswerer(client=client).stream("q", PASSAGES))
    assert out[0] == TextPiece(1, "", (to_citation(events[1].delta.citation, PASSAGES),))
    assert out[0].citations[0].passage_id == 11
    assert out[1] == TextPiece(1, "Frames are covered for 2 years.")
    assert out[2] == TextPiece(2, " Ask us about parts.")
    # The citation to document 7 was never sent, so it is dropped.
    assert out[3] == Finished("claude-opus-5-5", "end_turn", 812, 64)
    assert len(out) == 4
    assert len(client.requests) == 1


def test_a_refusal_stop_reason_is_reported() -> None:
    out = list(ClaudeAnswerer(client=FakeClient([], stop_reason="refusal")).stream("q", PASSAGES))
    assert out == [Finished("claude-opus-5-5", "refusal", 812, 64)]


@pytest.mark.parametrize("index", [-1, 2, None, "0"])
def test_out_of_range_document_indexes_are_dropped(index: object) -> None:
    assert to_citation(ns(document_index=index, cited_text="x"), PASSAGES) is None


def test_the_extractive_answer_quotes_the_top_passages_and_cites_them() -> None:
    out = list(ExtractiveAnswerer().stream("q", PASSAGES))
    assert out[0] == TextPiece(0, "Frames are covered for 2 years.", (out[0].citations[0],))
    assert out[0].citations[0].passage_id == 11
    assert out[1].text == " Returns within 30 days." and out[1].citations[0].passage_id == 12
    assert out[2] == Finished("extractive", "end_turn")


def test_the_scripted_answer_cites_by_position() -> None:
    answerer = ScriptedAnswerer([("A. ", [0]), ("B.", [1, 5])])
    out = list(answerer.stream("q", PASSAGES))
    assert [p.citations[0].passage_id for p in out[:2]] == [11, 12]
    assert len(out[1].citations) == 1 and answerer.calls == 1


def test_auto_picks_claude_only_with_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    assert get_answerer("auto").name == "extractive"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    assert get_answerer("auto").name == "claude-opus-5-5"
    assert get_answerer("extractive").name == "extractive"
