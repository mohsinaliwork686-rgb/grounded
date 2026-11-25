"""Writing the answer from the retrieved passages, with a citation for each claim.

Three answer models share one interface, a stream of text pieces with their citations:

- ``ClaudeAnswerer``: Claude with native citations. Each retrieved passage goes in as its own
  ``document`` block with citations enabled, so every citation names a passage by its position
  and carries the exact text it relied on. Nothing is parsed out of the prose.
- ``ExtractiveAnswerer``: no model, no key. The answer is the top passages themselves, each
  citing itself, so retrieval, refusal, the verifier and the page all work without a key.
- ``ScriptedAnswerer``: for tests. Plays back an answer written in the test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Protocol

from grounded.store import StoredPassage
from grounded.verify import Citation

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "low"
REFUSAL_SENTENCE = "Your documents don't cover this."
SYSTEM_PROMPT = files("grounded").joinpath("prompts/answer.md").read_text(encoding="utf-8")


@dataclass(frozen=True)
class TextPiece:
    """The next piece of the answer. ``block`` groups pieces that share citations."""

    block: int
    text: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Finished:
    model: str
    stop_reason: str
    input_tokens: int | None = None
    output_tokens: int | None = None


AnswerEvent = TextPiece | Finished


class AnswerModel(Protocol):
    @property
    def name(self) -> str: ...

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        """Pieces of the answer as they are written, then one ``Finished``."""
        ...


def document_title(passage: StoredPassage) -> str:
    where = " > ".join(passage.heading_path)
    page = f", page {passage.page}" if passage.page else ""
    return f"{passage.document_path}{page}" + (f": {where}" if where else "")


# ----------------------------------------------------------------------------- Claude


class ClaudeAnswerer:
    """Claude Opus 5.5 with native citations, streamed.

    Effort is set explicitly (Opus 5.5 defaults to medium): this is short question answering,
    so it starts at low and is measured. The server-side ``fallbacks`` option is on, so if the
    model's safety classifiers decline a request it is retried on a fallback model instead of
    failing; the final model is reported in ``Finished``.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        effort: str = DEFAULT_EFFORT,
        max_tokens: int = 16000,
        client: Any = None,
    ) -> None:
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._client = client

    @property
    def name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def request(self, question: str, passages: Sequence[StoredPassage]) -> dict[str, Any]:
        """The request body, kept separate so tests can check it without a network call."""
        documents = [
            {
                "type": "document",
                "source": {"type": "text", "media_type": "text/plain", "data": passage.text},
                "title": document_title(passage),
                "citations": {"enabled": True},
            }
            for passage in passages
        ]
        return {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self._effort},
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
            "messages": [
                {"role": "user", "content": [*documents, {"type": "text", "text": question}]}
            ],
        }

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        request = self.request(question, passages)
        with self._get_client().beta.messages.stream(**request) as stream:
            for event in stream:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "text_delta":
                    yield TextPiece(event.index, delta.text)
                elif delta.type == "citations_delta":
                    citation = to_citation(delta.citation, passages)
                    if citation is not None:
                        yield TextPiece(event.index, "", (citation,))
            final = stream.get_final_message()
        yield Finished(
            model=final.model,
            stop_reason=final.stop_reason or "end_turn",
            input_tokens=final.usage.input_tokens,
            output_tokens=final.usage.output_tokens,
        )


def to_citation(citation: Any, passages: Sequence[StoredPassage]) -> Citation | None:
    """Map a Claude citation to the retrieved passage it points at.

    A citation can only point at a document block that was sent, and those are exactly the
    retrieved passages, in order. An index outside them is dropped rather than trusted.
    """
    index = getattr(citation, "document_index", None)
    if not isinstance(index, int) or not 0 <= index < len(passages):
        return None
    return Citation(passage_id=passages[index].id, cited_text=str(citation.cited_text))


# ----------------------------------------------------------------------------- no model


class ExtractiveAnswerer:
    """The top passages, quoted. What someone without an Anthropic key sees."""

    def __init__(self, passages_used: int = 2) -> None:
        self._count = passages_used

    @property
    def name(self) -> str:
        return "extractive"

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        for block, passage in enumerate(passages[: self._count]):
            text = " ".join(passage.text.split())
            separator = " " if block else ""
            yield TextPiece(block, separator + text, (Citation(passage.id, passage.text),))
        yield Finished(model=self.name, stop_reason="end_turn")


class ScriptedAnswerer:
    """Plays back ``(text, [positions of cited passages])`` pairs. Tests only."""

    def __init__(self, script: Sequence[tuple[str, Sequence[int]]]) -> None:
        self._script = script
        self.calls = 0

    @property
    def name(self) -> str:
        return "scripted"

    def stream(self, question: str, passages: Sequence[StoredPassage]) -> Iterator[AnswerEvent]:
        self.calls += 1
        for block, (text, positions) in enumerate(self._script):
            citations = tuple(
                Citation(passages[i].id, passages[i].text) for i in positions if i < len(passages)
            )
            yield TextPiece(block, text, citations)
        yield Finished(model=self.name, stop_reason="end_turn")


def get_answerer(name: str) -> AnswerModel:
    """``auto`` picks Claude when an API key is set and the extractive answer otherwise."""
    if name == "auto":
        has_key = bool(
            os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        )
        name = DEFAULT_MODEL if has_key else "extractive"
    if name == "extractive":
        return ExtractiveAnswerer()
    return ClaudeAnswerer(name)
