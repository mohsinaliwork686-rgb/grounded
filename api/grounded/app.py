"""The HTTP API: ``POST /api/ask`` streams an answer as Server-Sent Events.

Events, in order:

- ``retrieval``: the passages found, their scores and the confidence decision;
- ``refused``: why there is no answer, then ``done``. No model call happens when retrieval isn't
  confident;
- ``text``: the next piece of the answer, with any citations attached to it;
- ``verdict``: one per sentence of the answer: supported, not_supported, no_citation or connective;
- ``error``: the answer model failed (the stream still ends with ``done``);
- ``done``: timings, the model that answered and token counts.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from grounded.config import Settings
from grounded.embed import Embedder, get_embedder
from grounded.generate import REFUSAL_SENTENCE, AnswerModel, Finished, get_answerer
from grounded.retrieve import retrieve
from grounded.store import PostgresStore, Store
from grounded.verify import Block, Citation, NliModel, Source, get_nli_model, verify

log = logging.getLogger("grounded")


@dataclass
class Components:
    store: Store
    embedder: Embedder
    answerer: AnswerModel
    nli: NliModel
    settings: Settings


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _citation(c: Citation) -> dict[str, Any]:
    return {"passage_id": c.passage_id, "cited_text": c.cited_text}


def _words(text: str) -> str:
    return " ".join(text.replace("’", "'").lower().split())


def ask_events(question: str, parts: Components) -> Iterator[str]:
    settings = parts.settings
    timings: dict[str, float] = {}
    started = time.perf_counter()

    def lap(name: str, since: float) -> float:
        now = time.perf_counter()
        timings[name] = round((now - since) * 1000, 1)
        return now

    def done(model: str | None, finished: Finished | None = None) -> str:
        timings["total_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return sse(
            "done",
            {
                "model": model,
                "timings": timings,
                "input_tokens": finished.input_tokens if finished else None,
                "output_tokens": finished.output_tokens if finished else None,
            },
        )

    retrieval = retrieve(
        question,
        parts.store,
        parts.embedder,
        k=settings.top_k,
        threshold=settings.refusal_threshold,
    )
    mark = lap("retrieval_ms", started)
    yield sse(
        "retrieval",
        {
            "passages": [
                {
                    "id": hit.passage.id,
                    "document": hit.passage.document_path,
                    "title": hit.passage.document_title,
                    "page": hit.passage.page,
                    "heading_path": list(hit.passage.heading_path),
                    "text": hit.passage.text,
                    "start": hit.passage.start,
                    "end": hit.passage.end,
                    "score": round(hit.score, 4),
                }
                for hit in retrieval.hits
            ],
            "top_score": round(retrieval.top_score, 4),
            "threshold": retrieval.threshold,
            "confident": retrieval.confident,
        },
    )

    if not retrieval.confident:
        yield sse(
            "refused",
            {
                "reason": "retrieval",
                "message": REFUSAL_SENTENCE,
                "detail": f"The best passage scored {retrieval.top_score:.2f}, "
                f"below the threshold of {retrieval.threshold:.2f}.",
            },
        )
        yield done(None)
        return

    passages = [hit.passage for hit in retrieval.hits]
    texts: dict[int, list[str]] = {}
    citations: dict[int, list[Citation]] = {}
    finished: Finished | None = None
    try:
        for event in parts.answerer.stream(question, passages):
            if isinstance(event, Finished):
                finished = event
                continue
            texts.setdefault(event.block, []).append(event.text)
            citations.setdefault(event.block, []).extend(event.citations)
            yield sse(
                "text",
                {
                    "block": event.block,
                    "text": event.text,
                    "citations": [_citation(c) for c in event.citations],
                },
            )
    except Exception as error:  # the model call failed mid-answer: report it, don't hang
        log.exception("answer model failed")
        yield sse("error", {"message": f"The answer model failed ({type(error).__name__})."})
        yield done(parts.answerer.name)
        return
    mark = lap("generation_ms", mark)
    model = finished.model if finished else parts.answerer.name

    if finished is not None and finished.stop_reason == "refusal":
        yield sse(
            "refused",
            {
                "reason": "model",
                "message": "The model declined to answer this question.",
                "detail": "Its safety classifiers declined the request, and so did the fallback.",
            },
        )
        yield done(model, finished)
        return

    blocks = [Block("".join(texts[b]), tuple(citations.get(b, ()))) for b in texts]
    if _words("".join(block.text for block in blocks)) == _words(REFUSAL_SENTENCE):
        yield sse(
            "refused",
            {
                "reason": "answer",
                "message": REFUSAL_SENTENCE,
                "detail": "The passages were related to the question but didn't answer it.",
            },
        )
        yield done(model, finished)
        return

    sources = {
        p.id: Source(p.text, " > ".join((p.document_title, *p.heading_path))) for p in passages
    }
    verdicts = verify(
        blocks,
        sources,
        parts.nli,
        threshold=settings.entailment_threshold,
    )
    lap("verification_ms", mark)
    for verdict in verdicts:
        best = verdict.best_entailment
        yield sse(
            "verdict",
            {
                "index": verdict.index,
                "text": verdict.text,
                "verdict": verdict.verdict,
                "entailment": None if best is None else round(best, 4),
                "checks": [
                    {
                        "passage_id": c.passage_id,
                        "entailment": round(c.scores.entailment, 4),
                        "neutral": round(c.scores.neutral, 4),
                        "contradiction": round(c.scores.contradiction, 4),
                    }
                    for c in verdict.checks
                ],
            },
        )
    yield done(model, finished)


def build_components(settings: Settings) -> Components:
    embedder = get_embedder(settings.embedding_model)
    return Components(
        store=PostgresStore(settings.database_url, embedder.dimensions),
        embedder=embedder,
        answerer=get_answerer(settings.answer_model),
        nli=get_nli_model(settings.nli_model),
        settings=settings,
    )


def create_app(components: Components | None = None) -> FastAPI:
    """With ``components`` (tests), use them; without, build them from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if components is None:
            parts = build_components(Settings.from_env())
            # Load both local models now, so the first question doesn't wait for them.
            parts.embedder.encode(["warm up"])
            parts.nli.predict([("warm up", "warm up")])
            app.state.parts = parts
        yield

    app = FastAPI(title="grounded", lifespan=lifespan)
    if components is not None:
        app.state.parts = components

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/ask")
    def ask(body: AskRequest, request: Request) -> StreamingResponse:
        parts: Components = request.app.state.parts
        return StreamingResponse(
            ask_events(body.question, parts),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
