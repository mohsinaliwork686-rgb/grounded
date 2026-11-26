"use client";

import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { allCitedIds, applyMessage, failed, idle, started, type AskState } from "../lib/answer";
import { readSse } from "../lib/sse";
import { Answer } from "./Answer";
import { Passages } from "./Passages";
import { NoCitationIcon, NotSupportedIcon, SupportedIcon } from "./icons";

/** From the demo documents: two they answer, one they don't cover. */
export const EXAMPLES = [
  "Can I return a bike I've already ridden?",
  "Do you match online prices?",
  "How long are carbon frames covered for?",
];

async function describeFailure(response: Response): Promise<string> {
  let detail = "";
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object") {
      const { message, detail: reason } = body as { message?: unknown; detail?: unknown };
      if (typeof message === "string") detail = message;
      else if (typeof reason === "string") detail = reason;
    }
  } catch {
    // Not JSON: the status line says enough.
  }
  const status = `${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
  return `The API answered ${status}.${detail ? ` ${detail}` : ""}`;
}

export function Ask() {
  const [question, setQuestion] = useState("");
  const [state, setState] = useState<AskState>(idle);
  const current = useRef<AbortController | null>(null);

  useEffect(() => () => current.current?.abort(), []);

  const ask = useCallback(async (text: string) => {
    const asked = text.trim();
    if (!asked) return;
    current.current?.abort();
    const controller = new AbortController();
    current.current = controller;
    const update = (next: (state: AskState) => AskState) => {
      if (!controller.signal.aborted) setState(next);
    };
    setState(started(asked));

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ question: asked }),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        const message = response.ok ? "The API sent an empty response." : await describeFailure(response);
        update((state) => failed(state, message));
        return;
      }
      for await (const message of readSse(response.body)) {
        if (controller.signal.aborted) return;
        update((state) => applyMessage(state, message));
      }
      update((state) =>
        state.status === "finished" ? state : failed(state, "The answer stopped before it finished."),
      );
    } catch {
      if (controller.signal.aborted) return;
      update((state) => failed(state, "The API couldn't be reached. Check that it is running and try again."));
    }
  }, []);

  const onSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    void ask(question);
  };

  const askExample = (example: string) => {
    setQuestion(example);
    void ask(example);
  };

  const streaming = state.status === "streaming";

  return (
    <>
      <section className="ask" aria-label="Ask a question">
        <form className="ask-form" onSubmit={onSubmit}>
          <label htmlFor="question">Your question</label>
          <div className="ask-row">
            <input
              id="question"
              name="question"
              type="text"
              autoComplete="off"
              maxLength={2000}
              required
              value={question}
              placeholder="Ask about returns, warranty, delivery, repairs…"
              onChange={(event) => setQuestion(event.target.value)}
            />
            <button type="submit" className="ask-button" disabled={!question.trim()}>
              Ask
            </button>
          </div>
        </form>
        <div className="examples">
          <span className="examples-label" id="examples-label">
            Try:
          </span>
          <ul aria-labelledby="examples-label">
            {EXAMPLES.map((example) => (
              <li key={example}>
                <button type="button" className="example" onClick={() => askExample(example)}>
                  {example}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <p className="visually-hidden" role="status">
        {streaming
          ? "Answering…"
          : state.status === "finished"
            ? state.error
              ? "The answer failed."
              : state.refusal
                ? "Refused: the documents don't cover this."
                : "Answer complete."
            : ""}
      </p>

      {state.status === "idle" ? (
        <Intro />
      ) : (
        <div className="results">
          <Answer state={state} />
          {state.retrieval && <Passages retrieval={state.retrieval} cited={allCitedIds(state)} />}
        </div>
      )}
    </>
  );
}

function Intro() {
  return (
    <section className="intro" aria-labelledby="intro-heading">
      <h2 id="intro-heading">How to read an answer</h2>
      <p>
        The answer streams in with a citation marker like <span className="marker-sample">[1]</span> after each
        claim. Hover over a marker, or move to it with the Tab key, to see the passage it cites. When the answer is
        complete, a local model checks every sentence against the passages it cites and marks it:
      </p>
      <ul className="legend">
        <li>
          <span className="verdict verdict-supported">
            <SupportedIcon />
            <span className="verdict-label">Supported</span>
          </span>
          a cited passage says this.
        </li>
        <li>
          <span className="verdict verdict-not_supported">
            <NotSupportedIcon />
            <span className="verdict-label">Not supported</span>
          </span>
          it cites passages, but none of them says this.
        </li>
        <li>
          <span className="verdict verdict-no_citation">
            <NoCitationIcon />
            <span className="verdict-label">No citation</span>
          </span>
          it makes a claim without citing anything.
        </li>
      </ul>
      <p>When the documents don&apos;t cover a question, it says so instead of answering.</p>
    </section>
  );
}
