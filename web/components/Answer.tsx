"use client";

import {
  citedIds,
  describeModel,
  markerNumbers,
  passagesById,
  placeSentences,
  type AnswerBlock,
  type AskState,
} from "../lib/answer";
import type { DoneEvent, Passage, RefusedEvent, VerdictEvent } from "../lib/types";
import { CitationMarker } from "./CitationMarker";
import { ErrorIcon, InfoIcon } from "./icons";
import { VerdictMark } from "./Verdict";

// A word joiner: keeps the markers on the same line as the word before them.
const GLUE = "\u2060";

interface Lookup {
  numbers: Map<number, number>;
  passages: Map<number, Passage>;
}

/** The answer area: what's happening, then the answer, the refusal or the error. */
export function Answer({ state }: { state: AskState }) {
  const lookup: Lookup = {
    numbers: markerNumbers(state.retrieval),
    passages: passagesById(state.retrieval),
  };
  const streaming = state.status === "streaming";
  const checked = state.verdicts.length > 0;

  return (
    <section className="answer" aria-labelledby="answer-heading" aria-busy={streaming}>
      <h2 id="answer-heading">Answer</h2>
      <p className="asked">
        <span className="asked-label">You asked:</span> {state.question}
      </p>

      {state.refusal ? (
        <Refusal refusal={state.refusal} />
      ) : checked ? (
        <Sentences blocks={state.blocks} verdicts={state.verdicts} lookup={lookup} />
      ) : state.blocks.length > 0 ? (
        <>
          <Blocks blocks={state.blocks} lookup={lookup} />
          {streaming && (
            <p className="progress">Each sentence is checked against its sources when the answer is complete…</p>
          )}
        </>
      ) : streaming ? (
        <p className="progress">{state.retrieval ? "Writing the answer…" : "Searching the documents…"}</p>
      ) : null}

      {state.error && (
        <div className="notice notice-error" role="alert">
          <ErrorIcon />
          <div>
            <p className="notice-title">Something went wrong</p>
            <p>{state.error}</p>
          </div>
        </div>
      )}

      {state.done && <RunInfo done={state.done} />}
    </section>
  );
}

interface MarkersProps {
  ids: number[];
  lookup: Lookup;
  cited: (id: number) => string[];
  entailment?: (id: number) => number | undefined;
}

function Markers({ ids, lookup, cited, entailment }: MarkersProps) {
  const known = ids.filter((id) => lookup.passages.has(id));
  if (known.length === 0) return null;
  return (
    <span className="markers">
      {GLUE}
      {known.map((id) => (
        <CitationMarker
          key={id}
          n={lookup.numbers.get(id) ?? 0}
          passage={lookup.passages.get(id) as Passage}
          cited={cited(id)}
          entailment={entailment?.(id)}
        />
      ))}
    </span>
  );
}

/** While the answer streams: Claude's text blocks, each followed by the markers it cites. */
function Blocks({ blocks, lookup }: { blocks: AnswerBlock[]; lookup: Lookup }) {
  return (
    <div className="answer-text" data-view="blocks">
      {blocks.map((block) => (
        <span key={block.block} className="block">
          {block.text}
          <Markers
            ids={citedIds(block.citations)}
            lookup={lookup}
            cited={(id) => block.citations.filter((c) => c.passage_id === id).map((c) => c.cited_text)}
          />
        </span>
      ))}
    </div>
  );
}

/** Once the verdicts arrive: each sentence with its markers and the verifier's mark. */
function Sentences({
  blocks,
  verdicts,
  lookup,
}: {
  blocks: AnswerBlock[];
  verdicts: VerdictEvent[];
  lookup: Lookup;
}) {
  const { sentences, rest } = placeSentences(blocks, verdicts);
  const counts = { supported: 0, not_supported: 0, no_citation: 0 };
  for (const { verdict } of sentences) if (verdict.verdict !== "connective") counts[verdict.verdict] += 1;
  const summary = [
    counts.supported && `${counts.supported} supported`,
    counts.not_supported && `${counts.not_supported} not supported`,
    counts.no_citation && `${counts.no_citation} without a citation`,
  ].filter(Boolean);

  return (
    <>
      {summary.length > 0 && (
        <p className="verdict-summary">Checked against the passages each sentence cites: {summary.join(", ")}.</p>
      )}
      <ol className="sentences" data-view="sentences">
        {sentences.map(({ verdict, citedText }) => {
          const entailment = new Map(verdict.checks.map((c) => [c.passage_id, c.entailment]));
          return (
            <li key={verdict.index} className={`sentence sentence-${verdict.verdict}`}>
              <span className="sentence-body">
                <span className="sentence-text">{verdict.text}</span>
                <Markers
                  ids={verdict.checks.map((c) => c.passage_id)}
                  lookup={lookup}
                  cited={(id) => citedText.get(id) ?? []}
                  entailment={(id) => entailment.get(id)}
                />
              </span>
              <VerdictMark verdict={verdict.verdict} entailment={verdict.entailment} />
            </li>
          );
        })}
      </ol>
      {rest && <p className="answer-rest">{rest}</p>}
    </>
  );
}

const REFUSAL_NOTES: Record<RefusedEvent["reason"], string> = {
  retrieval: "No passage was close enough to the question, so no model was asked to answer it.",
  answer: "The model read the passages and found they don't answer the question.",
  model: "The model declined to answer.",
};

/** A refusal is the app working as designed: shown as an answer, not as an error. */
function Refusal({ refusal }: { refusal: RefusedEvent }) {
  return (
    <div className="notice notice-refusal">
      <InfoIcon />
      <div>
        <p className="notice-title">{refusal.message}</p>
        {refusal.detail && <p>{refusal.detail}</p>}
        {REFUSAL_NOTES[refusal.reason] && <p className="notice-note">{REFUSAL_NOTES[refusal.reason]}</p>}
      </div>
    </div>
  );
}

function ms(value: number | undefined): string | null {
  if (value === undefined) return null;
  return `${Math.round(value).toLocaleString("en-GB")}\u00a0ms`;
}

/** The last line: which model answered and how long each step took. */
function RunInfo({ done }: { done: DoneEvent }) {
  const steps = [
    ["retrieval", ms(done.timings.retrieval_ms)],
    ["generation", ms(done.timings.generation_ms)],
    ["verification", ms(done.timings.verification_ms)],
    ["total", ms(done.timings.total_ms)],
  ].filter(([, value]) => value !== null);
  const tokens =
    done.input_tokens !== null && done.output_tokens !== null
      ? `${done.input_tokens.toLocaleString("en-GB")} tokens in, ${done.output_tokens.toLocaleString("en-GB")} out`
      : null;
  const model = done.model === null ? "No model called" : `Model: ${describeModel(done.model)}`;
  return (
    <footer className="run-info">
      <p>{tokens ? `${model} · ${tokens}` : model}</p>
      {/* Non-breaking spaces inside each step, so a line only wraps between steps. */}
      <p>{steps.map(([name, value]) => `${name}\u00a0${value}`).join(" · ")}</p>
    </footer>
  );
}
