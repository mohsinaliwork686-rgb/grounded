// The state of one question, built up event by event as the stream arrives.

import type {
  Citation,
  DoneEvent,
  ErrorEvent,
  Passage,
  RefusedEvent,
  RetrievalEvent,
  TextEvent,
  VerdictEvent,
} from "./types";
import type { SseMessage } from "./sse";

export interface AnswerBlock {
  block: number;
  text: string;
  citations: Citation[];
}

export interface AskState {
  question: string;
  /** "streaming" until `done` arrives, the stream breaks or the request fails. */
  status: "idle" | "streaming" | "finished";
  retrieval: RetrievalEvent | null;
  blocks: AnswerBlock[];
  verdicts: VerdictEvent[];
  refusal: RefusedEvent | null;
  error: string | null;
  done: DoneEvent | null;
}

export const idle: AskState = {
  question: "",
  status: "idle",
  retrieval: null,
  blocks: [],
  verdicts: [],
  refusal: null,
  error: null,
  done: null,
};

export function started(question: string): AskState {
  return { ...idle, question, status: "streaming" };
}

/** The state after one more event. Unknown events are ignored. */
export function applyMessage(state: AskState, message: SseMessage): AskState {
  let data: unknown;
  try {
    data = JSON.parse(message.data);
  } catch {
    return failed(state, `The API sent a "${message.event}" event that isn't valid JSON.`);
  }
  switch (message.event) {
    case "retrieval":
      return { ...state, retrieval: data as RetrievalEvent };
    case "text":
      return { ...state, blocks: addText(state.blocks, data as TextEvent) };
    case "verdict":
      return { ...state, verdicts: [...state.verdicts, data as VerdictEvent] };
    case "refused":
      return { ...state, refusal: data as RefusedEvent };
    case "error":
      return { ...state, error: (data as ErrorEvent).message || "The API reported an error." };
    case "done":
      return { ...state, done: data as DoneEvent, status: "finished" };
    default:
      return state;
  }
}

export function failed(state: AskState, error: string): AskState {
  return { ...state, error: state.error ?? error, status: "finished" };
}

function addText(blocks: AnswerBlock[], event: TextEvent): AnswerBlock[] {
  const at = blocks.findIndex((b) => b.block === event.block);
  if (at === -1) {
    return [...blocks, { block: event.block, text: event.text, citations: [...event.citations] }];
  }
  const current = blocks[at];
  const next = blocks.slice();
  next[at] = {
    block: current.block,
    text: current.text + event.text,
    citations: [...current.citations, ...event.citations],
  };
  return next;
}

// ----------------------------------------------------------------------------- reading it back

/** A passage's marker number: its 1-based place in the retrieval list, as the side panel shows. */
export function markerNumbers(retrieval: RetrievalEvent | null): Map<number, number> {
  return new Map((retrieval?.passages ?? []).map((p, i) => [p.id, i + 1]));
}

export function passagesById(retrieval: RetrievalEvent | null): Map<number, Passage> {
  return new Map((retrieval?.passages ?? []).map((p) => [p.id, p]));
}

/** Passage ids in first-cited order, without repeats. */
export function citedIds(citations: Citation[]): number[] {
  return [...new Set(citations.map((c) => c.passage_id))];
}

/** Every passage the answer cites, from the text and from the verdicts' checks. */
export function allCitedIds(state: AskState): Set<number> {
  const ids = new Set<number>();
  for (const block of state.blocks) for (const c of block.citations) ids.add(c.passage_id);
  for (const verdict of state.verdicts) for (const c of verdict.checks) ids.add(c.passage_id);
  return ids;
}

export interface PlacedSentence {
  verdict: VerdictEvent;
  /** The exact text each cited passage was cited for, from the blocks this sentence overlaps. */
  citedText: Map<number, string[]>;
}

/**
 * Lines each verdict's sentence up with the answer text, to find the cited text behind each of
 * its markers, and returns what of the answer has no verdict yet (shown plainly until it has).
 */
export function placeSentences(
  blocks: AnswerBlock[],
  verdicts: VerdictEvent[],
): { sentences: PlacedSentence[]; rest: string } {
  const text = blocks.map((b) => b.text).join("");
  const bounds: { start: number; end: number; citations: Citation[] }[] = [];
  let offset = 0;
  for (const block of blocks) {
    bounds.push({ start: offset, end: offset + block.text.length, citations: block.citations });
    offset += block.text.length;
  }

  const byPassage = (citations: Citation[]) => {
    const found = new Map<number, string[]>();
    for (const c of citations) {
      const list = found.get(c.passage_id) ?? [];
      if (!list.includes(c.cited_text)) list.push(c.cited_text);
      found.set(c.passage_id, list);
    }
    return found;
  };
  const everywhere = byPassage(blocks.flatMap((b) => b.citations));

  let cursor = 0;
  const sentences = [...verdicts]
    .sort((a, b) => a.index - b.index)
    .map((verdict) => {
      const start = text.indexOf(verdict.text, cursor);
      if (start === -1) return { verdict, citedText: everywhere };
      const end = start + verdict.text.length;
      cursor = end;
      const overlapping = bounds.filter((b) => b.start < end && b.end > start);
      const citedText = byPassage(overlapping.flatMap((b) => b.citations));
      // A passage checked for this sentence but not cited by its blocks: use any citation of it.
      for (const check of verdict.checks) {
        if (!citedText.has(check.passage_id) && everywhere.has(check.passage_id)) {
          citedText.set(check.passage_id, everywhere.get(check.passage_id) ?? []);
        }
      }
      return { verdict, citedText };
    });
  return { sentences, rest: text.slice(cursor).trim() };
}

/**
 * Where each cited text sits in the passage, merged and in order. A cited text that isn't in
 * the passage exactly is looked for again ignoring differences in whitespace.
 */
export function citedRanges(passage: string, cited: string[]): [number, number][] {
  const ranges: [number, number][] = [];
  for (const quote of cited) {
    const trimmed = quote.trim();
    if (!trimmed) continue;
    const exact = passage.indexOf(trimmed);
    if (exact !== -1) {
      ranges.push([exact, exact + trimmed.length]);
      continue;
    }
    const pattern = trimmed
      .split(/\s+/)
      .map((word) => word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))
      .join("\\s+");
    const match = new RegExp(pattern).exec(passage);
    if (match) ranges.push([match.index, match.index + match[0].length]);
  }
  ranges.sort((a, b) => a[0] - b[0]);
  const merged: [number, number][] = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && range[0] <= last[1]) last[1] = Math.max(last[1], range[1]);
    else merged.push([range[0], range[1]]);
  }
  return merged;
}

export function describeModel(model: string | null): string {
  if (model === null) return "no model called";
  if (model.toLowerCase().includes("extractive")) return "extractive fallback, no API key";
  return model;
}

/**
 * A passage as it reads best: the source's hard line breaks become spaces, but a line break
 * before a list item stays. The length doesn't change, so cited ranges still line up.
 */
export function unwrap(text: string): string {
  return text.replace(/\n(?![ \t]*(?:[-*+]|\d+[.)])[ \t])/g, " ");
}
