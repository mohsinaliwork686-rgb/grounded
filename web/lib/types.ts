// The events of POST /api/ask, as api/grounded/app.py sends them.

export interface Passage {
  id: number;
  document: string;
  title: string;
  page: number | null;
  heading_path: string[];
  text: string;
  start: number;
  end: number;
  score: number;
}

export interface RetrievalEvent {
  passages: Passage[];
  top_score: number;
  threshold: number;
  confident: boolean;
}

export interface RefusedEvent {
  reason: "retrieval" | "answer" | "model";
  message: string;
  detail: string;
}

export interface Citation {
  passage_id: number;
  cited_text: string;
}

export interface TextEvent {
  block: number;
  text: string;
  citations: Citation[];
}

export type Verdict = "supported" | "not_supported" | "no_citation" | "connective";

export interface PassageCheck {
  passage_id: number;
  entailment: number;
  neutral: number;
  contradiction: number;
}

export interface VerdictEvent {
  index: number;
  text: string;
  verdict: Verdict;
  entailment: number | null;
  checks: PassageCheck[];
}

export interface ErrorEvent {
  message: string;
}

export interface DoneEvent {
  model: string | null;
  timings: {
    retrieval_ms?: number;
    generation_ms?: number;
    verification_ms?: number;
    total_ms?: number;
  };
  input_tokens: number | null;
  output_tokens: number | null;
}
