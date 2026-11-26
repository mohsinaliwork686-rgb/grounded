# grounded: implementation plan

Written on Monday 23 November 2026, before any code. Built from 23 to 27 November.

> Ask your documents. Every sentence carries a citation, or it does not get shown as trustworthy.

## 1. Problem

A question-answering app over your own documents is the most common thing people build with a
language model, and its main failure is quiet: the answer reads well, cites a real passage, and says
something the passage never claimed. Pasting retrieved text into the prompt is half of grounding.
Checking that the answer actually used it is the other half, and almost nobody does it.

The second failure is answering at all when the documents don't cover the question. Models produce
something plausible because that is what they do. For a refund policy, a confident wrong answer is
worse than "your documents don't say".

grounded does both halves: it refuses when retrieval is weak, and it checks every sentence of the
answer against the passage it cites, showing which sentences are supported and which are not.

## 2. What version one does not have

Scope is the real risk this week, so these are decided now and held:

- **No user accounts, no sign-in.** One local user.
- **No document management interface.** Documents are a folder; `ingest` reads it. No upload page,
  no editing, no deleting from the UI.
- **No multi-turn conversation.** Each question stands alone. No memory of earlier questions.
- **Markdown and PDF only.** No Word, HTML, images or scanned PDFs (no OCR).
- **No fine-tuning, no reranker model, no hybrid keyword search.** One embedding model, one store.
- **No live hosted demo.** `docker compose up` runs it.

## 3. What it does

```
  documents --> chunk --> embed --> store (PostgreSQL + pgvector)
                                          |
  question ------------------------> retrieve top k + confidence
                                          |
                          is retrieval confident enough?
                            no --> "Your documents don't cover this." (no model call)
                            yes
                                          v
                Claude answers with native citations to the retrieved passages
                                          |
                                          v
             each sentence checked against its cited passages by a local NLI model
                                          |
                                          v
       stream to the page: text, citations, then a verdict per sentence:
       supported / not supported / no citation
```

**Ingest.** `grounded ingest ./docs` reads Markdown and PDF files, splits them into passages and
records where each came from: file, page (PDF), heading path (Markdown), and character offsets. A
passage can always be shown in context.

**Ask.** `POST /api/ask {"question": "..."}` returns a Server-Sent Events stream:

| Event | Data |
|-------|------|
| `retrieval` | the passages retrieved, their scores, and the confidence decision |
| `refused` | why (top score below the threshold), then the stream ends |
| `text` | the next piece of the answer, with any citations attached to it |
| `verdict` | per sentence: supported, not supported or no citation, with the NLI scores |
| `done` | timings and token counts |

**The page** (Next.js): a question box, the answer streaming in, each sentence with its citation
markers (hover shows the cited passage) and a mark for its verdict, and a side panel with every
retrieved passage and its score, so you can see what the model was working from.

## 4. Decisions made on Monday

### 4a. Refusal: when to say "I don't know"

**Refuse before calling the model when the best retrieved passage scores below a threshold.** The
score is the cosine similarity of the question to the top passage. The threshold is not guessed: a
labelled set of questions for the demo documents, about half answerable and half not (some close to
the topic, some far off), gives the share of answerable questions wrongly refused and unanswerable
ones wrongly answered at each threshold. The threshold that keeps wrongly answered unanswerable
questions near zero, at an acceptable cost in refusals, is the default, and the chart goes in the
README. The same labelled questions check retrieval.

*The selection rule, written on Wednesday 25 November before the first run:* each unanswerable
question let through costs 2 and each answerable question refused costs 1, because a confident
wrong answer is worse than a refusal. The threshold is the middle of the widest run of thresholds
(in steps of 0.01) with the lowest total cost, which leaves the most room on both sides.

A second layer: the prompt tells Claude to say the documents don't cover it if the passages don't
answer the question, which catches questions that retrieve something related but not the answer.
The verifier catches what slips through both.

**The demo that carries the README:** ask something the documents don't cover and watch it refuse,
then ask something they do and watch each sentence get its citation and a supported mark, then one
where the model goes a step beyond the passage and the verifier flags that sentence.

### 4b. Citations: Claude's native citations, not numbered markers

The retrieved passages go to Claude as `document` content blocks with `citations: {enabled: true}`.
The answer comes back split into text blocks, and each cited block carries the exact `cited_text`
and its location in the passage. That is more reliable than asking the model to write `[3]` markers
and parsing them, and it means the verifier gets the exact text the model says it relied on.

Model: **Claude Opus 5.5** (`claude-opus-5-5`), streamed. Effort set explicitly and measured on the
labelled questions (start at `low`, since this is short-form Q&A).
*(25 November: not measured. It needs a paid API key, which this project doesn't use; the default
stays at `low`, unmeasured, and the README says so. `eval/answers.py` runs the measurement for
anyone with a key.)* The server-side `fallbacks`
option is on, so a refusal by the model's safety classifiers falls back instead of failing. The
model is configuration, not code.

### 4c. Verification: a local NLI model, independent of the answering model

Each sentence of the answer and each passage it cites go to
[`cross-encoder/nli-deberta-v3-small`](https://hf.co/cross-encoder/nli-deberta-v3-small) (Apache-2.0,
142M parameters, runs on a CPU), which returns entailment, contradiction or neutral. A sentence is:

- **supported** if at least one cited passage entails it above a threshold;
- **not supported** if no cited passage entails it (neutral or contradiction);
- **no citation** if it cites nothing. Short connective sentences ("Here's what the policy says.")
  are exempt by a small rule, measured like the rest.

Why not ask Claude again: a second model call per sentence costs money and time, gives different
answers on different runs, and asks the same model to mark its own work. A local NLI model is free,
deterministic and independent. The entailment threshold is measured on a labelled set of
sentence and passage pairs: supported sentences, invented ones, ones that change a number or a
condition, and paraphrases. Its accuracy goes in the README, including where it fails.

*The selection rule, written on Wednesday 25 November before the first run:* the same as for
refusal. Marking an unsupported sentence as supported costs 2 (that is the failure the verifier
exists to catch) and flagging a supported sentence costs 1. The entailment threshold is the middle
of the widest run of lowest-cost thresholds, in steps of 0.01. The premise is the whole cited
passage, not only the cited span, so a sentence that needs two clauses of the passage can still be
supported. The connective-sentence exemption is deliberately narrow: an uncited sentence is exempt
only if it ends with a colon or is at most eight words, has no digits, and starts with a lead-in
("Here", "In short", "According to", "Based on").

### 4d. Chunking, embeddings and storage

- **Start from what ragsweep (week 2) measured:** dense retrieval with `all-MiniLM-L6-v2` and fixed
  256-character chunks with no overlap was best on its corpus. On Tuesday, ragsweep itself runs on
  grounded's demo documents and labelled questions to confirm or change that, since it reads the
  same format.
- Chunks keep their source offsets so they can be shown in context and cited exactly.
- **PostgreSQL with pgvector** stores passages and embeddings. One container, real SQL, and the
  retrieval query is ordinary SQL a reviewer can read.

### 4e. Running without an API key

Tests and CI never call a model: a fake model returns scripted answers with citations. For someone
who runs the demo without an Anthropic key, an extractive fallback builds the answer from the top
passages' own sentences (each citing itself), so retrieval, refusal, the verifier and the page all
still work. The README says plainly that real answers need a key.

## 5. Data model

| Table | What it holds |
|-------|---------------|
| `documents` | id, path, kind (markdown or pdf), title, content hash, ingested at |
| `passages` | id, document id, text, start and end offsets, page, heading path, embedding `vector(384)` |

Re-ingesting a changed file replaces its passages; an unchanged file (same hash) is skipped.

## 6. Projects

| Path | Job |
|------|-----|
| `api/grounded/ingest.py` | Markdown and PDF into passages with source positions |
| `api/grounded/retrieve.py` | Vector search and the confidence decision |
| `api/grounded/generate.py` | The prompt and the Claude call with citations; the fake and extractive models |
| `api/grounded/verify.py` | Sentence splitting and the NLI check. **The differentiator** |
| `api/grounded/app.py` | FastAPI routes and the SSE stream |
| `api/tests/` | pytest, with PostgreSQL in Testcontainers and the fake model |
| `web/` | Next.js: chat, citation popovers, verdict marks, the retrieved-context panel |
| `eval/` | The labelled questions and sentence pairs, and the scripts that pick both thresholds |
| `demo/docs/` | The demo documents |

The system prompt is linted with promptlint (week 1) in CI.

## 7. The demo documents

A handbook for a made-up bike shop, written for this project so it can be published: returns,
warranty, shipping, repairs, store hours and a membership scheme, as Markdown, plus a warranty
booklet as a PDF. Some facts are deliberately close to each other (30 days for returns, 2 years for
the frame warranty, 90 days for parts) so a careless answer mixes them up and the verifier has
something to catch.

## 8. Test plan

| Area | Must work | Must not happen |
|------|-----------|-----------------|
| Ingest | Markdown with headings, a PDF, an empty file; offsets and pages survive chunking | a passage that can't be traced to its source |
| Retrieval | a known question returns its known passage | the confidence score ignored |
| Refusal | fires below the threshold, with no model call | a model call for an unanswerable question below the threshold |
| Generation | citations map to the right passages; the fallback model works without a key | a citation pointing at a passage that wasn't retrieved |
| Verifier | a supported sentence passes; an invented one and a changed number fail; an uncited one is flagged | a sentence marked supported by a passage it doesn't cite |
| API | the SSE events arrive in order, with a fake model | tests that call a real model |
| End to end | one test over the demo documents: answer, refuse, flag | |
| Web | the page renders a streamed answer, a refusal and a flagged sentence | |

## 9. Build order

- **Monday:** this plan; the repo; the demo documents; the labelled questions and sentence pairs;
  the skeleton (API, web, compose) and CI.
- **Tuesday:** ingest, embeddings, pgvector, retrieval; ragsweep on the demo set; the refusal
  threshold measured.
- **Wednesday:** generation with citations and streaming; the verifier and its threshold measured;
  the SSE API.
- **Thursday:** the web page; `docker compose up` working end to end.
- **Friday:** the end-to-end test, the README with the demo GIF and both charts, publish, the
  weekly updates.

*Rescheduled on Wednesday 25 November:* Tuesday was missed, so Tuesday's and Wednesday's work both
happen on Wednesday, in the order above. Monday's web and compose skeleton moves to Thursday with
the page it belongs to. Nothing is cut and Friday's publish stays.

*Thursday 26 November, as built:* the page is Next.js with plain CSS, one route and a server-side
proxy (`/api/ask`) that streams the API's events through, so the browser only talks to its own
origin. The API image pins the model libraries to the versions the thresholds were measured with
and downloads both local models at build time, so it answers offline. `docker compose up --build`
starts PostgreSQL, the API (which indexes `demo/docs`, or any folder you mount) and the page.
Checked end to end without a key; with a key it uses the same request the unit tests check, but no
paid call was made.

## 10. Risks

| Risk | What I'll do |
|------|--------------|
| The biggest project of the ten in five days | The non-goals above. The web page stays plain: one page, no routing |
| The NLI model is wrong on paraphrases or long passages | Measure it and publish the failures; passages are short (256 characters), which suits it |
| The threshold is tuned to the demo set only | Say so in the README; the eval scripts let anyone re-run it on their own documents |
| Model downloads make CI slow | Cache the model files in CI; unit tests use the fake model and a tiny embedding stub |
| Docker on this machine also starts an unrelated cluster | Tests use their own Testcontainers PostgreSQL |

## 11. Definition of done

- [ ] Public repo, description and topics
- [ ] This plan, committed before the code
- [ ] README: the problem first, a demo GIF that shows a refusal, how to run it, the refusal and
      verifier measurements, and what it doesn't promise
- [ ] Every row of the test plan covered
- [ ] CI green: lint, types, the API tests against real PostgreSQL, the web tests, promptlint
- [ ] `docker compose up` works, with and without an Anthropic key
