# grounded

[![ci](https://github.com/mohsinaliwork686-rgb/grounded/actions/workflows/ci.yml/badge.svg)](https://github.com/mohsinaliwork686-rgb/grounded/actions/workflows/ci.yml)
[![MIT licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

**Ask your documents, and see which sentences you can trust.** Every sentence of the answer cites
the passage it came from, a second model checks that the passage really says it, and when your
documents don't cover the question, it says so instead of making something up.

FastAPI, PostgreSQL with pgvector, Claude with native citations, a small local NLI model as the
checker, and a Next.js page that streams the answer with each sentence marked. The thresholds that
decide when to refuse and when to flag were measured on labelled data, and the results are below.

![A question the documents don't cover gets "Your documents don't cover this." Then a question
they do cover is answered, each sentence with its citation numbers and a Supported mark, and a
citation's popover shows the passage it came from](docs/images/demo.gif)

*Recorded without an API key, so the answer is built from the passages' own sentences. With a key,
Claude writes it; a real Claude answer with a flagged sentence is [further down](#what-it-does).*

## The problem

Answering questions from your own documents is the most common thing people build with a language
model, and its usual failure is quiet. The answer reads well and cites a real passage, but says
something the passage never said. Putting the retrieved text in the prompt is half the job;
checking that the answer used it is the other half, and it's usually skipped.

The second failure is answering at all when the documents don't cover the question. A model will
produce something plausible, and for a refund policy a confident wrong answer is worse than "your
documents don't say".

grounded does both halves. It refuses when retrieval finds nothing close enough, and it checks every
sentence of the answer against the passage it cites, so you can see which ones to trust.

## What it does

- **Refuses before calling the model** when the best passage scores below a measured threshold. No
  model call, no cost, and the page says why.
- **Cites with Claude's native citations.** Each retrieved passage goes to Claude as its own
  document, and each part of the answer comes back pointing at the exact passages it used. Nothing
  is parsed out of the prose.
- **Checks every sentence** with `cross-encoder/nli-deberta-v3-small`, an entailment model that runs
  on the CPU. A sentence is *supported* when a passage it cites entails it, *not supported* when
  none does, and *no citation* when it cites nothing.
- **Shows its work.** The page streams the answer, puts a mark on each sentence, shows the cited
  passage when you hover a citation, and lists every retrieved passage with its score against the
  threshold.
- **Works without an API key.** The answer is then built from the top passages' own sentences, so
  retrieval, refusal, the checker and the page all still run. Real answers need a key.

Here is a flag from a real Claude answer. The booklet says "Frames are covered for 2 years …
Carbon frames are covered for 5 years"; Claude's second sentence calls the 2-year ones "standard
frames", which the booklet never says, so that sentence is marked for checking:

![An answer about carbon frames: the first sentence, citing the warranty booklet page 1, is marked
Supported; the second, "This compares with standard frames, which are covered for 2 years", is
marked Not supported](docs/images/claude-flagged.png)

## Try it

Needs Docker.

```sh
git clone https://github.com/mohsinaliwork686-rgb/grounded
cd grounded
docker compose up --build
```

Open http://localhost:3000. It indexes the demo handbook (a made-up bike shop's returns, warranty,
shipping, repairs, opening hours and membership: five Markdown files and a PDF) and answers without
a key. The first build downloads PyTorch for the CPU and both models, and the API image ends up
about 3 GB; after that it starts in under a minute and runs offline.

With Claude answering, pass your key from the shell so it isn't saved in a file or your history:

```sh
read -s ANTHROPIC_API_KEY && export ANTHROPIC_API_KEY
docker compose up --build
```

With your own documents (Markdown and PDF):

```sh
GROUNDED_DOCS_DIR=/path/to/your/docs docker compose up --build
```

Changed files are re-indexed when the API starts; unchanged ones are skipped. Files you delete stay
in the index until `docker compose down -v`.

## What the measurements found

The thresholds come from two labelled sets written before any code ran: 47 questions about the demo
handbook (28 answerable, 19 not) and 52 sentence and passage pairs. Each threshold was picked by a
rule fixed in advance: a wrong answer or a false "supported" costs 2, a refusal or a false flag
costs 1, and the threshold is the middle of the widest run of lowest-cost values. The full method,
tables and caveats are in [eval/README.md](eval/README.md).

**Retrieval.** With `all-MiniLM-L6-v2`, sentence-aligned passages of up to 256 characters that never
cross a heading, and the headings embedded with each passage, every answerable question has its
passage in the top 5, the ones the model sees.

**Refusal: threshold 0.55.** 2 of the 28 answerable questions are refused and none of the 19
unanswerable ones gets through. The margin is thin for questions close to the topic: "Which
e-bike brands do you sell?" scores 0.542. That is why the prompt and the checker are there too.

![Share of answerable questions refused and unanswerable questions let through at each threshold,
with the chosen 0.55 marked](docs/images/refusal-threshold.png)

**Checker: threshold 0.70.** 50 of the 52 pairs are judged correctly. Every changed number,
contradiction, mixed-up fact and invented detail is caught. The two misses are true sentences that
need two parts of a passage put together. The method was tuned on the same pairs, so 96% is
optimistic; the per-kind table in the eval README is the useful part.

![Share of unsupported sentences marked supported and supported sentences flagged at each
entailment threshold, with the chosen 0.70 marked](docs/images/verifier-threshold.png)

**Claude end to end.** With a key, Claude Opus 5.5 answered the 26 answerable questions that
retrieval let through, at effort `low` and `medium`:

| effort | sentences | supported | flagged | no citation | output tokens | seconds |
|--------|----------:|----------:|--------:|------------:|--------------:|--------:|
| low    | 52 | 43 | 7 | 2 | 97  | 1.97 |
| medium | 52 | 44 | 8 | 0 | 130 | 1.87 |

grounded uses `medium`, which cited every sentence. I read all 15 flagged sentences against the
handbook, and every one was true: near-copies, small inferences, or two facts joined from two
passages. So on real answers the checker flags about 1 correct sentence in 7, and a flag means
"check this sentence", not "this is wrong".

## What it doesn't promise

- **The thresholds fit this handbook.** Both sets are small and written by one person for one
  made-up shop. Run `eval/refusal.py` and `eval/verifier.py` on questions about your own documents
  before trusting the numbers above.
- **A small checker over-flags.** It misses inferences a person makes easily, so it errs towards
  "check this". It let no unsupported sentence through on the labelled pairs, but that is 52 pairs,
  not a guarantee.
- **One user, one question at a time.** No sign-in, no conversation history, no upload page, and no
  deleting documents from the page.
- **Markdown and PDFs with real text.** No Word or HTML, and no OCR for scanned PDFs.

## How it works

```
  documents --> passages --> embeddings --> PostgreSQL + pgvector
                                                  |
  question -------------------------------> the 5 closest passages
                                                  |
                     best score below 0.55?  yes --> "Your documents don't cover this."
                                                  | no
                                                  v
               Claude answers, citing the passages it used (native citations)
                                                  |
                                                  v
           each sentence checked against the passages it cites (local NLI model)
                                                  |
                                                  v
          the page: the answer as it streams, then a mark on every sentence
```

`POST /api/ask` streams Server-Sent Events: `retrieval` (the passages and their scores), then
either `refused`, or the answer as `text` events followed by one `verdict` per sentence, and finally
`done` with timings and token counts. The page reaches the API through its own server, so the
browser only talks to one origin.

## Configuration

All optional, read from the environment:

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | unset | With a key, Claude answers; without one, the extractive answer |
| `GROUNDED_DOCS_DIR` (compose) | `./demo/docs` | The folder to index |
| `GROUNDED_REFUSAL_THRESHOLD` | `0.55` | Refuse below this cosine similarity |
| `GROUNDED_ENTAILMENT_THRESHOLD` | `0.70` | A sentence is supported at or above this |
| `GROUNDED_TOP_K` | `5` | Passages retrieved per question |
| `GROUNDED_ANSWER_MODEL` | `auto` | `auto`, `extractive`, or a Claude model id |
| `GROUNDED_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it re-indexes everything |
| `GROUNDED_NLI_MODEL` | `cross-encoder/nli-deberta-v3-small` | |
| `GROUNDED_CHUNK_SIZE` | `256` | Characters per passage |
| `GROUNDED_DATABASE_URL` | local PostgreSQL on port 5433 | |

## Development

```sh
docker compose up -d db                          # PostgreSQL with pgvector on port 5433
cd api && pip install -e ".[ml,dev]"
pytest                                           # 83 tests; the database ones need Docker
grounded ingest ../demo/docs && uvicorn grounded.app:app --port 8000
cd ../web && npm ci && npm test && API_URL=http://127.0.0.1:8000 npm run dev
```

- **API** (83): passages and their source positions (Markdown headings, PDF pages, a heading
  carried over a page break), the pgvector store against a real PostgreSQL in Testcontainers, the
  refusal decision, the Claude request and its citations against a fake client, sentence splitting
  and every verdict, the SSE stream, and one end-to-end test that indexes the demo documents with
  the CLI, then answers, refuses and flags through the API. No test calls a model.
- **Web** (17): the SSE parser (events split across chunks, several in one), a streamed answer, a
  refusal, a flagged sentence, the citation popover, and four kinds of failure.

CI runs lint and types, the API tests on Python 3.12, 3.13 and 3.14, the web tests and build, and
[promptlint](https://github.com/mohsinaliwork686-rgb/promptlint) on the system prompt.

| Path | What it holds |
|---|---|
| `api/grounded` | Ingest, embeddings, the pgvector store, retrieval, the Claude call, the checker, the API |
| `api/grounded/prompts/answer.md` | The system prompt |
| `web/` | The Next.js page |
| `eval/` | The labelled questions and pairs, and the scripts that measured everything above |
| `demo/docs/` | The demo handbook (Markdown, plus a PDF built from `demo/pdf-source`) |

The plan, written before the code, is in [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md).
Built from 23 to 27 November 2026.

## Licence

MIT
