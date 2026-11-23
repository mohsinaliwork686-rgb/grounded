# grounded

[![MIT licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Ask your documents. Every sentence of the answer cites the passage it came from, a second model
checks that the passage really supports it, and when your documents don't cover the question, it
says so instead of making something up.

FastAPI, PostgreSQL with pgvector, Claude with native citations, a local NLI model as the verifier,
and a Next.js page that streams the answer with each sentence marked supported or not.

**Being built from 23 to 27 November 2026.** The plan is in
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md), written before any code.
