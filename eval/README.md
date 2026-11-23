# Labelled data for the two thresholds

Written on 23 November 2026, before any retrieval or verifier code ran, so neither set was tuned to
the results.

- `questions.jsonl`: 47 questions about the demo handbook in `demo/docs`. 28 are answerable, each with
  the file and a short piece of text the retrieved passage must contain. 19 are not answerable: 13
  close to the topic (a bike shop) and 6 far from it. They pick the refusal threshold and check
  retrieval.
- `pairs.jsonl`: 52 sentence and passage pairs labelled supported or unsupported, each with the kind
  of sentence: a copy, a paraphrase, an inference, a changed number, a mixed-up fact from elsewhere
  in the handbook, a contradiction, or an invented detail. They pick the verifier's entailment
  threshold and show where it fails.
