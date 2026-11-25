"""Run the labelled questions through the whole pipeline with Claude, at each effort level.

For every question that retrieval lets through, Claude answers and the verifier checks each
sentence. Reported per effort level:

- answerable questions answered, and their sentences supported / not supported / uncited;
- unanswerable questions that got past retrieval but were declined by Claude
  ("Your documents don't cover this.");
- tokens and time per answer.

This one calls the API, so it costs money (about 45 short requests per effort level) and needs
ANTHROPIC_API_KEY in the environment. Nothing else in the project calls a model.

    python eval/answers.py --effort low medium
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

from shared import build_index, load_questions, write_json

from grounded.embed import DEFAULT_MODEL, get_embedder
from grounded.generate import REFUSAL_SENTENCE, ClaudeAnswerer, Finished
from grounded.retrieve import DEFAULT_THRESHOLD, retrieve
from grounded.verify import DEFAULT_ENTAILMENT_THRESHOLD, Block, CrossEncoderNli, Source, verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--effort", nargs="+", default=["low", "medium"])
    args = parser.parse_args()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first: this script calls the Claude API.")

    embedder = get_embedder(DEFAULT_MODEL)
    store = build_index(embedder)
    nli = CrossEncoderNli()
    questions = load_questions()
    report = {}
    for effort in args.effort:
        answerer = ClaudeAnswerer(effort=effort)
        verdicts: Counter[str] = Counter()
        declined = {"answerable": 0, "unanswerable": 0}
        reached = {"answerable": 0, "unanswerable": 0}
        tokens_in = tokens_out = 0
        seconds = 0.0
        rows = []
        for q in questions:
            result = retrieve(q.question, store, embedder, threshold=DEFAULT_THRESHOLD)
            if not result.confident:
                continue
            group = "answerable" if q.answerable else "unanswerable"
            reached[group] += 1
            passages = [hit.passage for hit in result.hits]
            texts: dict[int, list[str]] = {}
            cites: dict[int, list] = {}
            started = time.perf_counter()
            for event in answerer.stream(q.question, passages):
                if isinstance(event, Finished):
                    tokens_in += event.input_tokens or 0
                    tokens_out += event.output_tokens or 0
                    continue
                texts.setdefault(event.block, []).append(event.text)
                cites.setdefault(event.block, []).extend(event.citations)
            seconds += time.perf_counter() - started
            blocks = [Block("".join(texts[b]), tuple(cites.get(b, ()))) for b in texts]
            answer = "".join(b.text for b in blocks).strip()
            if " ".join(answer.replace("’", "'").split()) == REFUSAL_SENTENCE:
                declined[group] += 1
                rows.append({"id": q.id, "answer": answer, "verdicts": []})
                continue
            checked = verify(
                blocks,
                {
                    p.id: Source(p.text, " > ".join((p.document_title, *p.heading_path)))
                    for p in passages
                },
                nli,
                threshold=DEFAULT_ENTAILMENT_THRESHOLD,
            )
            verdicts.update(v.verdict for v in checked)
            rows.append(
                {"id": q.id, "answer": answer, "verdicts": [[v.text, v.verdict] for v in checked]}
            )
        calls = reached["answerable"] + reached["unanswerable"]
        report[effort] = {
            "reached_model": reached,
            "declined_by_model": declined,
            "sentence_verdicts": dict(verdicts),
            "mean_input_tokens": round(tokens_in / calls) if calls else 0,
            "mean_output_tokens": round(tokens_out / calls) if calls else 0,
            "mean_seconds": round(seconds / calls, 2) if calls else 0,
            "answers": rows,
        }
        print(
            f"effort {effort}: {calls} calls; declined {declined}; verdicts {dict(verdicts)}; "
            f"{report[effort]['mean_output_tokens']} output tokens and "
            f"{report[effort]['mean_seconds']}s per answer"
        )
    write_json("answers.json", report)


if __name__ == "__main__":
    main()
