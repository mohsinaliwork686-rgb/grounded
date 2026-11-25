"""Does retrieval find the passage that answers each answerable question?

For each of the 28 answerable questions, a hit is a retrieved passage from the labelled file that
contains the labelled evidence text. Reported: hit@1, hit@3, hit@5 and mean reciprocal rank, for
the passage text embedded alone and with its headings in front, at two chunk sizes.

    python eval/retrieval.py                      # the real model (needs the ml extra)
    python eval/retrieval.py --model hashing      # the lexical stub, no download
"""

from __future__ import annotations

import argparse

from shared import build_index, contains_evidence, load_questions, write_json

from grounded.embed import DEFAULT_MODEL, get_embedder


def evaluate(model: str, chunk_size: int, with_headings: bool) -> dict[str, object]:
    embedder = get_embedder(model)
    store = build_index(embedder, chunk_size=chunk_size, with_headings=with_headings)
    questions = [q for q in load_questions() if q.answerable]
    ranks: dict[str, int | None] = {}
    for question in questions:
        assert question.evidence and question.source
        hits = store.search(embedder.encode([question.question])[0], 10)
        ranks[question.id] = next(
            (
                rank
                for rank, hit in enumerate(hits, start=1)
                if hit.passage.document_path == question.source
                and contains_evidence(hit.passage.text, question.evidence)
            ),
            None,
        )

    def hit_at(k: int) -> float:
        return sum(1 for r in ranks.values() if r is not None and r <= k) / len(ranks)

    mrr = sum(1 / r for r in ranks.values() if r is not None) / len(ranks)
    return {
        "model": model,
        "chunk_size": chunk_size,
        "with_headings": with_headings,
        "questions": len(ranks),
        "hit@1": round(hit_at(1), 3),
        "hit@3": round(hit_at(3), 3),
        "hit@5": round(hit_at(5), 3),
        "mrr": round(mrr, 3),
        "missed_at_5": sorted(qid for qid, r in ranks.items() if r is None or r > 5),
        "ranks": ranks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    runs = [
        evaluate(args.model, size, headings) for size in (256, 512) for headings in (False, True)
    ]
    print(
        f"{'chunk':>5}  {'headings':8}  {'hit@1':>5}  {'hit@3':>5}  {'hit@5':>5}  "
        f"{'MRR':>5}  missed@5"
    )
    for run in runs:
        print(
            f"{run['chunk_size']:>5}  {'yes' if run['with_headings'] else 'no':8}  "
            f"{run['hit@1']:>5}  {run['hit@3']:>5}  {run['hit@5']:>5}  {run['mrr']:>5}  "
            f"{', '.join(run['missed_at_5']) or '-'}"  # type: ignore[arg-type]
        )
    if args.model == DEFAULT_MODEL:
        path = write_json("retrieval.json", runs)
        print(f"wrote {path.relative_to(path.parents[2])}")


if __name__ == "__main__":
    main()
