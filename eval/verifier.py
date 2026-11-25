"""Pick the verifier's entailment threshold from the labelled sentence pairs.

Each of the 52 pairs in ``pairs.jsonl`` is a passage and a sentence labelled supported or
unsupported, with the kind of sentence (a copy, a paraphrase, an inference, a changed number, a
fact mixed up from elsewhere in the handbook, a contradiction, an invented detail). The NLI model
scores each pair; for every threshold from 0.00 to 1.00 this counts:

- unsupported sentences marked supported (entailment at or above the threshold): cost 2, the
  failure the verifier exists to catch;
- supported sentences flagged (entailment below it): cost 1.

The threshold is the middle of the widest run of lowest-cost thresholds, the rule written in
docs/IMPLEMENTATION_PLAN.md (4c) before the first run. Accuracy by kind shows where it fails.

Each pair is scored by the verifier's own code (``score_pairs``): the passage with its headings,
whole and sentence by sentence. The headings come from the demo documents, found by matching the
passage's opening words to the ingested passages.

    python eval/verifier.py            # the real model; writes eval/results/verifier.json
    python eval/verifier.py --chart    # also draws docs/images/verifier-threshold.png
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from shared import DOCS, EVAL, ROOT, squash, write_json

from grounded.ingest import find_documents, read_document
from grounded.verify import DEFAULT_NLI_MODEL, Source, get_nli_model, score_pairs

STEPS = [round(i / 100, 2) for i in range(101)]
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e0"


def choose(costs: dict[float, int]) -> tuple[float, list[float]]:
    best = min(costs.values())
    runs: list[list[float]] = []
    for t in STEPS:
        if costs[t] != best:
            continue
        if runs and round(t - runs[-1][-1], 2) == 0.01:
            runs[-1].append(t)
        else:
            runs.append([t])
    widest = max(runs, key=len)
    return widest[(len(widest) - 1) // 2], widest


def heading_context(passage: str) -> str:
    """The title and headings above the demo passage this text was copied from."""
    key = squash(passage)[:25]
    for path in find_documents(DOCS):
        document = read_document(path, DOCS)
        for candidate in document.passages:
            if key in squash(candidate.text):
                return " > ".join((document.title, *candidate.heading_path))
    raise SystemExit(f"no demo passage starts like: {passage[:40]!r}")


def chart(scores: list[tuple[float, bool]], chosen: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    supported = [s for s, ok in scores if ok]
    unsupported = [s for s, ok in scores if not ok]
    xs = STEPS
    figure, axes = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.plot(
        xs,
        [100 * sum(s < t for s in supported) / len(supported) for t in xs],
        color=BLUE,
        linewidth=2,
        label=f"supported sentences flagged (of {len(supported)})",
    )
    axes.plot(
        xs,
        [100 * sum(s >= t for s in unsupported) / len(unsupported) for t in xs],
        color=ORANGE,
        linewidth=2,
        label=f"unsupported sentences marked supported (of {len(unsupported)})",
    )
    axes.axvline(chosen, color=MUTED, linewidth=1)
    axes.annotate(
        f"threshold {chosen:.2f}",
        (chosen, 103),
        ha="center",
        fontsize=9,
        color=INK,
        annotation_clip=False,
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 100)
    axes.set_xlabel("entailment threshold (probability from the NLI model)", color=INK)
    axes.set_ylabel("% of sentences", color=INK)
    axes.tick_params(colors=MUTED)
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figure.tight_layout()
    out = ROOT / "docs" / "images" / "verifier-threshold.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args()

    rows = [
        json.loads(line) for line in (EVAL / "pairs.jsonl").read_text().splitlines() if line.strip()
    ]
    model = get_nli_model(args.model)
    contexts = {row["passage_id"]: heading_context(row["passage"]) for row in rows}
    predictions = score_pairs(
        model,
        [(Source(row["passage"], contexts[row["passage_id"]]), row["sentence"]) for row in rows],
    )
    entailment = {row["id"]: p.entailment for row, p in zip(rows, predictions, strict=True)}
    supported = {row["id"]: row["label"] == "supported" for row in rows}

    costs = {
        t: sum(2 for i, e in entailment.items() if not supported[i] and e >= t)
        + sum(1 for i, e in entailment.items() if supported[i] and e < t)
        for t in STEPS
    }
    chosen, run = choose(costs)

    by_kind: dict[str, list[bool]] = defaultdict(list)
    wrong = []
    for row in rows:
        correct = (entailment[row["id"]] >= chosen) == supported[row["id"]]
        by_kind[f"{row['label']} / {row['kind']}"].append(correct)
        if not correct:
            wrong.append(row)

    print(f"{'kind':34}  correct")
    for kind in sorted(by_kind):
        results = by_kind[kind]
        print(f"{kind:34}  {sum(results)}/{len(results)}")
    accuracy = sum(sum(r) for r in by_kind.values()) / len(rows)
    print(
        f"\nchosen threshold {chosen:.2f} (lowest-cost run {run[0]:.2f}-{run[-1]:.2f}); "
        f"accuracy {accuracy:.0%} ({len(rows) - len(wrong)}/{len(rows)})"
    )
    for row in wrong:
        print(
            f"  wrong: {row['id']} {row['label']}/{row['kind']} "
            f"entailment {entailment[row['id']]:.2f}: {row['sentence']}"
        )

    if args.model == DEFAULT_NLI_MODEL:
        write_json(
            "verifier.json",
            {
                "model": args.model,
                "rule": "cost = 2 x unsupported marked supported + 1 x supported flagged; "
                "middle of the widest lowest-cost run",
                "chosen": chosen,
                "lowest_cost_run": [run[0], run[-1]],
                "accuracy": round(accuracy, 3),
                "by_kind": {k: [sum(v), len(v)] for k, v in sorted(by_kind.items())},
                "wrong": [r["id"] for r in wrong],
                "entailment": {k: round(v, 4) for k, v in entailment.items()},
            },
        )
    if args.chart:
        chart([(entailment[i], supported[i]) for i in entailment], chosen)


if __name__ == "__main__":
    main()
