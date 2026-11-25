"""Pick the refusal threshold from the labelled questions.

grounded refuses, without calling the model, when the best passage's cosine similarity to the
question is below a threshold. For every threshold from 0.00 to 1.00 this counts:

- answerable questions wrongly refused (their top score is below the threshold), and
- unanswerable questions wrongly let through (their top score is at or above it).

The rule, written down before the first run (see docs/IMPLEMENTATION_PLAN.md, 4a): a confident
wrong answer is worse than a refusal, so each unanswerable question let through costs 2 and each
answerable one refused costs 1. The chosen threshold is the middle of the widest run of
thresholds with the lowest cost, which leaves the most room on both sides.

    python eval/refusal.py              # the real model; writes eval/results/refusal.json
    python eval/refusal.py --chart      # also draws docs/images/refusal-threshold.png
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from shared import ROOT, build_index, load_questions, write_json

from grounded.embed import DEFAULT_MODEL, get_embedder

STEPS = [round(i / 100, 2) for i in range(101)]

# Categorical slots 1-3 of a palette checked for colour-blind separation; text stays in ink.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6b66", "#e6e5e0"
COST_LET_THROUGH = 2
COST_REFUSED = 1


@dataclass(frozen=True)
class Row:
    threshold: float
    refused: int
    let_through_near: int
    let_through_far: int

    @property
    def let_through(self) -> int:
        return self.let_through_near + self.let_through_far

    @property
    def cost(self) -> int:
        return COST_LET_THROUGH * self.let_through + COST_REFUSED * self.refused


def sweep(
    scores: dict[str, float], answerable: dict[str, bool], kinds: dict[str, str]
) -> list[Row]:
    rows = []
    for t in STEPS:
        rows.append(
            Row(
                threshold=t,
                refused=sum(1 for q, s in scores.items() if answerable[q] and s < t),
                let_through_near=sum(
                    1
                    for q, s in scores.items()
                    if not answerable[q] and kinds[q] == "near" and s >= t
                ),
                let_through_far=sum(
                    1
                    for q, s in scores.items()
                    if not answerable[q] and kinds[q] == "far" and s >= t
                ),
            )
        )
    return rows


def choose(rows: list[Row]) -> tuple[float, list[float]]:
    """The middle of the widest run of consecutive lowest-cost thresholds."""
    best = min(row.cost for row in rows)
    runs: list[list[float]] = []
    for row in rows:
        if row.cost != best:
            continue
        if runs and round(row.threshold - runs[-1][-1], 2) == 0.01:
            runs[-1].append(row.threshold)
        else:
            runs.append([row.threshold])
    widest = max(runs, key=len)
    middle = widest[(len(widest) - 1) // 2]
    return middle, widest


def chart(rows: list[Row], chosen: float, totals: tuple[int, int, int]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    answerable, near, far = totals
    xs = [r.threshold for r in rows]
    figure, axes = plt.subplots(figsize=(8, 4.6), dpi=150, facecolor=SURFACE)
    axes.set_facecolor(SURFACE)
    axes.plot(
        xs,
        [100 * r.refused / answerable for r in rows],
        color=BLUE,
        linewidth=2,
        label=f"answerable questions refused (of {answerable})",
    )
    axes.plot(
        xs,
        [100 * r.let_through_near / near for r in rows],
        color=ORANGE,
        linewidth=2,
        label=f"unanswerable, close to the topic, let through (of {near})",
    )
    axes.plot(
        xs,
        [100 * r.let_through_far / far for r in rows],
        color=AQUA,
        linewidth=2,
        linestyle="--",
        label=f"unanswerable, off topic, let through (of {far})",
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
    axes.set_xlabel("refusal threshold (cosine similarity of the best passage)")
    axes.set_ylabel("% of questions")
    axes.spines[["top", "right"]].set_visible(False)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.tick_params(colors=MUTED)
    axes.xaxis.label.set_color(INK)
    axes.yaxis.label.set_color(INK)
    axes.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figure.tight_layout()
    out = ROOT / "docs" / "images" / "refusal-threshold.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--chart", action="store_true")
    args = parser.parse_args()

    embedder = get_embedder(args.model)
    store = build_index(embedder)
    questions = load_questions()
    scores = {q.id: store.search(embedder.encode([q.question])[0], 1)[0].score for q in questions}
    answerable = {q.id: q.answerable for q in questions}
    kinds = {q.id: q.kind or "" for q in questions}
    rows = sweep(scores, answerable, kinds)
    chosen, run = choose(rows)
    at = next(r for r in rows if r.threshold == chosen)

    totals = (
        sum(answerable.values()),
        sum(1 for q in questions if not q.answerable and q.kind == "near"),
        sum(1 for q in questions if not q.answerable and q.kind == "far"),
    )
    print("question  answerable  kind  top score")
    for q in sorted(questions, key=lambda q: scores[q.id]):
        print(
            f"{q.id:8}  {'yes' if q.answerable else 'no':10}  {q.kind or '':4}  "
            f"{scores[q.id]:.3f}  {q.question}"
        )
    print(
        f"\nchosen threshold {chosen:.2f} (lowest-cost run {run[0]:.2f}-{run[-1]:.2f}): "
        f"{at.refused}/{totals[0]} answerable refused, "
        f"{at.let_through_near}/{totals[1]} near and {at.let_through_far}/{totals[2]} far "
        f"unanswerable let through"
    )

    if args.model == DEFAULT_MODEL:
        write_json(
            "refusal.json",
            {
                "model": args.model,
                "chunk_size": 256,
                "with_headings": True,
                "rule": "cost = 2 x unanswerable let through + 1 x answerable refused; "
                "middle of the widest lowest-cost run",
                "chosen": chosen,
                "lowest_cost_run": [run[0], run[-1]],
                "at_chosen": {
                    "answerable_refused": at.refused,
                    "near_let_through": at.let_through_near,
                    "far_let_through": at.let_through_far,
                },
                "totals": {"answerable": totals[0], "near": totals[1], "far": totals[2]},
                "scores": {q: round(s, 4) for q, s in scores.items()},
                "sweep": [
                    [r.threshold, r.refused, r.let_through_near, r.let_through_far] for r in rows
                ],
            },
        )
    if args.chart:
        chart(rows, chosen, totals)


if __name__ == "__main__":
    main()
