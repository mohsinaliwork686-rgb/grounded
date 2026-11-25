from __future__ import annotations

from collections.abc import Sequence

import pytest

from grounded.verify import (
    Block,
    Citation,
    Entailment,
    OverlapNli,
    Source,
    is_connective,
    premises,
    score_pairs,
    split_sentences,
    verify,
)

FRAMES = "Frames are covered for 2 years. Carbon frames are covered for 5 years."
PARTS = "Parts and components are covered for 90 days."
PASSAGES = {1: FRAMES, 2: PARTS}
SOURCES = {1: Source(FRAMES), 2: Source(PARTS)}


class TableNli:
    """Returns the entailment written in the test for each (premise, hypothesis)."""

    name = "table"

    def __init__(self, table: dict[tuple[str, str], float]) -> None:
        self.table = table
        self.seen: list[tuple[str, str]] = []

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]:
        self.seen.extend(pairs)
        return [Entailment(self.table.get(p, 0.0), 0.0, 0.0) for p in pairs]


def cite(*ids: int) -> tuple[Citation, ...]:
    return tuple(Citation(i, PASSAGES.get(i, "")) for i in ids)


def test_sentences_take_the_citations_of_every_block_they_overlap() -> None:
    blocks = [
        Block("Frames are covered for 2 years", cite(1)),
        Block(" and parts for 90 days. ", cite(2)),
        Block("Ask in the shop."),
    ]
    sentences = split_sentences(blocks)
    assert [s.text for s in sentences] == [
        "Frames are covered for 2 years and parts for 90 days.",
        "Ask in the shop.",
    ]
    assert sentences[0].passage_ids == (1, 2)
    assert sentences[1].passage_ids == ()


def test_a_block_with_two_sentences_gives_both_its_citations() -> None:
    sentences = split_sentences([Block("One claim. Another claim.", cite(1))])
    assert [s.passage_ids for s in sentences] == [(1,), (1,)]


def test_decimals_and_list_markers_do_not_break_sentences() -> None:
    sentences = split_sentences([Block("A service costs £12.50 today.\n- Frames: 2 years.")])
    assert [s.text for s in sentences] == ["A service costs £12.50 today.", "Frames: 2 years."]


def test_a_lead_in_ending_in_a_colon_is_its_own_sentence_but_a_label_is_not() -> None:
    sentences = split_sentences(
        [Block("Here's what the booklet says: frames get 2 years. Parts: 90 days.")]
    )
    assert [s.text for s in sentences] == [
        "Here's what the booklet says:",
        "frames get 2 years.",
        "Parts: 90 days.",
    ]


def test_supported_not_supported_and_no_citation() -> None:
    blocks = [
        Block("Frames are covered for 2 years. ", cite(1)),
        Block("Frames are covered for 3 years. ", cite(1)),
        Block("Wheels are trued for free."),
    ]
    nli = TableNli({(FRAMES, "Frames are covered for 2 years."): 0.97})
    verdicts = verify(blocks, SOURCES, nli, threshold=0.5)
    assert [v.verdict for v in verdicts] == ["supported", "not_supported", "no_citation"]
    assert verdicts[0].best_entailment == pytest.approx(0.97)
    assert verdicts[2].checks == () and verdicts[2].best_entailment is None


def test_one_entailing_passage_is_enough() -> None:
    blocks = [Block("Parts are covered for 90 days.", cite(1, 2))]
    nli = TableNli({(PARTS, "Parts are covered for 90 days."): 0.9})
    verdict = verify(blocks, SOURCES, nli, threshold=0.5)[0]
    assert verdict.verdict == "supported"
    assert [c.passage_id for c in verdict.checks] == [1, 2]


def test_a_sentence_is_only_checked_against_the_passages_it_cites() -> None:
    # Passage 2 would entail this sentence, but the sentence cites passage 1.
    blocks = [Block("Parts are covered for 90 days.", cite(1))]
    nli = TableNli({(PARTS, "Parts are covered for 90 days."): 0.99})
    verdict = verify(blocks, SOURCES, nli, threshold=0.5)[0]
    assert verdict.verdict == "not_supported"
    assert {premise for premise, _ in nli.seen} == {
        FRAMES,
        "Frames are covered for 2 years.",
        "Carbon frames are covered for 5 years.",
    }


def test_a_citation_to_a_passage_that_was_not_retrieved_is_ignored() -> None:
    blocks = [Block("Frames are covered for 2 years.", cite(99))]
    verdict = verify(blocks, SOURCES, TableNli({}), threshold=0.5)[0]
    assert verdict.verdict == "no_citation" and verdict.checks == ()


def test_the_threshold_is_inclusive() -> None:
    blocks = [Block("Frames are covered for 2 years.", cite(1))]
    nli = TableNli({(FRAMES, "Frames are covered for 2 years."): 0.5})
    assert verify(blocks, SOURCES, nli, threshold=0.5)[0].verdict == "supported"
    assert verify(blocks, SOURCES, nli, threshold=0.51)[0].verdict == "not_supported"


@pytest.mark.parametrize(
    ("sentence", "expected"),
    [
        ("Here's what the warranty says:", True),
        ("Here is what your documents say.", True),
        ("According to the handbook, yes.", True),
        ("Returns are not accepted.", False),
        ("Here you have 30 days.", False),
        ("In short, frames are covered for two years and parts for ninety days.", False),
    ],
)
def test_connective_sentences_are_a_narrow_set(sentence: str, expected: bool) -> None:
    assert is_connective(sentence) is expected


def test_a_cited_lead_in_that_is_not_entailed_is_connective_not_flagged() -> None:
    blocks = [
        Block("Here's what the warranty says: ", cite(1)),
        Block("Frames get 2 years.", cite(1)),
    ]
    nli = TableNli({(FRAMES, "Frames get 2 years."): 0.8})
    verdicts = verify(blocks, SOURCES, nli, threshold=0.5)
    assert [v.verdict for v in verdicts] == ["connective", "supported"]


def test_no_model_call_without_cited_sentences() -> None:
    nli = TableNli({})
    verdicts = verify([Block("Nothing cited here.")], SOURCES, nli)
    assert nli.seen == [] and verdicts[0].verdict == "no_citation"


def test_the_overlap_stand_in_scores_shared_words() -> None:
    nli = OverlapNli()
    copy, changed = nli.predict(
        [(FRAMES, "Frames are covered for 2 years."), (PARTS, "Parts are covered for 30 days.")]
    )
    assert copy.entailment == 1.0
    assert changed.entailment < 1.0


def test_premises_are_the_whole_passage_then_each_sentence_with_headings() -> None:
    source = Source(FRAMES, "Warranty booklet > Frames")
    assert premises(source) == [
        f"Warranty booklet > Frames: {FRAMES}",
        "Warranty booklet > Frames: Frames are covered for 2 years.",
        "Warranty booklet > Frames: Carbon frames are covered for 5 years.",
    ]
    assert premises(Source(PARTS)) == [PARTS]


def test_the_best_premise_decides() -> None:
    sentence = "Carbon frames are covered for 5 years."
    nli = TableNli(
        {(FRAMES, sentence): 0.2, ("Carbon frames are covered for 5 years.", sentence): 0.95}
    )
    [best] = score_pairs(nli, [(Source(FRAMES), sentence)])
    assert best.entailment == 0.95
    verdict = verify([Block(sentence, cite(1))], SOURCES, nli, threshold=0.5)[0]
    assert verdict.verdict == "supported"
