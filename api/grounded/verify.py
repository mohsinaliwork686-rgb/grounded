"""Checking every sentence of an answer against the passages it cites.

An answer arrives as blocks of text, each with the citations the model attached to it. The
answer is split into sentences; each sentence takes the citations of every block it overlaps.
Then each (cited passage, sentence) pair goes to a local natural-language-inference model,
which says whether the passage entails the sentence. The passage is given with the headings
above it, and is checked whole and one sentence at a time, keeping the best score: a small NLI
model reads "Closed on bank holidays." under "Visiting the shop > Opening hours" correctly, and
the bare line as neutral. A sentence is:

- ``supported`` if at least one passage it cites entails it above the threshold;
- ``not_supported`` if it cites passages and none of them entails it;
- ``no_citation`` if it cites nothing;
- ``connective`` if it cites nothing but is a lead-in like "Here's what the policy says:".

Only the passages a sentence cites are checked. A sentence is never marked supported by a
passage it doesn't cite, even one that would entail it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from grounded.ingest import sentence_spans

# Measured on 25 Nov 2026 by eval/verifier.py on eval/pairs.jsonl (see eval/results/).
DEFAULT_ENTAILMENT_THRESHOLD = 0.7
DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-small"

Verdict = Literal["supported", "not_supported", "no_citation", "connective"]

# A sentence ends at . ! or ? followed by whitespace or the end (so "£12.50" stays whole),
# or at a line break.
_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n|:(?=\s)")
_LIST_MARKER = re.compile(r"(?:[-*+]|\d+[.)])[ \t]+")
_LEAD_INS = ("here", "in short", "in summary", "according to", "based on")
_DIGIT = re.compile(r"\d")


@dataclass(frozen=True)
class Citation:
    """A citation the model attached to a block of its answer."""

    passage_id: int
    cited_text: str


@dataclass(frozen=True)
class Block:
    text: str
    citations: tuple[Citation, ...] = ()


@dataclass(frozen=True)
class Source:
    """A retrieved passage as the verifier sees it: its text and the headings above it."""

    text: str
    context: str = ""


@dataclass(frozen=True)
class Sentence:
    text: str
    start: int
    end: int
    passage_ids: tuple[int, ...]


@dataclass(frozen=True)
class Entailment:
    entailment: float
    neutral: float
    contradiction: float


@dataclass(frozen=True)
class PassageCheck:
    passage_id: int
    scores: Entailment


@dataclass(frozen=True)
class SentenceVerdict:
    index: int
    text: str
    verdict: Verdict
    checks: tuple[PassageCheck, ...]

    @property
    def best_entailment(self) -> float | None:
        return max((c.scores.entailment for c in self.checks), default=None)


class NliModel(Protocol):
    @property
    def name(self) -> str: ...

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]:
        """For each (premise, hypothesis), the probability of each relation."""
        ...


# ----------------------------------------------------------------------------- sentences


def split_sentences(blocks: Sequence[Block]) -> list[Sentence]:
    """Join the blocks, split into sentences, and give each sentence its blocks' citations."""
    text = "".join(block.text for block in blocks)
    bounds: list[tuple[int, int, tuple[int, ...]]] = []
    offset = 0
    for block in blocks:
        block_ids = tuple(dict.fromkeys(c.passage_id for c in block.citations))
        bounds.append((offset, offset + len(block.text), block_ids))
        offset += len(block.text)

    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        # A colon ends a lead-in ("Here's what the booklet says:") and nothing else, so
        # "Frames: 2 years." stays one sentence.
        if match.group(0) == ":" and not _starts_with_lead_in(text[cursor : match.start()]):
            continue
        end = match.start() if match.group(0) == "\n" else match.end()
        spans.append((cursor, end))
        cursor = match.end()
    spans.append((cursor, len(text)))

    sentences: list[Sentence] = []
    for raw_start, raw_end in spans:
        raw = text[raw_start:raw_end]
        if not raw.strip():
            continue
        start = raw_start + len(raw) - len(raw.lstrip())
        marker = _LIST_MARKER.match(text, start, raw_end)
        if marker:
            start = marker.end()
        end = raw_end - (len(raw) - len(raw.rstrip()))
        if start >= end:
            continue
        cited: list[int] = []
        for block_start, block_end, block_ids in bounds:
            if block_start < end and block_end > start:
                cited.extend(i for i in block_ids if i not in cited)
        sentences.append(Sentence(text[start:end], start, end, tuple(cited)))
    return sentences


def _starts_with_lead_in(text: str) -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(lead) for lead in _LEAD_INS)


def is_connective(sentence: str) -> bool:
    """A lead-in that makes no claim of its own. Deliberately narrow; see the plan, 4c."""
    stripped = sentence.strip()
    if stripped.endswith(":"):
        return True
    return (
        len(stripped.split()) <= 8
        and not _DIGIT.search(stripped)
        and _starts_with_lead_in(stripped)
    )


# ----------------------------------------------------------------------------- verdicts


def premises(source: Source) -> list[str]:
    """The passage whole, then each of its sentences, each with the headings in front."""
    prefix = f"{source.context}: " if source.context else ""
    parts = [source.text]
    sentences = [source.text[a:b] for a, b in sentence_spans(source.text, 0, len(source.text))]
    if len(sentences) > 1:
        parts.extend(sentences)
    return [prefix + part for part in parts]


def score_pairs(model: NliModel, items: Sequence[tuple[Source, str]]) -> list[Entailment]:
    """For each (source, sentence), the scores of the premise that entails the sentence most."""
    flat: list[tuple[str, str]] = []
    owners: list[int] = []
    for index, (source, sentence) in enumerate(items):
        for premise in premises(source):
            flat.append((premise, sentence))
            owners.append(index)
    best: dict[int, Entailment] = {}
    for index, scores in zip(owners, model.predict(flat) if flat else [], strict=True):
        if index not in best or scores.entailment > best[index].entailment:
            best[index] = scores
    return [best[index] for index in range(len(items))]


def verify(
    blocks: Sequence[Block],
    sources: dict[int, Source],
    model: NliModel,
    *,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
) -> list[SentenceVerdict]:
    """A verdict for every sentence of the answer.

    ``sources`` maps the id of every retrieved passage to its text and headings. A citation to a
    passage that isn't in it is ignored: the answer can only be checked against what was
    retrieved.
    """
    sentences = split_sentences(blocks)
    items: list[tuple[Source, str]] = []
    owners: list[tuple[int, int]] = []
    for index, sentence in enumerate(sentences):
        for passage_id in sentence.passage_ids:
            if passage_id in sources:
                items.append((sources[passage_id], sentence.text))
                owners.append((index, passage_id))
    scores = score_pairs(model, items)

    checks: dict[int, list[PassageCheck]] = {}
    for (index, passage_id), score in zip(owners, scores, strict=True):
        checks.setdefault(index, []).append(PassageCheck(passage_id, score))

    verdicts = []
    for index, sentence in enumerate(sentences):
        own = tuple(checks.get(index, ()))
        verdict: Verdict
        if any(c.scores.entailment >= threshold for c in own):
            verdict = "supported"
        elif is_connective(sentence.text):
            # A lead-in makes no claim, so there is nothing to support, cited or not.
            verdict = "connective"
        else:
            verdict = "not_supported" if own else "no_citation"
        verdicts.append(SentenceVerdict(index, sentence.text, verdict, own))
    return verdicts


# ----------------------------------------------------------------------------- models


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(x - top) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class CrossEncoderNli:
    """``cross-encoder/nli-deberta-v3-small`` on the CPU. Needs the ``ml`` extra."""

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        self._model_name = model_name
        self._model: Any = None
        self._labels: dict[str, int] = {}

    @property
    def name(self) -> str:
        return self._model_name

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error:
                raise RuntimeError(
                    f'{self._model_name!r} needs sentence-transformers: pip install -e "api[ml]"'
                ) from error
            model = CrossEncoder(self._model_name, device="cpu")
            id2label = model.model.config.id2label
            self._labels = {str(label).lower(): int(i) for i, label in id2label.items()}
            missing = {"entailment", "neutral", "contradiction"} - set(self._labels)
            if missing:
                raise RuntimeError(f"{self._model_name!r} has no {sorted(missing)} labels")
            self._model = model
        return self._model

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]:
        if not pairs:
            return []
        logits = self._load().predict(list(pairs), apply_softmax=False, show_progress_bar=False)
        results = []
        for row in logits:
            probs = _softmax([float(x) for x in row])
            results.append(
                Entailment(
                    entailment=probs[self._labels["entailment"]],
                    neutral=probs[self._labels["neutral"]],
                    contradiction=probs[self._labels["contradiction"]],
                )
            )
        return results


class OverlapNli:
    """A stand-in for tests: entailment is the share of the sentence's words in the premise.

    Not a real inference model. It lets the test suite and CI run the verifier end to end
    without downloading one, and it behaves predictably enough to write tests against.
    """

    _WORD = re.compile(r"[a-z0-9£%]+")

    @property
    def name(self) -> str:
        return "overlap"

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[Entailment]:
        results = []
        for premise, hypothesis in pairs:
            have = set(self._WORD.findall(premise.lower()))
            words = self._WORD.findall(hypothesis.lower())
            share = sum(1 for w in words if w in have) / len(words) if words else 0.0
            results.append(Entailment(share, (1 - share) / 2, (1 - share) / 2))
        return results


def get_nli_model(name: str) -> NliModel:
    return OverlapNli() if name == "overlap" else CrossEncoderNli(name)
