"""Reading documents into passages that can always be traced back to their source.

A passage is an exact slice of its source text: ``source[start:end] == passage.text``. For
Markdown the source is the file; for a PDF it is the text of one page. Nothing is rewritten,
so a citation can be shown in context and checked character for character.

Chunks are packed from whole sentences up to a size limit and never cross a heading. A cut in
the middle of a sentence would hand the verifier half a claim, and a chunk spanning two sections
would mix facts that the demo documents deliberately keep close together.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DEFAULT_CHUNK_SIZE = 256

Kind = Literal["markdown", "pdf"]
Span = tuple[int, int]

MARKDOWN_SUFFIXES = {".md", ".markdown"}
PDF_SUFFIXES = {".pdf"}

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_LIST_MARKER = re.compile(r"[ \t]*(?:[-*+]|\d+[.)])[ \t]+")
# A sentence ends at . ! or ? followed by whitespace. Good enough for prose like the demo's;
# abbreviations like "e.g. " split early, which only makes a chunk boundary less tidy.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s)")


class IngestError(Exception):
    """A file could not be read as a document."""


@dataclass(frozen=True)
class Passage:
    """One chunk of a document, with everything needed to find it again."""

    text: str
    start: int
    end: int
    page: int | None
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    path: str
    kind: Kind
    title: str
    content_hash: str
    # Markdown: one entry, the file's text. PDF: one entry per page.
    sources: tuple[str, ...]
    passages: tuple[Passage, ...] = field(default=())

    def source_of(self, passage: Passage) -> str:
        """The text the passage's offsets point into."""
        return self.sources[0] if passage.page is None else self.sources[passage.page - 1]


# ----------------------------------------------------------------------------- chunking


def sentence_spans(text: str, start: int, end: int) -> list[Span]:
    """Split ``text[start:end]`` into sentences, trimmed of surrounding whitespace.

    Blank lines and list items also end a sentence, so a bullet without a full stop doesn't run
    into the next one.
    """
    spans: list[Span] = []
    cursor = start
    for line_start in _line_breaks(text, start, end):
        if line_start > cursor:
            spans.extend(_split_sentences(text, cursor, line_start))
            cursor = line_start
    spans.extend(_split_sentences(text, cursor, end))
    return [_strip(text, s, e) for s, e in spans if text[s:e].strip()]


def _line_breaks(text: str, start: int, end: int) -> Iterator[int]:
    """Line starts where a new unit must begin: after a blank line, or at a list item."""
    offset = start
    previous_blank = False
    for line in text[start:end].splitlines(keepends=True):
        stripped = line.strip()
        if offset > start and stripped and (previous_blank or _LIST_MARKER.match(line)):
            yield offset
        previous_blank = not stripped
        offset += len(line)


def _split_sentences(text: str, start: int, end: int) -> list[Span]:
    spans: list[Span] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        spans.append((cursor, match.end()))
        cursor = match.end()
    if cursor < end:
        spans.append((cursor, end))
    return spans


def _strip(text: str, start: int, end: int) -> Span:
    """Trim whitespace, and a list marker at the start, so a sentence starts at its first word."""
    piece = text[start:end]
    start += len(piece) - len(piece.lstrip())
    marker = _LIST_MARKER.match(text, start, end)
    if marker:
        start = marker.end()
    piece = text[start:end]
    start += len(piece) - len(piece.lstrip())
    return start, start + len(text[start:end].rstrip())


def _split_long(text: str, span: Span, size: int) -> list[Span]:
    """Break one over-long sentence at word boundaries, so no chunk exceeds ``size``."""
    start, end = span
    pieces: list[Span] = []
    while end - start > size:
        cut = text.rfind(" ", start, start + size + 1)
        if cut <= start:
            cut = start + size
        pieces.append(_strip(text, start, cut))
        start = cut
        while start < end and text[start].isspace():
            start += 1
    if start < end:
        pieces.append((start, end))
    return pieces


def pack(text: str, units: list[Span], size: int) -> list[Span]:
    """Merge consecutive sentences into chunks no longer than ``size`` characters.

    A chunk's length is measured from its first sentence's start to its last one's end, so the
    whitespace between sentences counts, and the chunk is still one exact slice of the source.
    """
    if size <= 0:
        raise ValueError("chunk size must be positive")
    pieces = [piece for unit in units for piece in _split_long(text, unit, size)]
    chunks: list[Span] = []
    for start, end in pieces:
        if chunks and end - chunks[-1][0] <= size:
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return chunks


# ----------------------------------------------------------------------------- Markdown


def markdown_sections(text: str) -> list[tuple[tuple[str, ...], Span]]:
    """The body of each section, with the path of headings above it."""
    sections: list[tuple[tuple[str, ...], Span]] = []
    path: list[tuple[int, str]] = []
    body_start = 0
    offset = 0
    in_fence = False
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        heading = None if in_fence else _HEADING.match(line.rstrip("\r\n"))
        if heading:
            sections.append((tuple(title for _, title in path), (body_start, offset)))
            level = len(heading.group(1))
            path = [(lvl, title) for lvl, title in path if lvl < level]
            path.append((level, heading.group(2).strip()))
            body_start = offset + len(line)
        offset += len(line)
    sections.append((tuple(title for _, title in path), (body_start, len(text))))
    return [(headings, span) for headings, span in sections if text[span[0] : span[1]].strip()]


def markdown_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        heading = _HEADING.match(line)
        if heading and len(heading.group(1)) == 1:
            return heading.group(2).strip()
    return fallback


def chunk_markdown(text: str, size: int = DEFAULT_CHUNK_SIZE) -> list[Passage]:
    passages: list[Passage] = []
    for headings, (start, end) in markdown_sections(text):
        for chunk_start, chunk_end in pack(text, sentence_spans(text, start, end), size):
            passages.append(
                Passage(text[chunk_start:chunk_end], chunk_start, chunk_end, None, headings)
            )
    return passages


# ----------------------------------------------------------------------------- PDF

_HEADING_LINE_MAX = 60


def _is_heading_line(line: str, previous: str | None) -> bool:
    """A short line that doesn't end like a sentence, after a line that did.

    PDFs keep no structure, only lines. A section title in a typeset document is a short line
    starting with a capital, without a full stop; a wrapped paragraph's lines are long, and its
    last line ends with punctuation. The rule is a heuristic, tested on the demo booklet.
    """
    stripped = line.strip()
    if not stripped or len(stripped) > _HEADING_LINE_MAX:
        return False
    if not (stripped[0].isupper() or stripped[0].isdigit()):
        return False
    if stripped[-1] in ".!?:;,":
        return False
    return previous is None or previous.strip() == "" or previous.rstrip()[-1] in ".!?:"


def pdf_sections(text: str) -> list[tuple[tuple[str, ...], Span]]:
    sections: list[tuple[tuple[str, ...], Span]] = []
    heading: tuple[str, ...] = ()
    body_start = 0
    offset = 0
    previous: str | None = None
    for line in text.splitlines(keepends=True):
        if _is_heading_line(line, previous):
            sections.append((heading, (body_start, offset)))
            heading = (line.strip(),)
            body_start = offset + len(line)
        offset += len(line)
        if line.strip():
            previous = line
    sections.append((heading, (body_start, len(text))))
    return [(h, span) for h, span in sections if text[span[0] : span[1]].strip()]


def chunk_pdf_pages(pages: list[str], size: int = DEFAULT_CHUNK_SIZE) -> list[Passage]:
    passages: list[Passage] = []
    carried: tuple[str, ...] = ()
    for number, text in enumerate(pages, start=1):
        for headings, (start, end) in pdf_sections(text):
            # A section that runs over a page break keeps the heading from the page before.
            path = headings or carried
            for chunk_start, chunk_end in pack(text, sentence_spans(text, start, end), size):
                passages.append(
                    Passage(text[chunk_start:chunk_end], chunk_start, chunk_end, number, path)
                )
            carried = path
    return passages


def read_pdf_pages(path: Path) -> tuple[list[str], str | None]:
    """The text of every page, and the title from the PDF's metadata if it has one."""
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as error:  # pragma: no cover - pypdf is a core dependency
        raise IngestError("reading PDFs needs pypdf") from error
    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError) as error:
        raise IngestError(f"{path}: not a readable PDF ({error})") from error
    title = reader.metadata.title if reader.metadata and reader.metadata.title else None
    return pages, title


def _first_line(pages: list[str]) -> str | None:
    for page in pages:
        for line in page.splitlines():
            if line.strip():
                return line.strip()
    return None


# ----------------------------------------------------------------------------- files


def content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_document(path: Path, root: Path, size: int = DEFAULT_CHUNK_SIZE) -> Document:
    """Read one Markdown or PDF file into a document and its passages."""
    relative = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    try:
        data = path.read_bytes()
    except OSError as error:
        raise IngestError(f"{relative}: {error.strerror or error}") from error

    if suffix in MARKDOWN_SUFFIXES:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise IngestError(f"{relative}: not UTF-8 text") from error
        return Document(
            path=relative,
            kind="markdown",
            title=markdown_title(text, fallback=path.stem),
            content_hash=content_hash(data),
            sources=(text,),
            passages=tuple(chunk_markdown(text, size)),
        )
    if suffix in PDF_SUFFIXES:
        pages, metadata_title = read_pdf_pages(path)
        return Document(
            path=relative,
            kind="pdf",
            title=metadata_title or _first_line(pages) or path.stem,
            content_hash=content_hash(data),
            sources=tuple(pages),
            passages=tuple(chunk_pdf_pages(pages, size)),
        )
    raise IngestError(f"{relative}: only Markdown and PDF files are supported")


def find_documents(root: Path) -> list[Path]:
    """Every Markdown and PDF file under ``root``, in a stable order. Hidden folders are skipped."""
    if not root.is_dir():
        raise IngestError(f"{root}: not a folder")
    readable = MARKDOWN_SUFFIXES | PDF_SUFFIXES
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in readable
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    )
