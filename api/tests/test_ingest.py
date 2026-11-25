from __future__ import annotations

from pathlib import Path

import pytest

from grounded.ingest import (
    IngestError,
    chunk_markdown,
    chunk_pdf_pages,
    find_documents,
    markdown_sections,
    pack,
    read_document,
    sentence_spans,
)

HANDBOOK = """# Returns

Intro line for the whole handbook.

## Sale items

Sale items can be returned within 14 days. They are refunded as store credit.

## Opening hours

- Monday to Saturday, 9am to 6pm
- Sunday, 10am to 4pm.

### Holidays

Closed on bank holidays.
"""


def _assert_traceable(text: str, passages: list) -> None:
    for passage in passages:
        assert text[passage.start : passage.end] == passage.text
        assert passage.text == passage.text.strip()


def test_markdown_passages_are_exact_slices_with_heading_paths() -> None:
    passages = chunk_markdown(HANDBOOK)
    _assert_traceable(HANDBOOK, passages)
    by_text = {p.text: p.heading_path for p in passages}
    assert by_text["Intro line for the whole handbook."] == ("Returns",)
    sale = next(p for p in passages if p.text.startswith("Sale items"))
    assert sale.heading_path == ("Returns", "Sale items")
    holidays = next(p for p in passages if "bank holidays" in p.text)
    assert holidays.heading_path == ("Returns", "Opening hours", "Holidays")


def test_a_passage_never_crosses_a_heading() -> None:
    passages = chunk_markdown(HANDBOOK, size=1000)
    assert not any("Sunday" in p.text and "bank holidays" in p.text for p in passages)
    assert not any("#" in p.text for p in passages)


def test_list_items_lose_their_markers_but_stay_in_the_slice() -> None:
    passages = chunk_markdown(HANDBOOK)
    hours = next(p for p in passages if p.heading_path[-1] == "Opening hours")
    assert hours.text.startswith("Monday to Saturday")
    assert "Sunday, 10am to 4pm." in hours.text


def test_a_list_item_without_a_full_stop_is_its_own_sentence() -> None:
    text = "- Monday to Saturday, 9am to 6pm\n- Sunday, 10am to 4pm."
    spans = sentence_spans(text, 0, len(text))
    assert [text[s:e] for s, e in spans] == [
        "Monday to Saturday, 9am to 6pm",
        "Sunday, 10am to 4pm.",
    ]


def test_chunks_pack_whole_sentences_up_to_the_size() -> None:
    text = "One two three. Four five six. Seven eight nine."
    spans = pack(text, sentence_spans(text, 0, len(text)), size=30)
    assert [text[s:e] for s, e in spans] == ["One two three. Four five six.", "Seven eight nine."]
    for start, end in spans:
        assert end - start <= 30


def test_a_sentence_longer_than_the_size_is_split_at_words() -> None:
    sentence = " ".join(["word"] * 40) + "."
    spans = pack(sentence, [(0, len(sentence))], size=50)
    assert len(spans) > 1
    for start, end in spans:
        assert end - start <= 50
        assert not sentence[start:end].startswith(" ") and not sentence[start:end].endswith(" ")
    rebuilt = " ".join(sentence[s:e] for s, e in spans)
    assert rebuilt == sentence


def test_pack_rejects_a_non_positive_size() -> None:
    with pytest.raises(ValueError):
        pack("text", [(0, 4)], size=0)


def test_a_heading_inside_a_code_fence_is_not_a_heading() -> None:
    text = "# Title\n\n```\n# not a heading\n```\n\nAfter the fence.\n"
    sections = markdown_sections(text)
    assert [headings for headings, _ in sections] == [("Title",)]


def test_empty_and_whitespace_markdown_give_no_passages(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text("")
    (tmp_path / "blank.md").write_text("\n\n   \n")
    assert read_document(tmp_path / "empty.md", tmp_path).passages == ()
    assert read_document(tmp_path / "blank.md", tmp_path).passages == ()
    assert read_document(tmp_path / "empty.md", tmp_path).title == "empty"


def test_a_markdown_file_that_is_not_utf8_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "latin1.md").write_bytes("caf\xe9".encode("latin-1"))
    with pytest.raises(IngestError, match="UTF-8"):
        read_document(tmp_path / "latin1.md", tmp_path)


def test_other_file_types_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "notes.docx").write_bytes(b"PK")
    with pytest.raises(IngestError, match="only Markdown and PDF"):
        read_document(tmp_path / "notes.docx", tmp_path)


def test_a_broken_pdf_is_an_ingest_error(tmp_path: Path) -> None:
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 this is not a pdf")
    with pytest.raises(IngestError, match="not a readable PDF"):
        read_document(tmp_path / "broken.pdf", tmp_path)


def test_pdf_headings_are_found_from_short_lines() -> None:
    pages = [
        "Warranty booklet\nIntro sentence for the booklet.\nFrames\nFrames are covered for 2\n"
        "years. Carbon frames are covered for 5 years.\nParts\nParts are covered for 90 days.",
        "Claims continue here on the next page.",
    ]
    passages = chunk_pdf_pages(pages)
    for passage in passages:
        assert pages[passage.page - 1][passage.start : passage.end] == passage.text
    frames = next(p for p in passages if "Carbon" in p.text)
    assert frames.heading_path == ("Frames",) and frames.page == 1
    assert next(p for p in passages if "90 days" in p.text).heading_path == ("Parts",)
    # A section running over a page break keeps its heading.
    assert next(p for p in passages if p.page == 2).heading_path == ("Parts",)


def test_the_demo_pdf_keeps_pages_and_sections(demo_docs: Path) -> None:
    document = read_document(demo_docs / "warranty.pdf", demo_docs)
    assert document.kind == "pdf"
    assert document.title == "Quillbrook Cycles warranty booklet"
    for passage in document.passages:
        assert document.source_of(passage)[passage.start : passage.end] == passage.text
    carbon = next(p for p in document.passages if "Carbon frames" in p.text)
    assert carbon.page == 1 and carbon.heading_path == ("Frames",)
    claim = next(p for p in document.passages if "inspect it within 3 working days" in p.text)
    assert claim.page == 2 and claim.heading_path == ("Making a claim",)


def test_every_demo_passage_can_be_traced_to_its_source(demo_docs: Path) -> None:
    paths = find_documents(demo_docs)
    assert [p.name for p in paths] == [
        "membership.md",
        "repairs.md",
        "returns.md",
        "shipping.md",
        "store.md",
        "warranty.pdf",
    ]
    for path in paths:
        document = read_document(path, demo_docs)
        assert document.passages
        for passage in document.passages:
            assert document.source_of(passage)[passage.start : passage.end] == passage.text
            assert len(passage.text) <= 256


def test_hidden_folders_are_skipped(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "notes.md").write_text("# Hidden\n\nText.")
    (tmp_path / "visible.md").write_text("# Shown\n\nText.")
    assert [p.name for p in find_documents(tmp_path)] == ["visible.md"]


def test_a_missing_folder_is_an_ingest_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not a folder"):
        find_documents(tmp_path / "nope")
