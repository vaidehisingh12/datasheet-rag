"""
Chunker tests on a small hand-built document, so they run in a second and
don't need Docling's models or any real PDFs.
"""

from docling_core.types.doc import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    ProvenanceItem,
    TableCell,
    TableData,
)

from rag.ingest.chunk import approx_tokens, chunk_document, split_table


def prov(page: int) -> ProvenanceItem:
    return ProvenanceItem(page_no=page, bbox=BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 1))


def make_table(rows: list[list[str]]) -> TableData:
    """First row is treated as the column header."""
    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=r, end_row_offset_idx=r + 1,
            start_col_offset_idx=c, end_col_offset_idx=c + 1,
            column_header=(r == 0),
        )
        for r, row in enumerate(rows)
        for c, text in enumerate(row)
    ]
    return TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)


def sample_doc(n_table_rows: int = 3) -> DoclingDocument:
    doc = DoclingDocument(name="BC337")
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="ON Semiconductor", prov=prov(1))
    doc.add_heading(text="Features", prov=prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="NPN general purpose amplifier.", prov=prov(1))
    doc.add_text(label=DocItemLabel.TEXT, text="Pb-free packages are available.", prov=prov(1))
    doc.add_heading(text="Electrical Characteristics", prov=prov(2))
    caption = doc.add_text(label=DocItemLabel.CAPTION, text="Table 1. Maximum ratings", prov=prov(2))
    rows = [["Parameter", "Symbol", "Value", "Unit"]]
    rows += [[f"Param {i}", f"S{i}", str(i * 10), "V"] for i in range(n_table_rows)]
    doc.add_table(data=make_table(rows), caption=caption, prov=prov(2))
    return doc


def test_sections_and_metadata():
    chunks = chunk_document(sample_doc(), "transistor-BC337", "transistor")

    text_chunks = [c for c in chunks if c.kind == "text"]
    assert len(text_chunks) == 1
    assert text_chunks[0].section == "Features"
    assert text_chunks[0].pages == [1]
    # both paragraphs of the section end up together
    assert "amplifier" in text_chunks[0].text and "Pb-free" in text_chunks[0].text


def test_page_headers_are_dropped():
    chunks = chunk_document(sample_doc(), "transistor-BC337", "transistor")
    assert not any("ON Semiconductor" in c.text for c in chunks)


def test_table_is_one_chunk_with_caption_not_duplicated():
    chunks = chunk_document(sample_doc(), "transistor-BC337", "transistor")

    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) == 1
    assert tables[0].caption == "Table 1. Maximum ratings"
    assert tables[0].section == "Electrical Characteristics"
    assert tables[0].text.splitlines()[0] == "Parameter | Symbol | Value | Unit"
    # the caption belongs to the table, so it shouldn't also be a text chunk
    assert not any("Maximum ratings" in c.text for c in chunks if c.kind == "text")


def test_long_table_repeats_header_in_every_piece():
    chunks = chunk_document(sample_doc(n_table_rows=200), "t", "transistor", max_tokens=60)

    tables = [c for c in chunks if c.kind == "table"]
    assert len(tables) > 1
    for t in tables:
        assert t.text.startswith("Parameter | Symbol | Value | Unit")
        assert approx_tokens(t.text) <= 60
    # no rows lost or duplicated when splitting
    body_rows = [line for t in tables for line in t.text.splitlines()[1:]]
    assert len(body_rows) == 200


def test_split_table_keeps_oversized_row():
    # A single row bigger than the budget still has to go somewhere.
    pieces = split_table("h", ["x " * 500], approx_tokens, max_tokens=10)
    assert len(pieces) == 1 and pieces[0].startswith("h\n")


def test_embed_text_puts_context_first():
    table = next(c for c in chunk_document(sample_doc(), "t", "transistor") if c.kind == "table")
    first_line = table.embed_text().splitlines()[0]
    assert first_line == "Electrical Characteristics | Table 1. Maximum ratings"
