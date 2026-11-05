"""
Split a parsed datasheet into chunks for retrieval.

Fixed-size token windows are the usual default, and they're wrong for
datasheets. A window boundary landing in the middle of the electrical
characteristics table leaves you with rows of numbers that no longer say which
parameter or unit they belong to. So the rules here are:

  * Text is grouped by section. A chunk never spans two sections, and a long
    section is split at paragraph boundaries, not mid-sentence.
  * A table is its own chunk. If it's too long to fit, it's split by rows and
    the header row is repeated in every piece, so each piece still makes sense
    on its own.
  * Every chunk records where it came from (document, page, section, table
    caption), so answers can cite a page and retrieval can filter by
    component class.
  * Text that Docling places inside a figure is mostly axis labels, which are
    skipped. Figure titles, footnotes and full sentences are kept, because
    datasheets put real notes inside drawings ("SD-130 has solid solder
    pins").
"""

import re
from dataclasses import dataclass, field
from typing import Callable

from docling_core.types.doc import (
    DocItemLabel,
    DoclingDocument,
    TableItem,
    TextItem,
)

# Rough budget, in tokens of the embedding model. bge-small truncates at 512,
# and I leave room for the section/caption prefix added in embed_text().
MAX_TOKENS = 400

# Running headers/footers repeat on every page ("ON Semiconductor - Rev 4")
# and would just add noise to every chunk.
SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}
HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}

# Inside a figure, these labels are always worth keeping.
FIGURE_KEEP_LABELS = {DocItemLabel.CAPTION, DocItemLabel.FOOTNOTE, DocItemLabel.LIST_ITEM}


def approx_tokens(text: str) -> int:
    # Close enough for budgeting. Ingestion passes the real tokenizer in.
    return int(len(text.split()) * 1.3) + 1


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    component_class: str
    kind: str                    # "text" or "table"
    text: str
    pages: list[int] = field(default_factory=list)
    section: str = ""
    caption: str = ""

    def document_label(self) -> str:
        # "transistor-DiodesIncorporated_ZTX415STZ" -> "DiodesIncorporated ZTX415STZ"
        return self.doc_id.split("-", 1)[-1].replace("_", " ")

    def embed_text(self) -> str:
        """
        What actually gets embedded. A table chunk on its own is just numbers,
        and usually doesn't name the part it belongs to either - that's in the
        page title, which is in a different chunk. So the document, section
        and caption go in front. Adding the document name alone was worth
        about +0.11 hit@5 to BM25, which is what prompted giving it to dense
        as well.
        """
        context = " | ".join(p for p in (self.document_label(), self.section, self.caption) if p)
        return f"{context}\n{self.text}"

    def payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "component_class": self.component_class,
            "kind": self.kind,
            "text": self.text,
            "pages": self.pages,
            "section": self.section,
            "caption": self.caption,
        }


def _page_of(item) -> int | None:
    return item.prov[0].page_no if item.prov else None


def table_rows(table: TableItem, doc: DoclingDocument) -> tuple[str, list[str]]:
    """Table as a pipe-separated header line plus one line per row."""
    df = table.export_to_dataframe(doc=doc)

    def clean(cell) -> str:
        return " ".join(str(cell).split())  # cells often contain line breaks

    header = " | ".join(clean(c) for c in df.columns)
    rows = [" | ".join(clean(c) for c in row) for row in df.itertuples(index=False)]
    return header, rows


def split_table(header: str, rows: list[str], count: Callable[[str], int],
                max_tokens: int) -> list[str]:
    """Group rows into pieces under the budget, repeating the header in each."""
    pieces, current = [], []
    for row in rows:
        candidate = "\n".join([header, *current, row])
        if current and count(candidate) > max_tokens:
            pieces.append("\n".join([header, *current]))
            current = []
        current.append(row)
    if current or not pieces:
        pieces.append("\n".join([header, *current]))
    return pieces


def split_long_text(text: str, count: Callable[[str], int], max_tokens: int) -> list[str]:
    """
    Break a paragraph that's over budget into pieces at sentence ends. Most
    paragraphs fit, but application notes inside datasheets sometimes run to
    700+ tokens, and the embedding model silently drops everything past 512.
    """
    if count(text) <= max_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces, current = [], ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if current and count(candidate) > max_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    pieces.append(current)
    return pieces


def worth_keeping_from_figure(item: TextItem) -> bool:
    """
    About 18% of all text in the corpus sits inside figures, and nearly all
    of it is graph furniture: tick values ("100k", "-30") and axis titles
    ("f - Frequency - Hz"). Those almost never end in a full stop. Notes
    and warnings written into a drawing do, so that's the test.
    """
    if item.label in FIGURE_KEEP_LABELS:
        return True
    text = item.text.strip()
    return len(text.split()) >= 3 and text[-1] in ".!?"


def chunk_document(doc: DoclingDocument, doc_id: str, component_class: str,
                   count: Callable[[str], int] = approx_tokens,
                   max_tokens: int = MAX_TOKENS) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""
    buffer: list[str] = []
    buffer_pages: list[int] = []

    # Captions show up twice: attached to their table, and again as ordinary
    # text in the reading order. Remember which ones belong to tables so we
    # don't index them twice.
    table_captions = {
        ref.cref
        for item, _ in doc.iterate_items()
        if isinstance(item, TableItem)
        for ref in item.captions
    }

    def add(kind: str, text: str, pages: list[int], caption: str = "") -> None:
        chunks.append(Chunk(
            chunk_id=f"{doc_id}#{len(chunks)}",
            doc_id=doc_id,
            component_class=component_class,
            kind=kind,
            text=text,
            pages=sorted(set(pages)),
            section=section,
            caption=caption,
        ))

    def flush() -> None:
        if buffer:
            add("text", "\n".join(buffer), buffer_pages)
            buffer.clear()
            buffer_pages.clear()

    # The default walk skips everything inside pictures. Walk them too, but
    # remember what's outside so figure text can be filtered separately.
    outside_figures = {item.self_ref for item, _ in doc.iterate_items()}

    for item, _level in doc.iterate_items(traverse_pictures=True):
        if item.label in SKIP_LABELS or item.self_ref in table_captions:
            continue

        in_figure = item.self_ref not in outside_figures
        if in_figure and isinstance(item, TextItem):
            if not worth_keeping_from_figure(item):
                continue
            # A heading inside a figure is a plot title, not a new section,
            # so it goes in with the text instead of changing `section`.
            item_text = item.text.strip()
            if buffer and count("\n".join([*buffer, item_text])) > max_tokens:
                flush()
            buffer.append(item_text)
            if _page_of(item):
                buffer_pages.append(_page_of(item))
            continue

        page = _page_of(item)

        if isinstance(item, TableItem):
            flush()
            header, rows = table_rows(item, doc)
            caption = item.caption_text(doc)
            for piece in split_table(header, rows, count, max_tokens):
                add("table", piece, [page] if page else [], caption)
            continue

        if not isinstance(item, TextItem) or not item.text.strip():
            continue  # pictures, empty boxes

        if item.label in HEADING_LABELS:
            flush()
            section = item.text.strip()
            continue

        for text in split_long_text(item.text.strip(), count, max_tokens):
            if buffer and count("\n".join([*buffer, text])) > max_tokens:
                flush()
            buffer.append(text)
            if page:
                buffer_pages.append(page)

    flush()
    return chunks
