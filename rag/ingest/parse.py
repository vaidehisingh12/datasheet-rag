"""
PDF -> structured document, using Docling.

Can also be run as a worker process (see scripts/ingest.py for why):

    python -m rag.ingest.parse <pdf path> <doc_id> [<pdf path> <doc_id> ...]

Plain text extraction is useless for datasheets: a two-column page turns into
interleaved nonsense and the electrical characteristics table collapses into a
stream of numbers with no idea which column they belonged to. Docling keeps
reading order, section headings and table cells, which is what the chunker
needs.

Parsed output is cached as JSON in PARSED_DIR so the slow step only runs once
per document.
"""

from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from rag.settings import PARSED_DIR


def make_converter() -> DocumentConverter:
    options = PdfPipelineOptions()
    # These datasheets are born-digital, so the text layer is already there.
    # OCR would roughly triple the parse time for no gain.
    options.do_ocr = False
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def parse_pdf(pdf_path: Path, doc_id: str, converter: DocumentConverter) -> DoclingDocument:
    cached = PARSED_DIR / f"{doc_id}.json"
    if cached.exists():
        return DoclingDocument.load_from_json(cached)

    doc = converter.convert(pdf_path).document
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp name first so a crash mid-save can't leave a half-written
    # file that later looks like a valid cache entry.
    tmp = cached.with_suffix(".tmp")
    doc.save_as_json(tmp)
    tmp.replace(cached)
    return doc


def is_parsed(doc_id: str) -> bool:
    return (PARSED_DIR / f"{doc_id}.json").exists()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    converter = make_converter()
    for pdf, doc_id in zip(args[::2], args[1::2]):
        # The parent reads these lines to know which document we were on if
        # this process dies.
        print(f"START {doc_id}", flush=True)
        try:
            parse_pdf(Path(pdf), doc_id, converter)
            print(f"OK {doc_id}", flush=True)
        except Exception as e:
            print(f"ERROR {doc_id} {e!r}", flush=True)
