"""Document parsers for PDF, DOCX, TXT and MD formats."""

from app.adapters.parsers.docx_parser import DocxParser
from app.adapters.parsers.factory import get_parser_for_filename
from app.adapters.parsers.pdf_parser import PDFParser
from app.adapters.parsers.text_parser import TextParser

__all__ = [
    "DocxParser",
    "PDFParser",
    "TextParser",
    "get_parser_for_filename",
]
