from app.adapters.parsers.docx_parser import DocxParser
from app.adapters.parsers.pdf_parser import PDFParser
from app.adapters.parsers.text_parser import TextParser
from app.domain.interfaces import DocumentParser


def get_parser_for_filename(filename: str) -> DocumentParser:
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return PDFParser()
    elif lower.endswith(".docx"):
        return DocxParser()
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return TextParser()
    else:
        raise ValueError(f"Formato de arquivo não suportado: '{filename}'. Use PDF, DOCX, TXT ou MD.")
