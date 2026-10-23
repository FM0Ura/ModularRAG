from typing import Optional
from app.adapters.parsers.docx_parser import DocxParser
from app.adapters.parsers.pdf_parser import PDFParser
from app.adapters.parsers.text_parser import TextParser
from app.adapters.parsers.docling_parser import DoclingParser
from app.domain.interfaces import DocumentParser
from app.core.config import Settings, settings as default_settings


def get_parser_for_filename(filename: str, config: Optional[Settings] = None) -> DocumentParser:
    cfg = config or default_settings
    provider = getattr(cfg, "PARSER_PROVIDER", "docling").lower()
    lower = filename.lower()

    if lower.endswith(".pdf") or lower.endswith(".docx"):
        if provider == "docling":
            return DoclingParser()
        elif lower.endswith(".pdf"):
            return PDFParser()
        else:
            return DocxParser()
    elif lower.endswith(".txt") or lower.endswith(".md"):
        return TextParser()
    else:
        raise ValueError(f"Formato de arquivo não suportado: '{filename}'. Use PDF, DOCX, TXT ou MD.")

