import io
from typing import List
import pypdf
from app.domain.interfaces import DocumentParser
from app.domain.models import RawDocumentPage


class PDFParser(DocumentParser):
    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Arquivo vazio: '{filename}' não contém dados.")

        stream = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(stream)
        pages: List[RawDocumentPage] = []

        for idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(
                    RawDocumentPage(
                        page_number=idx + 1,
                        content=text,
                        metadata={"filename": filename},
                    )
                )

        if not pages:
            raise ValueError(f"Não foi possível extrair texto legível de '{filename}'.")

        return pages
