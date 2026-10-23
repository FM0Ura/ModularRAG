from typing import List
from app.domain.interfaces import DocumentParser
from app.domain.models import RawDocumentPage


class TextParser(DocumentParser):
    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise ValueError(f"Arquivo vazio: '{filename}' não contém dados.")

        # Tenta decodificar UTF-8 com fallback para latin-1
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")

        return [RawDocumentPage(page_number=1, content=content, metadata={"filename": filename})]
