import io
from typing import List, Optional, Any
from app.domain.interfaces import DocumentParser
from app.domain.models import RawDocumentPage


class DoclingParser(DocumentParser):
    def __init__(self, converter: Optional[Any] = None):
        self._converter = converter

    @property
    def converter(self):
        if self._converter is None:
            from docling.document_converter import DocumentConverter
            self._converter = DocumentConverter()
        return self._converter

    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        if not file_bytes or len(file_bytes.strip()) == 0:
            raise ValueError(f"Arquivo vazio: '{filename}' não contém dados.")

        from docling.datamodel.base_models import DocumentStream

        stream = DocumentStream(name=filename, stream=io.BytesIO(file_bytes))
        conv_res = self.converter.convert(stream)
        doc = conv_res.document

        pages: List[RawDocumentPage] = []

        if getattr(doc, "pages", None) and len(doc.pages) > 0:
            for page_no in sorted(doc.pages.keys()):
                page_text = doc.export_to_markdown(page_no=page_no).strip()
                if page_text:
                    pages.append(
                        RawDocumentPage(
                            page_number=int(page_no),
                            content=page_text,
                            metadata={"filename": filename},
                        )
                    )
        else:
            full_text = doc.export_to_markdown().strip()
            if full_text:
                pages.append(
                    RawDocumentPage(
                        page_number=1,
                        content=full_text,
                        metadata={"filename": filename},
                    )
                )

        if not pages:
            raise ValueError(f"Não foi possível extrair conteúdo legível de '{filename}'.")

        return pages
