from typing import List
from app.domain.models import RawDocumentPage, DocumentChunk


class ChunkingService:
    """Service responsible for splitting document pages into chunks with overlap and metadata."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            if end >= text_len:
                chunk = text[start:].strip()
                if chunk:
                    chunks.append(chunk)
                break

            # Find best split point according to separators hierarchy
            best_break = -1
            for sep in self.separators:
                if not sep:
                    break
                pos = text.rfind(sep, start, end)
                if pos > start:
                    best_break = pos + len(sep)
                    break

            if best_break == -1:
                best_break = end

            chunk = text[start:best_break].strip()
            if chunk:
                chunks.append(chunk)

            # Move window pointer considering overlap, guaranteeing progress
            start = max(start + 1, best_break - self.chunk_overlap)

        return chunks

    def split_pages(self, pages: List[RawDocumentPage], document_id: str) -> List[DocumentChunk]:
        """Split document pages into sequential chunks preserving document and page metadata."""
        all_chunks: List[DocumentChunk] = []
        chunk_idx = 0

        for page in pages:
            splits = self._split_text(page.content)
            for split in splits:
                all_chunks.append(
                    DocumentChunk(
                        document_id=document_id,
                        chunk_index=chunk_idx,
                        page_number=page.page_number,
                        content=split,
                        token_count=len(split.split()),
                    )
                )
                chunk_idx += 1

        return all_chunks
