from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional
from app.domain.models import DocumentChunk, SearchResult, ChatMessage, RawDocumentPage


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        """Extrai texto e páginas preservando a numeração original de cada página."""
        pass


class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Gera embedding para a consulta do usuário."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings em lote para uma lista de chunks."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensão numérica dos vetores gerados."""
        pass


class VectorStore(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Salva chunks, vetores e índices textuais."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> List[SearchResult]:
        """Executa busca densa e esparsa."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Remove documento e todos os chunks relacionados."""
        pass


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 4,
        threshold: float = 0.30,
    ) -> List[SearchResult]:
        """Reclassifica os candidatos pelo score profundo de Cross-Encoder."""
        pass


class LLMProvider(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage],
    ) -> AsyncIterator[str]:
        """Emite tokens em tempo real de forma assíncrona."""
        pass
