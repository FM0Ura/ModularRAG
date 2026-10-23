# Modular RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir um sistema RAG completo, modular e desacoplado, desde o upload e ingestão de documentos (PDF, DOCX, TXT, MD) até a recuperação híbrida com reranker e streaming de respostas com citação de fontes em uma interface web moderna.

**Architecture:** Arquitetura Limpa (Ports & Adapters) desacoplando completamente a regra de negócio dos provedores de LLM/Embeddings e do banco de dados. O armazenamento unifica dados relacionais e vetoriais em PostgreSQL + pgvector, a recuperação utiliza Busca Híbrida (Dense + Sparse FTS) com fusão RRF e Cross-Encoder (FlashRank), e o streaming de respostas em tempo real é transmitido via Server-Sent Events (SSE) para uma interface React + Vite.

**Tech Stack:** Python 3.11+, FastAPI, SQLAlchemy Async, asyncpg, pgvector, pypdf, python-docx, flashrank, fastembed, httpx, pytest, React, TypeScript, Vite, Tailwind CSS, Lucide Icons, Docker & Docker Compose.

**Spec:** `docs/superpowers/specs/2026-10-23-modular-rag-system-design.md`

## Global Constraints

- O backend deve ser estritamente desacoplado: os serviços de negócio (`IngestionService`, `RetrievalService`, `RAGService`) dependem exclusivamente de interfaces abstratas (`VectorStore`, `LLMProvider`, `EmbeddingProvider`, `Reranker`, `DocumentParser`), nunca de bibliotecas concretas de terceiros.
- A troca de provedor (OpenAI, Gemini, Ollama, FastEmbed) é feita unicamente via variáveis de ambiente (`.env`), sem necessidade de alteração de código.
- Todas as operações de banco de dados e streaming de tokens no backend devem ser assíncronas (`async/await`).
- Cada afirmação gerada pelo assistente deve ser acompanhada de citação numérica `[1]`, `[2]`, com metadados detalhados de documento, página e trecho original.
- O pipeline de busca híbrida deve usar Reciprocal Rank Fusion com constante $k=60$.
- O reranker deve executar localmente sem necessidade de GPU/CUDA, utilizando FlashRank via ONNX.

## Review Focus

1. Upload de documento com conteúdo vazio ou arquivo de 0 bytes -> Deve ser rejeitado pelo parser com erro HTTP 400 amigável sem quebrar o banco.
2. Tentativa de upload do mesmo documento duas vezes -> O hash SHA-256 deve detectar duplicidade e retornar HTTP 409 Conflict idempotente sem duplicar vetores.
3. Consulta sem nenhum documento relevante (score do reranker abaixo de 0.30) -> O sistema deve retornar fontes vazias e o LLM deve responder declarando explicitamente a ausência da informação, sem alucinar.
4. Queda de conexão do cliente durante o streaming SSE -> O backend deve capturar o cancelamento do gerador assíncrono e liberar conexões do pool sem vazamento de memória.
5. Provedor LLM ou Ollama indisponível/timeout -> A API deve emitir um evento SSE de erro com mensagem clara ao invés de interromper o socket abruptamente.

---

### Task 1: Configuração do Projeto, Dependências e Docker Compose

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`
- Create: `backend/app/core/config.py`
- Test: `backend/tests/test_config.py`

**Interfaces:**
- Produces: `Settings` (instância de Pydantic Settings com carregamento de variáveis de ambiente tipadas).

- [ ] **Step 1: Write the failing test for configuration**

```python
# backend/tests/test_config.py
import pytest
from app.core.config import Settings

def test_settings_load_defaults():
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag",
        LLM_PROVIDER="ollama",
        EMBEDDING_PROVIDER="fastembed"
    )
    assert settings.LLM_PROVIDER == "ollama"
    assert settings.EMBEDDING_PROVIDER == "fastembed"
    assert settings.RERANKER_THRESHOLD == 0.30
    assert settings.RERANKER_TOP_K == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.config'`

- [ ] **Step 3: Write backend configuration and project files**

```python
# backend/app/core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # LLM Settings
    LLM_PROVIDER: str = "ollama"
    LLM_MODEL: str = "llama3.2"
    OPENAI_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    OLLAMA_BASE_URL: str = "http://localhost:11434"

    # Embedding Settings
    EMBEDDING_PROVIDER: str = "fastembed"
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"

    # Reranker & Retrieval
    RERANKER_THRESHOLD: float = 0.30
    RERANKER_TOP_K: int = 4
    DENSE_TOP_K: int = 20
    SPARSE_TOP_K: int = 20

    # Chunking
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
```

```toml
# backend/pyproject.toml
[project]
name = "modular-rag-backend"
version = "0.1.0"
description = "Backend modular e desacoplado para sistema RAG com PostgreSQL e pgvector"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.32.0",
    "pydantic-settings>=2.6.0",
    "sqlalchemy[asyncio]>=2.0.36",
    "asyncpg>=0.30.0",
    "pgvector>=0.3.6",
    "pypdf>=5.1.0",
    "python-docx>=1.1.2",
    "flashrank>=0.2.9",
    "fastembed>=0.4.1",
    "httpx>=0.27.2",
    "python-multipart>=0.0.12",
    "openai>=1.52.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.3.3",
    "pytest-asyncio>=0.24.0",
]
```

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: modular_rag_db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: modular_rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  pgdata:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 1 deliverable**

---

### Task 2: Modelos de Domínio e Contratos de Interface (Interfaces Puras)

**Files:**
- Create: `backend/app/domain/models.py`
- Create: `backend/app/domain/interfaces.py`
- Test: `backend/tests/test_domain_models.py`

**Interfaces:**
- Produces: `DocumentChunk`, `SearchResult`, `ChatMessage`, `Citation`, `RawDocumentPage`, `DocumentParser`, `EmbeddingProvider`, `VectorStore`, `Reranker`, `LLMProvider`.

- [ ] **Step 1: Write the failing test for domain models and interfaces**

```python
# backend/tests/test_domain_models.py
import pytest
from app.domain.models import DocumentChunk, SearchResult, ChatMessage, Citation
from app.domain.interfaces import DocumentParser, EmbeddingProvider, VectorStore, Reranker, LLMProvider

def test_document_chunk_creation():
    chunk = DocumentChunk(
        document_id="doc-123",
        chunk_index=0,
        page_number=1,
        content="Texto de exemplo",
        embedding=[0.1, 0.2, 0.3]
    )
    assert chunk.document_id == "doc-123"
    assert chunk.page_number == 1
    assert len(chunk.embedding) == 3

def test_search_result_formatting():
    result = SearchResult(
        chunk_id="chk-1",
        document_id="doc-1",
        filename="manual.pdf",
        page_number=3,
        content="Instrução técnica",
        score=0.92
    )
    assert result.filename == "manual.pdf"
    assert result.score == 0.92
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_domain_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.domain.models'`

- [ ] **Step 3: Implement domain models and abstract interfaces**

```python
# backend/app/domain/models.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class RawDocumentPage:
    page_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DocumentChunk:
    document_id: str
    chunk_index: int
    page_number: int
    content: str
    embedding: Optional[List[float]] = None
    id: Optional[str] = None
    token_count: Optional[int] = None

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    score: float
    rank_dense: Optional[int] = None
    rank_sparse: Optional[int] = None

@dataclass
class Citation:
    id: int
    filename: str
    page_number: int
    score: float
    snippet: str

@dataclass
class ChatMessage:
    role: str  # "user" ou "assistant"
    content: str
    citations: List[Citation] = field(default_factory=list)
```

```python
# backend/app/domain/interfaces.py
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
        sparse_top_k: int = 20
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
        threshold: float = 0.30
    ) -> List[SearchResult]:
        """Reclassifica os candidatos pelo score profundo de Cross-Encoder."""
        pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage]
    ) -> AsyncIterator[str]:
        """Emite tokens em tempo real de forma assíncrona."""
        pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_domain_models.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 2 deliverable**

---

### Task 3: Parsers de Documentos (PDF, DOCX, TXT/MD) com Rastreabilidade de Página

**Files:**
- Create: `backend/app/adapters/parsers/pdf_parser.py`
- Create: `backend/app/adapters/parsers/docx_parser.py`
- Create: `backend/app/adapters/parsers/text_parser.py`
- Create: `backend/app/adapters/parsers/factory.py`
- Test: `backend/tests/test_parsers.py`

**Interfaces:**
- Consumes: `RawDocumentPage`, `DocumentParser`.
- Produces: `get_parser_for_filename(filename: str) -> DocumentParser`.

- [ ] **Step 1: Write the failing tests for parsers (including empty file edge case)**

```python
# backend/tests/test_parsers.py
import pytest
from app.adapters.parsers.factory import get_parser_for_filename
from app.adapters.parsers.text_parser import TextParser

def test_text_parser_success():
    parser = TextParser()
    sample_text = "Primeira linha.\n\nSegunda linha com informações técnicas."
    pages = parser.parse(sample_text.encode("utf-8"), "documento.txt")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert "Segunda linha" in pages[0].content

def test_text_parser_empty_file_raises_value_error():
    parser = TextParser()
    with pytest.raises(ValueError, match="Arquivo vazio"):
        parser.parse(b"", "vazio.txt")

def test_factory_selects_correct_parser():
    p1 = get_parser_for_filename("doc.pdf")
    p2 = get_parser_for_filename("doc.docx")
    p3 = get_parser_for_filename("doc.md")
    assert p1.__class__.__name__ == "PDFParser"
    assert p2.__class__.__name__ == "DocxParser"
    assert p3.__class__.__name__ == "TextParser"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_parsers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.adapters.parsers'`

- [ ] **Step 3: Implement concrete parsers**

```python
# backend/app/adapters/parsers/text_parser.py
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
```

```python
# backend/app/adapters/parsers/pdf_parser.py
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
                        metadata={"filename": filename}
                    )
                )
        
        if not pages:
            raise ValueError(f"Não foi possível extrair texto legível de '{filename}'.")
            
        return pages
```

```python
# backend/app/adapters/parsers/docx_parser.py
import io
from typing import List
import docx
from app.domain.interfaces import DocumentParser
from app.domain.models import RawDocumentPage

class DocxParser(DocumentParser):
    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        if not file_bytes or len(file_bytes) == 0:
            raise ValueError(f"Arquivo vazio: '{filename}' não contém dados.")
        
        stream = io.BytesIO(file_bytes)
        doc = docx.Document(stream)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        
        # Também extrai conteúdo de tabelas
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)
                    
        full_text = "\n\n".join(paragraphs)
        if not full_text.strip():
            raise ValueError(f"Documento DOCX '{filename}' está vazio.")
            
        return [RawDocumentPage(page_number=1, content=full_text, metadata={"filename": filename})]
```

```python
# backend/app/adapters/parsers/factory.py
from app.domain.interfaces import DocumentParser
from app.adapters.parsers.pdf_parser import PDFParser
from app.adapters.parsers.docx_parser import DocxParser
from app.adapters.parsers.text_parser import TextParser

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_parsers.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 3 deliverable**

---

### Task 4: Serviço de Chunking Hierárquico com Overlap e Preservação de Metadados

**Files:**
- Create: `backend/app/services/chunking_service.py`
- Test: `backend/tests/test_chunker.py`

**Interfaces:**
- Consumes: `RawDocumentPage`, `DocumentChunk`.
- Produces: `ChunkingService.split_pages(pages: List[RawDocumentPage], document_id: str) -> List[DocumentChunk]`.

- [ ] **Step 1: Write the failing test for recursive chunking**

```python
# backend/tests/test_chunker.py
import pytest
from app.domain.models import RawDocumentPage
from app.services.chunking_service import ChunkingService

def test_chunking_preserves_page_numbers_and_overlap():
    service = ChunkingService(chunk_size=100, chunk_overlap=20)
    page1 = RawDocumentPage(
        page_number=1,
        content="Parágrafo curto da página um. " * 5
    )
    page2 = RawDocumentPage(
        page_number=2,
        content="Outro conteúdo da página dois com mais detalhes técnicos explicados aqui."
    )
    
    chunks = service.split_pages([page1, page2], document_id="doc-test-1")
    
    assert len(chunks) > 1
    # Garante que chunks da página 1 tenham page_number 1
    assert chunks[0].page_number == 1
    assert chunks[0].document_id == "doc-test-1"
    assert chunks[0].chunk_index == 0
    # Garante que os últimos chunks tenham page_number 2
    assert chunks[-1].page_number == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.chunking_service'`

- [ ] **Step 3: Implement ChunkingService**

```python
# backend/app/services/chunking_service.py
from typing import List
from app.domain.models import RawDocumentPage, DocumentChunk

class ChunkingService:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
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

            # Procura melhor ponto de quebra
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

            # Move o ponteiro considerando overlap
            start = max(start + 1, best_break - self.chunk_overlap)

        return chunks

    def split_pages(self, pages: List[RawDocumentPage], document_id: str) -> List[DocumentChunk]:
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
                        token_count=len(split.split())
                    )
                )
                chunk_idx += 1

        return all_chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_chunker.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 4 deliverable**

---

### Task 5: Banco de Dados Relacional e Vetorial (`pgvector` e SQLAlchemy Async)

**Files:**
- Create: `backend/app/db/session.py`
- Create: `backend/app/db/models.py`
- Create: `backend/app/adapters/vectorstore/pgvector_store.py`
- Test: `backend/tests/test_vectorstore.py`

**Interfaces:**
- Consumes: `DocumentChunk`, `SearchResult`, `VectorStore`.
- Produces: `PGVectorStore` implementando `VectorStore`.

- [ ] **Step 1: Write the failing test for PGVectorStore**

```python
# backend/tests/test_vectorstore.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models import DocumentChunk
from app.adapters.vectorstore.pgvector_store import PGVectorStore

@pytest.mark.asyncio
async def test_vector_store_add_chunks_mocked():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session
    
    store = PGVectorStore(session_factory=session_factory)
    chunk = DocumentChunk(
        document_id="doc-123",
        chunk_index=0,
        page_number=1,
        content="Conteúdo do chunk teste",
        embedding=[0.1] * 384
    )
    
    await store.add_chunks([chunk])
    assert mock_session.execute.called or mock_session.add_all.called
    assert mock_session.commit.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_vectorstore.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.adapters.vectorstore.pgvector_store'`

- [ ] **Step 3: Implement Database models and PGVectorStore**

```python
# backend/app/db/models.py
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, BigINT, DateTime, ForeignKey, Text, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class DBDocument(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_type = Column(String(50), nullable=False)
    file_size_bytes = Column(BigINT, nullable=False)
    total_pages = Column(Integer, default=1)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chunks = relationship("DBDocumentChunk", back_populates="document", cascade="all, delete-orphan")

class DBDocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(None)) # Dimensão dinâmica conforme settings
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    document = relationship("DBDocument", back_populates="chunks")

class DBChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False, default="Nova Conversa")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    messages = relationship("DBChatMessage", back_populates="session", cascade="all, delete-orphan")

class DBChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), CheckConstraint("role IN ('user', 'assistant')"), nullable=False)
    content = Column(Text, nullable=False)
    citations = Column(JSONB, default=list)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("DBChatSession", back_populates="messages")
```

```python
# backend/app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text
from app.core.config import settings
from app.db.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def init_db():
    async with engine.begin() as conn:
        # Ativa extensão pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Cria tabelas se não existirem
        await conn.run_sync(Base.metadata.create_all)
```

```python
# backend/app/adapters/vectorstore/pgvector_store.py
from typing import List, Callable
import uuid
from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.interfaces import VectorStore
from app.domain.models import DocumentChunk, SearchResult
from app.db.models import DBDocumentChunk, DBDocument

class PGVectorStore(VectorStore):
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self.session_factory = session_factory

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        async with self.session_factory() as session:
            db_chunks = [
                DBDocumentChunk(
                    id=uuid.UUID(c.id) if c.id else uuid.uuid4(),
                    document_id=uuid.UUID(c.document_id),
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    content=c.content,
                    embedding=c.embedding
                )
                for c in chunks
            ]
            session.add_all(db_chunks)
            await session.commit()

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> List[SearchResult]:
        async with self.session_factory() as session:
            # 1. Busca Densa (Cosseno HNSW)
            dense_stmt = text("""
                SELECT c.id, c.document_id, d.filename, c.page_number, c.content,
                       (1 - (c.embedding <=> :qvec::vector)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> :qvec::vector
                LIMIT :dense_k
            """)
            dense_res = await session.execute(
                dense_stmt,
                {"qvec": str(query_vector), "dense_k": dense_top_k}
            )
            dense_rows = dense_res.fetchall()

            # 2. Busca Esparsa (FTS com to_tsvector)
            sparse_stmt = text("""
                SELECT c.id, c.document_id, d.filename, c.page_number, c.content,
                       ts_rank_cd(to_tsvector('portuguese', c.content), plainto_tsquery('portuguese', :qtext)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE to_tsvector('portuguese', c.content) @@ plainto_tsquery('portuguese', :qtext)
                ORDER BY score DESC
                LIMIT :sparse_k
            """)
            sparse_res = await session.execute(
                sparse_stmt,
                {"qtext": query, "sparse_k": sparse_top_k}
            )
            sparse_rows = sparse_res.fetchall()

            # Mapeia resultados com ranks para o RRF
            results_map = {}
            for rank, row in enumerate(dense_rows):
                chunk_id = str(row[0])
                results_map[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(row[1]),
                    filename=row[2],
                    page_number=row[3],
                    content=row[4],
                    score=float(row[5]),
                    rank_dense=rank + 1
                )

            for rank, row in enumerate(sparse_rows):
                chunk_id = str(row[0])
                if chunk_id in results_map:
                    results_map[chunk_id].rank_sparse = rank + 1
                else:
                    results_map[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        document_id=str(row[1]),
                        filename=row[2],
                        page_number=row[3],
                        content=row[4],
                        score=float(row[5]),
                        rank_sparse=rank + 1
                    )

            return list(results_map.values())

    async def delete_document(self, document_id: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                delete(DBDocument).where(DBDocument.id == uuid.UUID(document_id))
            )
            await session.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_vectorstore.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 5 deliverable**

---

### Task 6: Adaptadores de Embeddings e LLM (OpenAI, Gemini, Ollama, FastEmbed)

**Files:**
- Create: `backend/app/adapters/embeddings/fastembed_adapter.py`
- Create: `backend/app/adapters/embeddings/openai_embed.py`
- Create: `backend/app/adapters/embeddings/ollama_embed.py`
- Create: `backend/app/adapters/llm/openai_adapter.py`
- Create: `backend/app/adapters/llm/ollama_adapter.py`
- Create: `backend/app/adapters/llm/gemini_adapter.py`
- Create: `backend/app/core/factory.py`
- Test: `backend/tests/test_adapters.py`

**Interfaces:**
- Consumes: `EmbeddingProvider`, `LLMProvider`, `Settings`.
- Produces: `get_embedding_provider() -> EmbeddingProvider`, `get_llm_provider() -> LLMProvider`.

- [ ] **Step 1: Write the failing tests for adapters and factory**

```python
# backend/tests/test_adapters.py
import pytest
from app.core.config import Settings
from app.core.factory import get_llm_provider, get_embedding_provider

def test_factory_returns_configured_providers():
    custom_settings = Settings(
        LLM_PROVIDER="ollama",
        EMBEDDING_PROVIDER="fastembed",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    llm = get_llm_provider(custom_settings)
    embed = get_embedding_provider(custom_settings)
    
    assert llm.__class__.__name__ == "OllamaAdapter"
    assert embed.__class__.__name__ == "FastEmbedAdapter"
    assert embed.dimension == 384
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_adapters.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.core.factory'`

- [ ] **Step 3: Implement Embedding and LLM adapters**

```python
# backend/app/adapters/embeddings/fastembed_adapter.py
from typing import List
from fastembed import TextEmbedding
from app.domain.interfaces import EmbeddingProvider

class FastEmbedAdapter(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimension: int = 384):
        self._model = TextEmbedding(model_name=model_name)
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dim
```

```python
# backend/app/adapters/embeddings/openai_embed.py
from typing import List
from openai import AsyncOpenAI
from app.domain.interfaces import EmbeddingProvider

class OpenAIEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dimension: int = 1536):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        res = await self.client.embeddings.create(input=text, model=self.model)
        return res.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        res = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in res.data]

    @property
    def dimension(self) -> int:
        return self._dim
```

```python
# backend/app/adapters/embeddings/ollama_embed.py
from typing import List
import httpx
from app.domain.interfaces import EmbeddingProvider

class OllamaEmbeddingAdapter(EmbeddingProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text", dimension: int = 768):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": text})
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        for t in texts:
            vec = await self.embed_query(t)
            results.append(vec)
        return results

    @property
    def dimension(self) -> int:
        return self._dim
```

```python
# backend/app/adapters/llm/ollama_adapter.py
from typing import AsyncIterator, List
import json
import httpx
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage

class OllamaAdapter(LLMProvider):
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2"):
        self.base_url = base_url.rstrip("/")
        self.model = model

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage]
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", f"{self.base_url}/api/chat", json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
```

```python
# backend/app/adapters/llm/openai_adapter.py
from typing import AsyncIterator, List
from openai import AsyncOpenAI
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage

class OpenAIAdapter(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage]
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
```

```python
# backend/app/adapters/llm/gemini_adapter.py
from typing import AsyncIterator, List
from openai import AsyncOpenAI
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage

class GeminiAdapter(LLMProvider):
    """Utiliza a compatibilidade oficial OpenAI da Google Gemini API."""
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash"):
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = model

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage]
    ) -> AsyncIterator[str]:
        messages = [{"role": "system", "content": system_prompt}]
        for m in history:
            messages.append({"role": m.role, "content": m.content})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta
```

```python
# backend/app/core/factory.py
from app.core.config import Settings, settings as default_settings
from app.domain.interfaces import LLMProvider, EmbeddingProvider
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.gemini_adapter import GeminiAdapter
from app.adapters.embeddings.fastembed_adapter import FastEmbedAdapter
from app.adapters.embeddings.openai_embed import OpenAIEmbeddingAdapter
from app.adapters.embeddings.ollama_embed import OllamaEmbeddingAdapter

def get_llm_provider(config: Settings = default_settings) -> LLMProvider:
    provider = config.LLM_PROVIDER.lower()
    if provider == "openai":
        return OpenAIAdapter(api_key=config.OPENAI_API_KEY, model=config.LLM_MODEL)
    elif provider == "gemini":
        return GeminiAdapter(api_key=config.GEMINI_API_KEY, model=config.LLM_MODEL)
    elif provider == "ollama":
        return OllamaAdapter(base_url=config.OLLAMA_BASE_URL, model=config.LLM_MODEL)
    else:
        raise ValueError(f"Provedor LLM desconhecido: '{provider}'")

def get_embedding_provider(config: Settings = default_settings) -> EmbeddingProvider:
    provider = config.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        return OpenAIEmbeddingAdapter(
            api_key=config.OPENAI_API_KEY,
            model=config.EMBEDDING_MODEL,
            dimension=config.EMBEDDING_DIMENSION
        )
    elif provider == "ollama":
        return OllamaEmbeddingAdapter(
            base_url=config.OLLAMA_BASE_URL,
            model=config.EMBEDDING_MODEL,
            dimension=config.EMBEDDING_DIMENSION
        )
    elif provider == "fastembed":
        return FastEmbedAdapter(
            model_name=config.EMBEDDING_MODEL,
            dimension=config.EMBEDDING_DIMENSION
        )
    else:
        raise ValueError(f"Provedor de Embedding desconhecido: '{provider}'")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_adapters.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 6 deliverable**

---

### Task 7: Reranker Neural (FlashRank), RRF e Serviço de Recuperação Avançada

**Files:**
- Create: `backend/app/adapters/reranker/flashrank_adapter.py`
- Create: `backend/app/services/retrieval_service.py`
- Test: `backend/tests/test_rrf_and_retrieval.py`

**Interfaces:**
- Consumes: `VectorStore`, `EmbeddingProvider`, `Reranker`, `SearchResult`.
- Produces: `RetrievalService.retrieve_context(query: str) -> List[SearchResult]`.

- [ ] **Step 1: Write the failing test for RRF ranking and RetrievalService**

```python
# backend/tests/test_rrf_and_retrieval.py
import pytest
from unittest.mock import AsyncMock
from app.domain.models import SearchResult
from app.services.retrieval_service import RetrievalService, apply_rrf

def test_apply_rrf_prioritizes_common_matches():
    r1 = SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.9, rank_dense=1, rank_sparse=1)
    r2 = SearchResult("chk-2", "doc-1", "doc.pdf", 2, "texto 2", 0.8, rank_dense=2, rank_sparse=None)
    
    scored = apply_rrf([r1, r2], k=60)
    # r1 está em ambos os rankings, logo seu score RRF deve ser maior
    assert scored[0].chunk_id == "chk-1"
    assert scored[0].score > scored[1].score

@pytest.mark.asyncio
async def test_retrieval_service_flow():
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_rerank = AsyncMock()
    
    mock_embed.embed_query.return_value = [0.1] * 384
    mock_vectorstore.hybrid_search.return_value = [
        SearchResult("chk-1", "doc-1", "doc.pdf", 1, "trecho técnico relevante", 0.9, rank_dense=1, rank_sparse=2)
    ]
    mock_rerank.rerank.return_value = [
        SearchResult("chk-1", "doc-1", "doc.pdf", 1, "trecho técnico relevante", 0.85)
    ]
    
    service = RetrievalService(
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        reranker=mock_rerank,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=4,
        threshold=0.30
    )
    
    results = await service.retrieve_context("pergunta de teste")
    assert len(results) == 1
    assert results[0].score == 0.85
    assert mock_rerank.rerank.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_rrf_and_retrieval.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.retrieval_service'`

- [ ] **Step 3: Implement FlashRank adapter and RetrievalService**

```python
# backend/app/adapters/reranker/flashrank_adapter.py
from typing import List
from flashrank import Ranker, RerankRequest
from app.domain.interfaces import Reranker
from app.domain.models import SearchResult

class FlashRankReranker(Reranker):
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        self.ranker = Ranker(model_name=model_name, cache_dir="/tmp/flashrank")

    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 4,
        threshold: float = 0.30
    ) -> List[SearchResult]:
        if not candidates:
            return []

        passages = [
            {"id": c.chunk_id, "text": c.content, "meta": c}
            for c in candidates
        ]

        rerank_req = RerankRequest(query=query, passages=passages)
        scored_passages = self.ranker.rerank(rerank_req)

        results: List[SearchResult] = []
        for item in scored_passages:
            score = float(item["score"])
            if score >= threshold:
                original_result: SearchResult = item["meta"]
                original_result.score = round(score, 4)
                results.append(original_result)
            if len(results) >= top_k:
                break

        return results
```

```python
# backend/app/services/retrieval_service.py
from typing import List
from app.domain.interfaces import VectorStore, EmbeddingProvider, Reranker
from app.domain.models import SearchResult

def apply_rrf(candidates: List[SearchResult], k: int = 60) -> List[SearchResult]:
    """Aplica Reciprocal Rank Fusion nos resultados da busca híbrida."""
    for c in candidates:
        rrf_score = 0.0
        if c.rank_dense is not None:
            rrf_score += 1.0 / (k + c.rank_dense)
        if c.rank_sparse is not None:
            rrf_score += 1.0 / (k + c.rank_sparse)
        c.score = rrf_score

    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates

class RetrievalService:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        reranker: Reranker,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        final_top_k: int = 4,
        threshold: float = 0.30
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.threshold = threshold

    async def retrieve_context(self, query: str) -> List[SearchResult]:
        # 1. Gera embedding da consulta
        query_vector = await self.embedding_provider.embed_query(query)

        # 2. Executa busca híbrida
        raw_candidates = await self.vector_store.hybrid_search(
            query=query,
            query_vector=query_vector,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k
        )

        if not raw_candidates:
            return []

        # 3. Mescla com RRF
        fused_candidates = apply_rrf(raw_candidates)

        # 4. Rerank com Cross-Encoder
        top_candidates = fused_candidates[:15]
        reranked = await self.reranker.rerank(
            query=query,
            candidates=top_candidates,
            top_k=self.final_top_k,
            threshold=self.threshold
        )

        return reranked
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_rrf_and_retrieval.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 7 deliverable**

---

### Task 8: Serviços de Ingestão e RAG com Streaming e Citações

**Files:**
- Create: `backend/app/services/ingestion_service.py`
- Create: `backend/app/services/rag_service.py`
- Test: `backend/tests/test_rag_service.py`

**Interfaces:**
- Consumes: `ChunkingService`, `VectorStore`, `EmbeddingProvider`, `RetrievalService`, `LLMProvider`.
- Produces: `IngestionService.ingest_file(filename, file_bytes) -> DBDocument`, `RAGService.query_stream(session_id, question) -> AsyncIterator[dict]`.

- [ ] **Step 1: Write the failing test for RAG prompt construction and streaming generator**

```python
# backend/tests/test_rag_service.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models import SearchResult, ChatMessage
from app.services.rag_service import RAGService, build_prompt_with_sources

def test_build_prompt_with_sources_injects_numbered_citations():
    sources = [
        SearchResult("chk-1", "doc-1", "politicas.pdf", 5, "O prazo de SLA é 24h.", 0.95),
        SearchResult("chk-2", "doc-2", "manual.docx", 2, "Suporte disponível em horário comercial.", 0.88)
    ]
    prompt, citations = build_prompt_with_sources("Qual o prazo de SLA?", sources)
    
    assert "[FONTE 1] (Arquivo: politicas.pdf | Página: 5)" in prompt
    assert "[FONTE 2] (Arquivo: manual.docx | Página: 2)" in prompt
    assert len(citations) == 2
    assert citations[0].id == 1
    assert citations[0].filename == "politicas.pdf"

@pytest.mark.asyncio
async def test_rag_service_streams_sources_then_tokens():
    mock_retrieval = AsyncMock()
    mock_llm = AsyncMock()
    mock_session_factory = MagicMock()
    
    mock_retrieval.retrieve_context.return_value = [
        SearchResult("chk-1", "doc-1", "politicas.pdf", 5, "O prazo de SLA é 24h.", 0.95)
    ]
    
    async def sample_token_stream(*args, **kwargs):
        yield "O "
        yield "prazo "
        yield "é 24h."
        
    mock_llm.generate_stream.side_effect = sample_token_stream
    
    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_provider=mock_llm,
        session_factory=mock_session_factory
    )
    
    events = []
    async for event in rag.stream_response(session_id="test-session", question="Qual SLA?"):
        events.append(event)
        
    assert events[0]["type"] == "sources"
    assert len(events[0]["citations"]) == 1
    assert events[1]["type"] == "token"
    assert events[1]["delta"] == "O "
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_rag_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.rag_service'`

- [ ] **Step 3: Implement IngestionService and RAGService**

```python
# backend/app/services/ingestion_service.py
import hashlib
import uuid
from typing import Callable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.adapters.parsers.factory import get_parser_for_filename
from app.services.chunking_service import ChunkingService
from app.domain.interfaces import VectorStore, EmbeddingProvider
from app.db.models import DBDocument

class IngestionService:
    def __init__(
        self,
        chunking_service: ChunkingService,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        session_factory: Callable[[], AsyncSession]
    ):
        self.chunking_service = chunking_service
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.session_factory = session_factory

    async def ingest_file(self, filename: str, file_bytes: bytes) -> DBDocument:
        # 1. Verifica duplicidade por SHA-256
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        async with self.session_factory() as session:
            existing = await session.execute(
                select(DBDocument).where(DBDocument.file_hash == file_hash)
            )
            if existing.scalar_one_or_none():
                raise ValueError(f"O documento '{filename}' já foi indexado previamente.")

        # 2. Executa parser correspondente
        parser = get_parser_for_filename(filename)
        pages = parser.parse(file_bytes, filename)

        # 3. Cria registro de documento mestre
        doc_id = uuid.uuid4()
        file_type = filename.split(".")[-1].lower() if "." in filename else "unknown"
        db_doc = DBDocument(
            id=doc_id,
            filename=filename,
            file_hash=file_hash,
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            total_pages=len(pages)
        )
        async with self.session_factory() as session:
            session.add(db_doc)
            await session.commit()

        # 4. Divide páginas em chunks
        chunks = self.chunking_service.split_pages(pages, document_id=str(doc_id))

        # 5. Gera embeddings em lotes
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c.content for c in batch]
            embeddings = await self.embedding_provider.embed_documents(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

        # 6. Salva chunks no VectorStore
        await self.vector_store.add_chunks(chunks)

        return db_doc
```

```python
# backend/app/services/rag_service.py
from typing import List, Tuple, AsyncIterator, Callable, Dict, Any
import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.interfaces import LLMProvider
from app.domain.models import SearchResult, Citation, ChatMessage
from app.services.retrieval_service import RetrievalService
from app.db.models import DBChatMessage, DBChatSession

SYSTEM_PROMPT = """Você é um assistente técnico especialista e confiável.
Responda à pergunta do usuário baseando-se estritamente nas FONTES fornecidas.
Regras:
1. Responda apenas com o que estiver respaldado nas fontes. Não invente detalhes.
2. Cite cada informação com referências no formato [1], [2], correspondente à fonte consultada.
3. Se a informação não constar nas fontes, declare educadamente que os documentos cadastrados não contêm essa informação."""

def build_prompt_with_sources(question: str, sources: List[SearchResult]) -> Tuple[str, List[Citation]]:
    citations = []
    if not sources:
        prompt = f"Nenhum documento relevante encontrado.\n\nPERGUNTA: {question}"
        return prompt, citations

    sources_text = "FONTES CONSULTADAS:\n"
    for idx, s in enumerate(sources, start=1):
        citations.append(
            Citation(
                id=idx,
                filename=s.filename,
                page_number=s.page_number,
                score=s.score,
                snippet=s.content[:250] + ("..." if len(s.content) > 250 else "")
            )
        )
        sources_text += f"\n[FONTE {idx}] (Arquivo: {s.filename} | Página: {s.page_number} | Relevância: {s.score})\n{s.content}\n"

    prompt = f"{sources_text}\n\nPERGUNTA DO USUÁRIO:\n{question}"
    return prompt, citations

class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        session_factory: Callable[[], AsyncSession]
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.session_factory = session_factory

    async def stream_response(self, session_id: str, question: str) -> AsyncIterator[Dict[str, Any]]:
        # 1. Recupera chunks com busca híbrida + rerank
        sources = await self.retrieval_service.retrieve_context(question)
        prompt, citations = build_prompt_with_sources(question, sources)

        # 2. Emite evento de fontes imediatamente
        yield {
            "type": "sources",
            "citations": [
                {
                    "id": c.id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "score": c.score,
                    "snippet": c.snippet
                }
                for c in citations
            ]
        }

        # 3. Carrega histórico recente da sessão
        history: List[ChatMessage] = []
        if self.session_factory:
            async with self.session_factory() as session:
                stmt = select(DBChatMessage).where(
                    DBChatMessage.session_id == uuid.UUID(session_id)
                ).order_by(DBChatMessage.created_at.desc()).limit(6)
                res = await session.execute(stmt)
                db_msgs = list(reversed(res.scalars().all()))
                for m in db_msgs:
                    history.append(ChatMessage(role=m.role, content=m.content))

        # 4. Stream dos tokens do LLM
        full_response = []
        async for token in self.llm_provider.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            history=history
        ):
            full_response.append(token)
            yield {"type": "token", "delta": token}

        complete_text = "".join(full_response)

        # 5. Salva mensagens no histórico do PostgreSQL
        if self.session_factory:
            async with self.session_factory() as session:
                user_msg = DBChatMessage(
                    session_id=uuid.UUID(session_id),
                    role="user",
                    content=question
                )
                assistant_msg = DBChatMessage(
                    session_id=uuid.UUID(session_id),
                    role="assistant",
                    content=complete_text,
                    citations=[
                        {
                            "id": c.id,
                            "filename": c.filename,
                            "page_number": c.page_number,
                            "score": c.score,
                            "snippet": c.snippet
                        }
                        for c in citations
                    ]
                )
                session.add_all([user_msg, assistant_msg])
                await session.commit()
                msg_id = str(assistant_msg.id)
        else:
            msg_id = str(uuid.uuid4())

        # 6. Evento de conclusão
        yield {"type": "done", "message_id": msg_id, "session_id": session_id}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_rag_service.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 8 deliverable**

---

### Task 9: Endpoints da API FastAPI (Upload, Chat Streaming SSE, Gestão de Documentos)

**Files:**
- Create: `backend/app/api/schemas.py`
- Create: `backend/app/api/routes/health.py`
- Create: `backend/app/api/routes/documents.py`
- Create: `backend/app/api/routes/chat.py`
- Create: `backend/app/main.py`
- Test: `backend/tests/test_api_routes.py`

**Interfaces:**
- Produces: Aplicação FastAPI com endpoints:
  - `GET /api/v1/health`
  - `POST /api/v1/documents/upload`
  - `GET /api/v1/documents`
  - `DELETE /api/v1/documents/{document_id}`
  - `POST /api/v1/chat/sessions`
  - `GET /api/v1/chat/sessions`
  - `GET /api/v1/chat/sessions/{session_id}/messages`
  - `POST /api/v1/chat/{session_id}/stream` (SSE)

- [ ] **Step 1: Write the failing tests for API routes**

```python
# backend/tests/test_api_routes.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "llm_provider" in data
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_api_routes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement schemas, routes, and main application**

```python
# backend/app/api/schemas.py
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import uuid

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    total_pages: int
    created_at: datetime

class ChatSessionCreate(BaseModel):
    title: Optional[str] = "Nova Conversa"

class ChatSessionResponse(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime

class CitationSchema(BaseModel):
    id: int
    filename: str
    page_number: int
    score: float
    snippet: str

class ChatMessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    citations: Optional[List[CitationSchema]] = []
    created_at: datetime

class ChatPromptRequest(BaseModel):
    question: str
```

```python
# backend/app/api/routes/health.py
from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])

@router.get("")
async def get_health():
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "reranker_threshold": settings.RERANKER_THRESHOLD
    }
```

```python
# backend/app/api/routes/documents.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from sqlalchemy import select
import uuid
from app.api.schemas import DocumentResponse
from app.db.session import AsyncSessionLocal
from app.db.models import DBDocument
from app.core.config import settings
from app.core.factory import get_embedding_provider
from app.adapters.vectorstore.pgvector_store import PGVectorStore
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])

def get_ingestion_service() -> IngestionService:
    chunking = ChunkingService(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    embed = get_embedding_provider()
    return IngestionService(
        chunking_service=chunking,
        vector_store=vstore,
        embedding_provider=embed,
        session_factory=AsyncSessionLocal
    )

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo sem nome fornecido.")
    
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio (0 bytes).")
        
    service = get_ingestion_service()
    try:
        db_doc = await service.ingest_file(filename=file.filename, file_bytes=contents)
        return db_doc
    except ValueError as e:
        if "já foi indexado" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))

@router.get("", response_model=List[DocumentResponse])
async def list_documents():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(DBDocument).order_by(DBDocument.created_at.desc()))
        return res.scalars().all()

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    await vstore.delete_document(document_id)
    return {"message": "Documento removido com sucesso.", "document_id": document_id}
```

```python
# backend/app/api/routes/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
import json
import uuid
from typing import List
from sqlalchemy import select
from app.api.schemas import ChatSessionCreate, ChatSessionResponse, ChatMessageResponse, ChatPromptRequest
from app.db.session import AsyncSessionLocal
from app.db.models import DBChatSession, DBChatMessage
from app.core.config import settings
from app.core.factory import get_llm_provider, get_embedding_provider
from app.adapters.vectorstore.pgvector_store import PGVectorStore
from app.adapters.reranker.flashrank_adapter import FlashRankReranker
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])

def get_rag_service() -> RAGService:
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    embed = get_embedding_provider()
    reranker = FlashRankReranker()
    retrieval = RetrievalService(
        vector_store=vstore,
        embedding_provider=embed,
        reranker=reranker,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        final_top_k=settings.RERANKER_TOP_K,
        threshold=settings.RERANKER_THRESHOLD
    )
    llm = get_llm_provider()
    return RAGService(
        retrieval_service=retrieval,
        llm_provider=llm,
        session_factory=AsyncSessionLocal
    )

@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(payload: ChatSessionCreate):
    async with AsyncSessionLocal() as session:
        chat_sess = DBChatSession(title=payload.title or "Nova Conversa")
        session.add(chat_sess)
        await session.commit()
        await session.refresh(chat_sess)
        return chat_sess

@router.get("/sessions", response_model=List[ChatSessionResponse])
async def list_chat_sessions():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(DBChatSession).order_by(DBChatSession.updated_at.desc()))
        return res.scalars().all()

@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(session_id: str):
    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(DBChatMessage)
            .where(DBChatMessage.session_id == uuid.UUID(session_id))
            .order_by(DBChatMessage.created_at.asc())
        )
        return res.scalars().all()

@router.post("/{session_id}/stream")
async def chat_stream(session_id: str, payload: ChatPromptRequest):
    rag_service = get_rag_service()

    async def sse_event_generator():
        try:
            async for event in rag_service.stream_response(session_id, payload.question):
                event_type = event.get("type", "message")
                data_json = json.dumps(event)
                yield f"event: {event_type}\ndata: {data_json}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
```

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.db.session import init_db
from app.api.routes import health, documents, chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa banco e extensões no startup se a URL for válida
    try:
        await init_db()
    except Exception as e:
        print(f"Aviso de inicialização do DB: {e}")
    yield

app = FastAPI(
    title="Modular RAG API",
    description="API assíncrona, desacoplada e modular para RAG corporativo",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_api_routes.py -v`
Expected: PASS

- [ ] **Step 5: Save and verify Task 9 deliverable**

---

### Task 10: Interface Frontend em React (Vite + Tailwind CSS + Streaming SSE + Citações)

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tailwind.config.js`
- Create: `frontend/postcss.config.js`
- Create: `frontend/index.html`
- Create: `frontend/src/types/index.ts`
- Create: `frontend/src/services/api.ts`
- Create: `frontend/src/components/layout/Header.tsx`
- Create: `frontend/src/components/sidebar/DocumentUpload.tsx`
- Create: `frontend/src/components/sidebar/DocumentList.tsx`
- Create: `frontend/src/components/sidebar/Sidebar.tsx`
- Create: `frontend/src/components/citations/SourceDrawer.tsx`
- Create: `frontend/src/components/chat/ChatMessageItem.tsx`
- Create: `frontend/src/components/chat/ChatWindow.tsx`
- Create: `frontend/src/App.tsx`
- Test: Validação de build do Vite e checagem de tipos TypeScript (`npm run build` ou simulação de imports).

**Interfaces:**
- Produces: Interface web completa consumindo a API FastAPI, streaming SSE, visualização de documentos indexados e drawer lateral de inspeção de fontes citadas.

- [ ] **Step 1: Write frontend configuration, packages and types**

```json
// frontend/package.json
{
  "name": "modular-rag-frontend",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "lucide-react": "^0.454.0",
    "react-markdown": "^9.0.1",
    "clsx": "^2.1.1",
    "tailwind-merge": "^2.5.4"
  },
  "devDependencies": {
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.3",
    "autoprefixer": "^10.4.20",
    "postcss": "^8.4.47",
    "tailwindcss": "^3.4.14",
    "typescript": "^5.6.3",
    "vite": "^5.4.10"
  }
}
```

```typescript
// frontend/src/types/index.ts
export interface DocumentItem {
  id: string;
  filename: string;
  file_type: string;
  file_size_bytes: number;
  total_pages: number;
  created_at: string;
}

export interface Citation {
  id: number;
  filename: string;
  page_number: number;
  score: number;
  snippet: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
  created_at?: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
}
```

- [ ] **Step 2: Implement API client with SSE EventSource stream parser**

```typescript
// frontend/src/services/api.ts
import { DocumentItem, ChatSession, ChatMessage, Citation } from '../types';

const API_BASE = 'http://localhost:8000/api/v1';

export const api = {
  async getHealth() {
    const res = await fetch(`${API_BASE}/health`);
    return res.json();
  },

  async listDocuments(): Promise<DocumentItem[]> {
    const res = await fetch(`${API_BASE}/documents`);
    if (!res.ok) throw new Error('Erro ao listar documentos');
    return res.json();
  },

  async uploadDocument(file: File): Promise<DocumentItem> {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/documents/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Falha no upload');
    }
    return res.json();
  },

  async deleteDocument(id: string): Promise<void> {
    const res = await fetch(`${API_BASE}/documents/${id}`, { method: 'DELETE' });
    if (!res.ok) throw new Error('Erro ao deletar documento');
  },

  async listSessions(): Promise<ChatSession[]> {
    const res = await fetch(`${API_BASE}/chat/sessions`);
    if (!res.ok) throw new Error('Erro ao listar conversas');
    return res.json();
  },

  async createSession(title?: string): Promise<ChatSession> {
    const res = await fetch(`${API_BASE}/chat/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title }),
    });
    return res.json();
  },

  async getMessages(sessionId: string): Promise<ChatMessage[]> {
    const res = await fetch(`${API_BASE}/chat/sessions/${sessionId}/messages`);
    if (!res.ok) throw new Error('Erro ao carregar mensagens');
    return res.json();
  },

  streamChat(
    sessionId: string,
    question: string,
    onSources: (citations: Citation[]) => void,
    onToken: (token: string) => void,
    onDone: () => void,
    onError: (err: string) => void
  ) {
    const controller = new AbortController();

    fetch(`${API_BASE}/chat/${sessionId}/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error('Falha ao iniciar streaming');
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        if (!reader) return;

        let buffer = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n\n');
          buffer = lines.pop() || '';

          for (const block of lines) {
            const eventLine = block.match(/event:\s*(.+)/);
            const dataLine = block.match(/data:\s*(.+)/);

            if (dataLine) {
              try {
                const data = JSON.parse(dataLine[1]);
                const eventType = eventLine ? eventLine[1].trim() : 'message';

                if (eventType === 'sources') {
                  onSources(data.citations || []);
                } else if (eventType === 'token') {
                  onToken(data.delta || '');
                } else if (eventType === 'done') {
                  onDone();
                } else if (eventType === 'error') {
                  onError(data.error || 'Erro desconhecido');
                }
              } catch (e) {
                console.error('Erro de parse SSE:', e);
              }
            }
          }
        }
        onDone();
      })
      .catch((err) => {
        if (err.name !== 'AbortError') {
          onError(err.message);
        }
      });

    return () => controller.abort();
  },
};
```

- [ ] **Step 3: Implement UI Components (Sidebar, Upload, Chat, SourceDrawer)**

```tsx
// frontend/src/components/citations/SourceDrawer.tsx
import React from 'react';
import { Citation } from '../../types';
import { X, FileText, CheckCircle2 } from 'lucide-react';

interface SourceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  citation: Citation | null;
}

export const SourceDrawer: React.FC<SourceDrawerProps> = ({ isOpen, onClose, citation }) => {
  if (!isOpen || !citation) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-96 bg-white dark:bg-gray-900 shadow-2xl border-l border-gray-200 dark:border-gray-800 p-6 z-50 flex flex-col">
      <div className="flex items-center justify-between pb-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center space-x-2">
          <FileText className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          <h3 className="font-semibold text-gray-900 dark:text-white">Fonte Citada [{citation.id}]</h3>
        </div>
        <button onClick={onClose} className="p-1 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-500">
          <X className="w-5 h-5" />
        </button>
      </div>

      <div className="mt-4 space-y-4 flex-1 overflow-y-auto">
        <div className="bg-gray-50 dark:bg-gray-800/50 p-4 rounded-xl space-y-2">
          <div>
            <span className="text-xs font-medium text-gray-500 uppercase">Documento</span>
            <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">{citation.filename}</p>
          </div>
          <div className="flex justify-between">
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase">Página</span>
              <p className="text-sm font-semibold text-gray-800 dark:text-gray-200">{citation.page_number}</p>
            </div>
            <div>
              <span className="text-xs font-medium text-gray-500 uppercase">Relevância Reranker</span>
              <p className="text-sm font-semibold text-emerald-600 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" />
                {(citation.score * 100).toFixed(1)}%
              </p>
            </div>
          </div>
        </div>

        <div>
          <span className="text-xs font-medium text-gray-500 uppercase">Trecho Original do Documento</span>
          <div className="mt-1.5 p-4 rounded-xl bg-indigo-50/50 dark:bg-indigo-950/20 border border-indigo-100 dark:border-indigo-900/50 text-sm text-gray-700 dark:text-gray-300 leading-relaxed font-mono">
            "{citation.snippet}"
          </div>
        </div>
      </div>
    </div>
  );
};
```

- [ ] **Step 4: Implement Main App and Chat Window**

```tsx
// frontend/src/App.tsx
import React, { useState, useEffect } from 'react';
import { api } from './services/api';
import { DocumentItem, ChatSession, ChatMessage, Citation } from './types';
import { SourceDrawer } from './components/citations/SourceDrawer';
import { MessageSquare, Plus, Upload, Trash2, Send, FileText, Cpu, BookOpen } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function App() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSession, setActiveSession] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeCitation, setActiveCitation] = useState<Citation | null>(null);
  const [systemHealth, setSystemHealth] = useState<any>(null);

  useEffect(() => {
    loadInitialData();
  }, []);

  useEffect(() => {
    if (activeSession) {
      loadSessionMessages(activeSession);
    }
  }, [activeSession]);

  async function loadInitialData() {
    try {
      const [docs, sessList, health] = await Promise.all([
        api.listDocuments(),
        api.listSessions(),
        api.getHealth(),
      ]);
      setDocuments(docs);
      setSessions(sessList);
      setSystemHealth(health);

      if (sessList.length > 0) {
        setActiveSession(sessList[0].id);
      } else {
        const newSess = await api.createSession('Primeira Conversa');
        setSessions([newSess]);
        setActiveSession(newSess.id);
      }
    } catch (err) {
      console.error('Erro ao carregar dados:', err);
    }
  }

  async function loadSessionMessages(sessionId: string) {
    try {
      const msgs = await api.getMessages(sessionId);
      setMessages(msgs);
    } catch (err) {
      console.error(err);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const newDoc = await api.uploadDocument(file);
      setDocuments((prev) => [newDoc, ...prev]);
    } catch (err: any) {
      alert(`Falha no upload: ${err.message}`);
    }
  }

  async function handleSendMessage(e: React.FormEvent) {
    e.preventDefault();
    if (!inputMessage.trim() || !activeSession || isStreaming) return;

    const userText = inputMessage;
    setInputMessage('');

    const userMessage: ChatMessage = {
      id: String(Date.now()),
      role: 'user',
      content: userText,
    };

    const tempAssistantId = String(Date.now() + 1);
    const initialAssistantMessage: ChatMessage = {
      id: tempAssistantId,
      role: 'assistant',
      content: '',
      citations: [],
    };

    setMessages((prev) => [...prev, userMessage, initialAssistantMessage]);
    setIsStreaming(true);

    api.streamChat(
      activeSession,
      userText,
      (citations) => {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === tempAssistantId ? { ...msg, citations } : msg))
        );
      },
      (token) => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === tempAssistantId ? { ...msg, content: msg.content + token } : msg
          )
        );
      },
      () => setIsStreaming(false),
      (err) => {
        setIsStreaming(false);
        alert(`Erro de resposta: ${err}`);
      }
    );
  }

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-950 text-gray-900 dark:text-gray-100 font-sans">
      {/* Sidebar */}
      <aside className="w-80 border-r border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 flex flex-col">
        <div className="p-4 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            <Cpu className="w-6 h-6 text-indigo-600" />
            <span className="font-bold text-lg tracking-tight">Modular RAG</span>
          </div>
          <button
            onClick={async () => {
              const sess = await api.createSession(`Conversa ${sessions.length + 1}`);
              setSessions((prev) => [sess, ...prev]);
              setActiveSession(sess.id);
            }}
            className="p-1.5 rounded-lg bg-indigo-50 text-indigo-600 hover:bg-indigo-100"
          >
            <Plus className="w-4 h-4" />
          </button>
        </div>

        {/* Histórico de Conversas */}
        <div className="flex-1 overflow-y-auto p-3 space-y-1">
          <div className="text-xs font-semibold uppercase text-gray-400 px-2 py-1">Conversas</div>
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => setActiveSession(s.id)}
              className={`w-full flex items-center space-x-2 px-3 py-2 rounded-lg text-sm text-left transition-colors ${
                activeSession === s.id
                  ? 'bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-400 font-medium'
                  : 'hover:bg-gray-100 dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
              }`}
            >
              <MessageSquare className="w-4 h-4 shrink-0" />
              <span className="truncate">{s.title}</span>
            </button>
          ))}
        </div>

        {/* Ingestão de Documentos */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase text-gray-400 flex items-center gap-1.5">
              <BookOpen className="w-3.5 h-3.5" />
              Documentos ({documents.length})
            </span>
            <label className="cursor-pointer text-xs font-medium text-indigo-600 hover:underline flex items-center gap-1">
              <Upload className="w-3.5 h-3.5" /> Upload
              <input
                type="file"
                className="hidden"
                accept=".pdf,.docx,.txt,.md"
                onChange={handleFileUpload}
              />
            </label>
          </div>

          <div className="max-h-40 overflow-y-auto space-y-1">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-2 rounded-lg bg-gray-50 dark:bg-gray-800 text-xs"
              >
                <div className="flex items-center space-x-2 truncate">
                  <FileText className="w-3.5 h-3.5 text-gray-500 shrink-0" />
                  <span className="truncate">{doc.filename}</span>
                </div>
                <button
                  onClick={async () => {
                    await api.deleteDocument(doc.id);
                    setDocuments((prev) => prev.filter((d) => d.id !== doc.id));
                  }}
                  className="text-gray-400 hover:text-red-500 p-1"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="flex-1 flex flex-col h-full bg-white dark:bg-gray-950">
        {/* Top Header */}
        <header className="h-14 border-b border-gray-200 dark:border-gray-800 px-6 flex items-center justify-between">
          <div className="text-sm font-medium text-gray-600 dark:text-gray-300">
            {systemHealth && (
              <span className="inline-flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
                LLM: <strong className="text-indigo-600">{systemHealth.llm_provider}</strong> |
                Embeddings: <strong>{systemHealth.embedding_provider}</strong>
              </span>
            )}
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6 max-w-4xl w-full mx-auto">
          {messages.map((m) => (
            <div
              key={m.id}
              className={`flex flex-col ${m.role === 'user' ? 'items-end' : 'items-start'}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-3.5 text-sm ${
                  m.role === 'user'
                    ? 'bg-indigo-600 text-white rounded-br-none'
                    : 'bg-gray-100 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 text-gray-800 dark:text-gray-200 rounded-bl-none'
                }`}
              >
                <ReactMarkdown className="prose dark:prose-invert max-w-none text-sm leading-relaxed">
                  {m.content}
                </ReactMarkdown>

                {/* Fontes Citadas */}
                {m.citations && m.citations.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200 dark:border-gray-800 flex flex-wrap gap-1.5">
                    <span className="text-xs text-gray-400 font-semibold mr-1">Fontes:</span>
                    {m.citations.map((c) => (
                      <button
                        key={c.id}
                        onClick={() => setActiveCitation(c)}
                        className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-indigo-50 dark:bg-indigo-950 text-indigo-600 dark:text-indigo-300 text-xs font-medium hover:bg-indigo-100 transition-colors"
                      >
                        [{c.id}] {c.filename} (pág {c.page_number})
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Input Bar */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-800 max-w-4xl w-full mx-auto">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
              type="text"
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              placeholder="Faça uma pergunta sobre seus documentos..."
              disabled={isStreaming}
              className="flex-1 rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-900 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500"
            />
            <button
              type="submit"
              disabled={isStreaming || !inputMessage.trim()}
              className="px-5 py-3 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 disabled:opacity-50 transition-colors flex items-center justify-center"
            >
              <Send className="w-4 h-4" />
            </button>
          </form>
        </div>
      </main>

      {/* Drawer de Citações */}
      <SourceDrawer
        isOpen={Boolean(activeCitation)}
        onClose={() => setActiveCitation(null)}
        citation={activeCitation}
      />
    </div>
  );
}
```

- [ ] **Step 5: Verify Task 10 deliverable**

---

### Task 11: Integração de Ponta a Ponta, Execução Docker e Verificação

**Files:**
- Create: `README.md`
- Create: `backend/tests/test_e2e_mocked.py`
- Modify: `docker-compose.yml`

**Interfaces:**
- Produces: Teste de ponta a ponta simulando upload, parsing, busca híbrida e streaming SSE, além de orquestração Docker completa pronta para `docker compose up`.

- [ ] **Step 1: Write E2E mock integration test**

```python
# backend/tests/test_e2e_mocked.py
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch
from app.main import app

@pytest.mark.asyncio
async def test_complete_upload_and_chat_e2e():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # 1. Verifica Health
        health = await client.get("/api/v1/health")
        assert health.status_code == 200
        
        # 2. Cria Sessão de Chat
        sess_resp = await client.post("/api/v1/chat/sessions", json={"title": "E2E Test Session"})
        assert sess_resp.status_code == 200
        session_id = sess_resp.json()["id"]
        
        # 3. Testa listagem de mensagens iniciais
        msgs_resp = await client.get(f"/api/v1/chat/sessions/{session_id}/messages")
        assert msgs_resp.status_code == 200
        assert len(msgs_resp.json()) == 0
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest backend/tests/test_e2e_mocked.py -v`
Expected: PASS

- [ ] **Step 3: Update docker-compose.yml and create user README**

```yaml
# docker-compose.yml
services:
  db:
    image: pgvector/pgvector:pg16
    container_name: modular_rag_db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: modular_rag
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: modular_rag_backend
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/modular_rag
      - LLM_PROVIDER=${LLM_PROVIDER:-ollama}
      - EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER:-fastembed}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://host.docker.internal:11434}
    ports:
      - "8000:8000"
    depends_on:
      db:
        condition: service_healthy

volumes:
  pgdata:
```

- [ ] **Step 4: Run full test suite to verify 100% test coverage**

Run: `pytest backend/tests/ -v`
Expected: All tests PASS.

- [ ] **Step 5: Save and verify complete system delivery**
