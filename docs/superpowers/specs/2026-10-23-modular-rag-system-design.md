# Especificação Técnica de Design: Modular RAG System

- **Data:** 23 de Outubro de 2026
- **Status:** Aprovado em Brainstorming (Pronto para Planejamento de Implementação)
- **Autor/Time:** Antigravity AI & Engenheiro de Software

---

## 1. Visão Geral do Sistema

O **Modular RAG** é um sistema completo, desacoplado e escalável de *Retrieval-Augmented Generation* (RAG) voltado para bases de conhecimento técnicas e corporativas. O sistema cobre desde o upload e ingestão de documentos estruturados e semi-estruturados (PDF, DOCX, TXT, Markdown) até o streaming de respostas ao usuário com citação exata de fontes (arquivo, número de página e trecho original).

### Objetivos Principais
1. **Desacoplamento Rigoroso (Clean Architecture):** Interfaces abstratas e bem definidas para LLMs, Embeddings, Armazenamento Vetorial e Parsers de Documentos, permitindo trocar componentes ou provedores via `.env` sem alterar a regra de negócio.
2. **Alta Fidelidade de Resposta:** Pipeline de recuperação avançado com **Busca Híbrida** (Vetorial Densa via pgvector + Esparsa Textual via PostgreSQL Full-Text Search), fusão por **RRF (Reciprocal Rank Fusion)** e reclassificação via **Reranker neural (Cross-Encoder / FlashRank)**.
3. **Execução Híbrida e Flexível:** Suporte nativo tanto para provedores cloud (**OpenAI**, **Google Gemini**) quanto para execução 100% local e privada (**Ollama**).
4. **Experiência do Usuário (UX):** Interface moderna em **React + Vite + Tailwind CSS** com visualização de fontes citadas em gaveta lateral e streaming de respostas em tempo real via **Server-Sent Events (SSE)**.
5. **Simplicidade Operacional:** Todo o ambiente (banco relacional/vetorial unificado, backend e frontend) orquestrado por um único `docker-compose.yml`.

---

## 2. Arquitetura Geral e Estrutura de Diretórios

O projeto segue a arquitetura de **Portas e Adaptadores (Hexagonal / Clean Architecture)**:

```
ModularRAG/
├── docker-compose.yml             # Orquestra PostgreSQL + pgvector, Backend e Frontend
├── .env.example                   # Modelo documentado de variáveis de ambiente
├── backend/
│   ├── pyproject.toml             # Gerenciamento de dependências Python
│   ├── Dockerfile                 # Imagem Docker do Backend
│   ├── app/
│   │   ├── main.py                # Entrada FastAPI, middlewares e CORS
│   │   ├── core/
│   │   │   ├── config.py          # Pydantic Settings (.env)
│   │   │   └── factory.py         # Injeção de dependência dos provedores (LLM, Embeddings, Store)
│   │   ├── domain/                # Contratos / Interfaces Puras (Zero dependências externas)
│   │   │   ├── interfaces.py      # LLMProvider, EmbeddingProvider, VectorStore, Reranker, DocumentParser
│   │   │   └── models.py          # Dataclasses (DocumentChunk, SearchResult, ChatMessage, Citation)
│   │   ├── adapters/              # Implementações Concretas das Interfaces
│   │   │   ├── llm/
│   │   │   │   ├── openai_adapter.py
│   │   │   │   ├── gemini_adapter.py
│   │   │   │   └── ollama_adapter.py
│   │   │   ├── embeddings/
│   │   │   │   ├── openai_embed.py
│   │   │   │   ├── ollama_embed.py
│   │   │   │   └── fastembed_adapter.py
│   │   │   ├── vectorstore/
│   │   │   │   └── pgvector_store.py  # SQLAlchemy Async + pgvector + FTS
│   │   │   ├── parsers/
│   │   │   │   ├── pdf_parser.py      # pypdf / extração com numeração de página
│   │   │   │   ├── docx_parser.py     # python-docx
│   │   │   │   └── text_parser.py     # txt e markdown UTF-8
│   │   │   └── reranker/
│   │   │       └── flashrank_adapter.py # FlashRank (Cross-Encoder em CPU ultrarrápido)
│   │   ├── services/              # Casos de Uso / Regra de Negócio
│   │   │   ├── chunking_service.py   # Recursive Character Splitting com overlap e metadados
│   │   │   ├── ingestion_service.py  # Pipeline completo de upload -> parsing -> chunking -> vectorstore
│   │   │   ├── retrieval_service.py  # Orquestra busca híbrida + RRF + Reranker
│   │   │   └── rag_service.py        # Montagem do prompt com fontes, histórico e streaming
│   │   ├── api/                   # Camada de Apresentação HTTP
│   │   │   ├── routes/
│   │   │   │   ├── documents.py   # Upload, listagem, remoção de documentos
│   │   │   │   ├── chat.py        # Criação de sessões, histórico e streaming SSE (/chat/stream)
│   │   │   │   └── health.py      # Healthcheck do sistema e conexões
│   │   │   └── schemas.py         # Schemas Pydantic de requisição/resposta
│   │   └── db/
│   │       ├── session.py         # Engine e SessionMaker assíncronos do SQLAlchemy
│   │       └── models.py          # Modelos relacionais (Document, DocumentChunk, ChatSession, Message)
│   └── tests/                     # Testes unitários e de integração
│       ├── test_parsers.py
│       ├── test_chunker.py
│       ├── test_rrf.py
│       └── test_rag_flow.py
├── frontend/
│   ├── package.json
│   ├── Dockerfile
│   ├── vite.config.ts
│   ├── tailwind.config.js
│   └── src/
│       ├── types/                 # Interfaces TypeScript de API, Mensagens e Fontes
│       ├── services/              # Cliente HTTP e consumidor de EventSource / SSE
│       ├── hooks/                 # useChat, useDocuments, useSSE
│       ├── components/
│       │   ├── layout/Header.tsx
│       │   ├── sidebar/Sidebar.tsx
│       │   ├── sidebar/DocumentUpload.tsx
│       │   ├── sidebar/DocumentList.tsx
│       │   ├── chat/ChatWindow.tsx
│       │   ├── chat/ChatMessage.tsx
│       │   ├── chat/ChatInput.tsx
│       │   └── citations/SourceDrawer.tsx # Drawer lateral exibindo trecho original e página
│       └── App.tsx
└── docs/
    └── superpowers/specs/         # Especificações técnicas e planos
```

---

## 3. Interfaces de Domínio (Contratos Abstratos)

Todas as implementações concretas derivam de classes abstratas no arquivo `backend/app/domain/interfaces.py`:

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Optional
from app.domain.models import DocumentChunk, SearchResult, ChatMessage, Citation, RawDocumentPage

class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_bytes: bytes, filename: str) -> List[RawDocumentPage]:
        """Extrai texto e páginas preservando metadados de localização."""
        pass

class EmbeddingProvider(ABC):
    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Gera embedding para a consulta do usuário."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Gera embeddings em lote para chunks de documentos."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensão do vetor gerado (ex: 1536 ou 768)."""
        pass

class VectorStore(ABC):
    @abstractmethod
    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Persiste chunks, embeddings e vetores de busca textual."""
        pass

    @abstractmethod
    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        dense_top_k: int = 20,
        sparse_top_k: int = 20
    ) -> List[SearchResult]:
        """Executa busca densa (vetor) e esparsa (FTS) e retorna os candidatos preliminares."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> None:
        """Remove documento e todos os seus chunks associados."""
        pass

class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 5,
        threshold: float = 0.30
    ) -> List[SearchResult]:
        """Reavalia relevância dos candidatos utilizando modelo Cross-Encoder."""
        pass

class LLMProvider(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str,
        history: List[ChatMessage]
    ) -> AsyncIterator[str]:
        """Emite tokens da resposta de forma assíncrona para streaming."""
        pass
```

---

## 4. Pipeline de Ingestão e Modelo de Dados

### 4.1 Estratégia de Ingestão
1. **Recepção e Idempotência:**
   - Calcula o hash SHA-256 do arquivo. Se já existir na tabela `documents`, rejeita com erro informativo evitando desperdício de processamento e vetores redundantes.
2. **Parsing Especializado:**
   - **PDF:** Extrai página a página via `pypdf`, mantendo a página exata em que o texto foi encontrado.
   - **DOCX:** Extrai blocos de parágrafos e tabelas via `python-docx`.
   - **TXT/Markdown:** Processa o conteúdo estruturado com suporte a UTF-8.
3. **Divisão em Chunks (Recursive Splitting):**
   - Utiliza quebras hierárquicas: `["\n\n", "\n", ". ", " ", ""]`.
   - Tamanho configurável: `chunk_size = 800` caracteres com `chunk_overlap = 120` caracteres.
   - Cada chunk herda o identificador do documento, o índice do chunk, o número da página de origem e a contagem de caracteres/tokens.
4. **Vetorização em Batch:**
   - Os chunks são enviados ao `EmbeddingProvider` em lotes (batch size de 64 chunks) para otimizar tempo de rede e custo de chamada.
5. **Persistência Relacional e Vetorial Unificada:**
   - Gravados em transação única no PostgreSQL.

### 4.2 Esquema Relacional no PostgreSQL

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    file_hash VARCHAR(64) UNIQUE NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    file_size_bytes BIGINT NOT NULL,
    total_pages INT DEFAULT 1,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    page_number INT NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536), -- Ajustável conforme o modelo ativo no .env
    tsv_content TSVECTOR GENERATED ALWAYS AS (to_tsvector('portuguese', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Índices otimizados
CREATE INDEX idx_chunks_embedding_hnsw ON document_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_tsv ON document_chunks USING gin (tsv_content);
CREATE INDEX idx_chunks_doc_id ON document_chunks (document_id);

CREATE TABLE chat_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title VARCHAR(255) NOT NULL DEFAULT 'Nova Conversa',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE chat_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

---

## 5. Pipeline de Recuperação Avançada (Hybrid + RRF + Reranker)

A recuperação de contexto opera em três camadas estritas:

1. **Camada 1: Busca Híbrida no PostgreSQL**
   - **Densa:** Recupera os top 20 chunks com menor distância de cosseno:
     `SELECT id, content, page_number, 1 - (embedding <=> :query_vector) AS dense_score FROM document_chunks ORDER BY embedding <=> :query_vector LIMIT 20;`
   - **Esparsa:** Recupera os top 20 chunks pelo ranking BM25 do Postgres:
     `SELECT id, content, page_number, ts_rank_cd(tsv_content, plainto_tsquery('portuguese', :query_text)) AS sparse_score FROM document_chunks WHERE tsv_content @@ plainto_tsquery('portuguese', :query_text) ORDER BY sparse_score DESC LIMIT 20;`

2. **Camada 2: Reciprocal Rank Fusion (RRF)**
   - O algoritmo calcula a pontuação consolidada de cada chunk $d$:
     $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{Rank}_m(d)}$$
   - Extrai os 15 melhores candidatos combinados.

3. **Camada 3: Re-ranking Neural com FlashRank (Cross-Encoder)**
   - Os 15 candidatos são submetidos ao `FlashRankReranker` que avalia a interação profunda de atenção entre a pergunta e o trecho na CPU.
   - Chunks com score abaixo de `0.30` são descartados.
   - Seleciona os top 4 ou 5 chunks com maior relevância comprovada.

---

## 6. Orquestração do Prompt, Provedores e Streaming SSE

### 6.1 Formatação do Prompt e Injeção de Fontes
O sistema gera o prompt final injetando as fontes numeradas de forma estruturada:

```text
Você é um assistente técnico especialista e confiável.
Sua tarefa é responder à pergunta do usuário baseando-se estritamente nas FONTES fornecidas abaixo.

Regras Obrigatórias:
1. Responda apenas com base no conteúdo das fontes. Não invente ou presuma informações.
2. Cada afirmação deve ser respaldada por uma referência numérica [1], [2], correspondente à fonte consultada.
3. Se as fontes fornecidas não contiverem a informação necessária para responder, diga clara e educadamente que os documentos cadastrados não contêm essa informação.

FONTES CONSULTADAS:
[FONTE 1] (Arquivo: manual_servidor.pdf | Página: 8 | Relevância: 0.96)
Trecho relevante do documento...

[FONTE 2] (Arquivo: relatorio_anual.docx | Página: 15 | Relevância: 0.88)
Trecho relevante do documento...

PERGUNTA DO USUÁRIO:
{pergunta}
```

### 6.2 Protocolo de Streaming (Server-Sent Events)
O endpoint `POST /api/v1/chat/{session_id}/stream` emite mensagens formatadas no padrão SSE:
1. **Evento `sources` (imediato):**
   ```json
   event: sources
   data: {"citations": [{"id": 1, "filename": "manual_servidor.pdf", "page": 8, "score": 0.96, "snippet": "..."}]}
   ```
2. **Evento `token` (repetido a cada palavra gerada):**
   ```json
   event: token
   data: {"delta": "Para "}
   ```
3. **Evento `done` (ao concluir):**
   ```json
   event: done
   data: {"message_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d", "session_id": "..."}
   ```

---

## 7. Frontend (React + Vite + Tailwind CSS)

### 7.1 Componentes Principais
1. **`Sidebar`:**
   - Lista de conversas com botão de criar nova e alternar entre sessões.
   - Componente de dropzone com drag-and-drop para arquivos `.pdf`, `.docx`, `.txt`, `.md`.
   - Listagem de documentos indexados exibindo tamanho, páginas e opção de exclusão.
2. **`ChatWindow`:**
   - Exibição de mensagens com renderizador de Markdown (`react-markdown` + syntax highlighter).
   - Renderização dos números de citação (`[1]`, `[2]`) como botões interativos estilizados.
   - Barra de digitação com atalhos de envio e botão de interromper geração.
3. **`SourceDrawer`:**
   - Painel lateral ativado ao clicar em qualquer citação ou no botão "Ver Fontes" de uma mensagem.
   - Mostra o nome do arquivo, página, pontuação de relevância do Reranker e o trecho original destacado.

---

## 8. Estratégia de Testes Automatizados

A suíte de testes do backend com `pytest` cobrirá:
1. **`test_parsers.py`:**
   - Validação de extração correta de texto e páginas em amostras de PDF, DOCX e TXT.
2. **`test_chunker.py`:**
   - Validação de divisão de texto respeitando tamanhos e overlap sem cortes incorretos em palavras.
3. **`test_rrf.py`:**
   - Validação matemática de que o Reciprocal Rank Fusion prioriza corretamente itens com alta classificação combinada.
4. **`test_rag_flow.py`:**
   - Teste de ponta a ponta do pipeline usando mocks de `LLMProvider` e `EmbeddingProvider`, garantindo execução sem requisições externas ou custos de API.

---

## 9. Configuração de Variáveis de Ambiente (`.env.example`)

```bash
# Provedor de LLM: "openai", "gemini" ou "ollama"
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Provedor de Embeddings: "openai", "ollama" ou "fastembed"
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536

# Chaves de API (quando aplicável)
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
OLLAMA_BASE_URL=http://localhost:11434

# Banco de Dados
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag

# Reranker
RERANKER_THRESHOLD=0.30
RERANKER_TOP_K=4
```

---

## 10. Critérios de Sucesso e Verificação

- [ ] Ingestão de arquivos PDF, DOCX, TXT e MD funcionando com cálculo de hash idempotente.
- [ ] Chunks e embeddings persistidos com sucesso no PostgreSQL com extensão `pgvector`.
- [ ] Busca Híbrida combinando vetores e FTS com RRF e reordenação via FlashRank.
- [ ] Streaming SSE em tempo real enviando fontes antes dos tokens da resposta.
- [ ] Frontend React exibindo chat fluido, markdown, citação interativa e drawer de fontes.
- [ ] Troca de provedor via `.env` operacional sem alteração de código.
- [ ] Suíte de testes automatizados executando e passando com 100% de sucesso.
