# Modular RAG System

<p align="center">
  <b>Sistema de Geração Aumentada por Recuperação (RAG) Corporativo e Modular</b><br>
  <i>Modular, decoupled Enterprise Retrieval-Augmented Generation (RAG) System</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.115+-009688.svg" alt="FastAPI">
  <img src="https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg" alt="PostgreSQL">
  <img src="https://img.shields.io/badge/React-18-61DAFB.svg" alt="React">
  <img src="https://img.shields.io/badge/TypeScript-5-3178C6.svg" alt="TypeScript">
  <img src="https://img.shields.io/badge/TailwindCSS-3.4-38B2AC.svg" alt="TailwindCSS">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED.svg" alt="Docker">
</p>

---

[🇧🇷 Português](#-visão-geral-pt-br) | [🇺🇸 English](#-overview-en)

---

## 🇧🇷 Visão Geral (PT-BR)

O **Modular RAG** é uma solução completa e pronta para produção construída segundo os princípios da **Arquitetura Limpa (Ports & Adapters)**. O sistema permite upload, parsing e ingestão assíncrona de múltiplos formatos de documento (PDF, DOCX, TXT, MD), recuperação híbrida em duas fases (busca vetorial densa + busca lexical sparse com *Reciprocal Rank Fusion - RRF*), reclassificação semântica local via *Cross-Encoder* (FlashRank ONNX), e streaming em tempo real token a token via *Server-Sent Events (SSE)* com citação numérica de fontes (`[1]`, `[2]`).

### Principais Diferenciais
- **Desacoplamento Total de Provedores:** Alterne entre modelos locais (**Ollama**, **FastEmbed**) e em nuvem (**OpenAI**, **Gemini**) alterando apenas variáveis no `.env`.
- **Busca Híbrida de Alta Precisão:** Combinação do melhor dos dois mundos — relevância semântica (vetores densos com pgvector) e correspondência exata de termos (PostgreSQL Full-Text Search - tsvector) ponderados por RRF ($k=60$).
- **Reranker Local Leve:** Utiliza FlashRank com ONNX Runtime para reordenar candidatos sem necessidade de GPU/CUDA, filtrando conteúdos com score abaixo do threshold configurado.
- **Rastreabilidade e Citações:** Cada resposta gerada inclui citações clicáveis mapeando número de página, arquivo de origem, score de relevância e o trecho de texto correspondente.
- **Interface Moderna:** Frontend responsivo construído em React 18, TypeScript, Tailwind CSS e Lucide Icons, com alternância de sessões, upload por arrastar e soltar e gaveta lateral de fontes.

---

## 🇺🇸 Overview (EN)

**Modular RAG** is a production-ready, decoupled Retrieval-Augmented Generation system engineered around **Clean Architecture (Ports & Adapters)**. It supports multi-format document ingestion (PDF, DOCX, TXT, MD), two-phase hybrid search (dense embeddings + sparse PostgreSQL full-text search fused with *Reciprocal Rank Fusion - RRF*), local CPU cross-encoder reranking (FlashRank ONNX), and real-time token streaming over *Server-Sent Events (SSE)* with exact source citations (`[1]`, `[2]`).

### Key Features
- **Strictly Decoupled Adapters:** Switch LLM and Embedding providers (Ollama, FastEmbed, OpenAI, Gemini) via environment variables without changing any code.
- **Hybrid Retrieval (Dense + Sparse):** Leverages pgvector for semantic search and PostgreSQL FTS for keyword accuracy, combined via RRF ($k=60$).
- **Local Reranking without GPU:** Uses FlashRank ONNX Runtime to filter irrelevant snippets below confidence thresholds.
- **Precise Source Grounding:** Detailed citation badges linking assertions directly to page numbers and exact document excerpts.
- **Modern Fullstack Web UI:** Built with React 18, TypeScript, Tailwind CSS, and Server-Sent Events for fluid interaction.

---

## 📐 Arquitetura do Sistema / System Architecture

```mermaid
flowchart TD
    subgraph Client ["Frontend (React 18 + Vite + Tailwind)"]
        UI[Interface Web / Chat Window]
        DocUpload[Upload de Documentos]
        SourcesDrawer[Drawer de Citações]
    end

    subgraph API ["Backend API (FastAPI - Clean Architecture)"]
        Routes[API Routes: /documents, /chat, /health]
        
        subgraph CoreServices ["Core Domain & Services"]
            IngestSvc[IngestionService]
            RetrievSvc[RetrievalService]
            RAGSvc[RAGService]
            Chunker[ChunkingService]
        end

        subgraph Ports ["Interfaces / Ports"]
            IVector[VectorStore]
            ILLM[LLMProvider]
            IEmbed[EmbeddingProvider]
            IRerank[Reranker]
            IParser[DocumentParser]
        end

        subgraph Adapters ["Concretes / Adapters"]
            PGVector[PGVectorStore (pgvector)]
            AdapFastEmbed[FastEmbedAdapter]
            AdapOllama[OllamaAdapter]
            AdapOpenAI[OpenAIAdapter]
            AdapGemini[GeminiAdapter]
            FlashRank[FlashRankReranker (ONNX)]
            Parsers[PDF / DOCX / Text Parsers]
        end
    end

    subgraph Storage ["Banco de Dados Unificado"]
        PG[(PostgreSQL 16 + pgvector)]
    end

    %% Fluxo de Ingestão
    DocUpload -->|POST /api/v1/documents/upload| Routes
    Routes --> IngestSvc
    IngestSvc --> Parsers
    IngestSvc --> Chunker
    IngestSvc --> IEmbed
    IngestSvc --> IVector
    IVector --> PG

    %% Fluxo de Pergunta & Streaming
    UI -->|POST /api/v1/chat/{session}/stream| Routes
    Routes --> RAGSvc
    RAGSvc --> RetrievSvc
    RetrievSvc --> IVector
    RetrievSvc --> IRerank
    RAGSvc --> ILLM
    RAGSvc -->|SSE Stream: sources, token, done| UI
    UI -.-> SourcesDrawer
```

---

## 🚀 Como Executar / Getting Started

### Pré-requisitos / Prerequisites
- [Docker](https://docs.docker.com/get-docker/) & [Docker Compose](https://docs.docker.com/compose/)
- *(Opcional)* [Ollama](https://ollama.ai) instalado na máquina host se optar por rodar modelos 100% locais sem Docker para a inferência:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```

---

### Executando com Docker Compose (Recomendado)

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/modular-rag.git
   cd modular-rag
   ```

2. **Configure o arquivo de ambiente (`.env`):**
   ```bash
   cp .env.example .env
   ```
   *Edite o arquivo `.env` para apontar para o provedor desejado (veja [Configuração](#-configuração-de-ambiente--environment-variables)).*

3. **Inicie os serviços:**
   ```bash
   docker compose up --build -d
   ```

4. **Acesse a aplicação:**
   - **Frontend Web UI:** [http://localhost:3000](http://localhost:3000)
   - **Swagger / OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Health Check:** [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

5. **Para parar os contêineres:**
   ```bash
   docker compose down
   ```

---

## ⚙️ Configuração de Ambiente / Environment Variables

O arquivo `.env` controla o comportamento de todos os componentes do sistema:

```ini
# ==============================================================================
# Provedor de LLM: "ollama", "openai" ou "gemini"
# ==============================================================================
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2:3b

# ==============================================================================
# Provedor de Embeddings: "fastembed", "ollama" ou "openai"
# ==============================================================================
EMBEDDING_PROVIDER=fastembed
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIMENSION=384

# ==============================================================================
# Chaves de API (necessárias apenas se utilizar OpenAI ou Gemini)
# ==============================================================================
OPENAI_API_KEY=
GEMINI_API_KEY=
OLLAMA_BASE_URL=http://host.docker.internal:11434

# ==============================================================================
# Banco de Dados PostgreSQL + pgvector
# ==============================================================================
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/modular_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=modular_rag

# ==============================================================================
# Busca Híbrida & Reranking
# ==============================================================================
RERANKER_THRESHOLD=0.30
RERANKER_TOP_K=4
DENSE_TOP_K=20
SPARSE_TOP_K=20

# ==============================================================================
# Chunking
# ==============================================================================
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

---

## 🔌 API Endpoints

| Método | Endpoint | Descrição |
| :--- | :--- | :--- |
| `GET` | `/api/v1/health` | Status operacional e provedores ativos |
| `POST` | `/api/v1/documents/upload` | Upload e ingestão de documento (multipart/form-data) |
| `GET` | `/api/v1/documents` | Lista metadados de documentos indexados |
| `DELETE` | `/api/v1/documents/{document_id}` | Exclui documento e seus chunks vetoriais em cascata |
| `POST` | `/api/v1/chat/sessions` | Cria nova sessão de conversa |
| `GET` | `/api/v1/chat/sessions` | Lista histórico de sessões |
| `GET` | `/api/v1/chat/sessions/{session_id}/messages` | Recupera mensagens e citações da sessão |
| `POST` | `/api/v1/chat/{session_id}/stream` | Streaming de resposta SSE (`sources`, `token`, `done`, `error`) |

### Estrutura dos Eventos SSE (`/stream`)
1. **`event: sources`**: Retorna lista de chunks relevantes pós-reranking (`id`, `filename`, `page_number`, `score`, `snippet`).
2. **`event: token`**: Transmite deltas parciais de texto gerados pelo LLM.
3. **`event: done`**: Confirma conclusão e identificador da mensagem persistida.
4. **`event: error`**: Emite mensagem de erro sem interrupção abrupta do socket em falhas de LLM ou rede.

---

## 🧪 Testes e Qualidade / Testing & Quality

O projeto conta com uma suíte de testes automatizados completa com 100% de cobertura nos fluxos principais:

### Testes do Backend (Pytest)
```bash
# Entrar no ambiente virtual ou backend
cd backend
source .venv/bin/activate

# Executar todos os testes unitários e de integração
pytest tests/ -v

# Executar apenas o teste end-to-end simulado
pytest tests/test_e2e_mocked.py -v
```

### Verificação do Frontend (TypeScript & Vite Build)
```bash
cd frontend
npm install
npm run build
```

---

## 📂 Estrutura de Diretórios / Project Structure

```
ModularRAG/
├── docker-compose.yml          # Orquestração (Postgres + pgvector, Backend, Frontend)
├── .env.example                # Modelo de variáveis de ambiente
├── README.md                   # Documentação do projeto
├── backend/
│   ├── Dockerfile              # Dockerfile da API Python 3.11
│   ├── pyproject.toml          # Dependências do projeto (FastAPI, SQLAlchemy, etc.)
│   ├── app/
│   │   ├── main.py             # Entrada da aplicação FastAPI e lifespan
│   │   ├── core/               # Configurações Pydantic e Factory de Provedores
│   │   ├── domain/             # Interfaces abstratas e modelos de domínio
│   │   ├── adapters/           # Implementações concretas (VectorStore, LLMs, Embeddings, Reranker)
│   │   ├── services/           # Regra de negócio (Ingestion, Retrieval com RRF, RAG)
│   │   ├── db/                 # Modelos ORM SQLAlchemy e conexão asyncpg
│   │   └── api/                # Rotas da API REST e Schemas Pydantic
│   └── tests/                  # Suíte de testes automatizados com Pytest
└── frontend/
    ├── Dockerfile              # Multi-stage build com Node 20 e Nginx Alpine
    ├── nginx.conf              # Reverse proxy Nginx com buffer SSE desativado
    ├── package.json            # Dependências React, Tailwind, Lucide
    ├── vite.config.ts          # Configurações Vite e proxy dev
    └── src/
        ├── components/         # Componentes React (Chat, Sidebar, Citations, Header)
        ├── services/           # Cliente HTTP e consumidor de streaming SSE
        └── types/              # Tipagens TypeScript compartilhadas
```

---

## 🛡️ Licença / License

Distribuído sob a licença MIT. Consulte `LICENSE` para mais informações.
