import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.core.config import Settings
from app.core.factory import get_llm_provider, get_embedding_provider
from app.adapters.embeddings.fastembed_adapter import FastEmbedAdapter
from app.adapters.embeddings.openai_embed import OpenAIEmbeddingAdapter
from app.adapters.embeddings.ollama_embed import OllamaEmbeddingAdapter
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.gemini_adapter import GeminiAdapter
from app.domain.models import ChatMessage


# --- Factory Tests ---

def test_factory_returns_configured_providers():
    custom_settings = Settings(
        LLM_PROVIDER="ollama",
        EMBEDDING_PROVIDER="fastembed",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    llm = get_llm_provider(custom_settings)
    embed = get_embedding_provider(custom_settings)

    assert isinstance(llm, OllamaAdapter)
    assert isinstance(embed, FastEmbedAdapter)
    assert embed.dimension == 384


def test_factory_returns_openai_providers():
    custom_settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o",
        OPENAI_API_KEY="sk-test-key",
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="text-embedding-3-large",
        EMBEDDING_DIMENSION=3072,
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    llm = get_llm_provider(custom_settings)
    embed = get_embedding_provider(custom_settings)

    assert isinstance(llm, OpenAIAdapter)
    assert llm.model == "gpt-4o"
    assert isinstance(embed, OpenAIEmbeddingAdapter)
    assert embed.model == "text-embedding-3-large"
    assert embed.dimension == 3072


def test_factory_returns_gemini_and_ollama_embed():
    custom_settings = Settings(
        LLM_PROVIDER="gemini",
        LLM_MODEL="gemini-1.5-pro",
        GEMINI_API_KEY="gm-test-key",
        EMBEDDING_PROVIDER="ollama",
        EMBEDDING_MODEL="nomic-embed-text",
        EMBEDDING_DIMENSION=768,
        OLLAMA_BASE_URL="http://ollama-host:11434",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    llm = get_llm_provider(custom_settings)
    embed = get_embedding_provider(custom_settings)

    assert isinstance(llm, GeminiAdapter)
    assert llm.model == "gemini-1.5-pro"
    assert isinstance(embed, OllamaEmbeddingAdapter)
    assert embed.base_url == "http://ollama-host:11434"
    assert embed.dimension == 768


def test_factory_default_settings():
    llm = get_llm_provider()
    embed = get_embedding_provider()
    assert isinstance(llm, OllamaAdapter)
    assert isinstance(embed, FastEmbedAdapter)


def test_factory_unknown_llm_provider_raises():
    custom_settings = Settings(
        LLM_PROVIDER="anthropic",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    with pytest.raises(ValueError, match="Provedor LLM desconhecido: 'anthropic'"):
        get_llm_provider(custom_settings)


def test_factory_unknown_embedding_provider_raises():
    custom_settings = Settings(
        EMBEDDING_PROVIDER="cohere",
        DATABASE_URL="postgresql+asyncpg://postgres:postgres@localhost:5432/modular_rag"
    )
    with pytest.raises(ValueError, match="Provedor de Embedding desconhecido: 'cohere'"):
        get_embedding_provider(custom_settings)


# --- FastEmbedAdapter Tests ---

@pytest.mark.asyncio
async def test_fastembed_adapter_embed_query_and_documents():
    mock_model = MagicMock()
    # mock_model.embed returns an iterable of objects with .tolist()
    class MockEmbedding:
        def __init__(self, vec):
            self._vec = vec
        def tolist(self):
            return self._vec

    mock_model.embed.side_effect = lambda texts: [MockEmbedding([0.1, 0.2, 0.3]) for _ in texts]

    adapter = FastEmbedAdapter(model_name="BAAI/bge-small-en-v1.5", dimension=3)
    adapter._model = mock_model

    query_vec = await adapter.embed_query("Qual a capital da França?")
    assert query_vec == [0.1, 0.2, 0.3]
    mock_model.embed.assert_called_with(["Qual a capital da França?"])

    docs_vecs = await adapter.embed_documents(["doc1", "doc2"])
    assert docs_vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert adapter.dimension == 3

    empty_docs = await adapter.embed_documents([])
    assert empty_docs == []


# --- OpenAIEmbeddingAdapter Tests ---

@pytest.mark.asyncio
async def test_openai_embedding_adapter():
    adapter = OpenAIEmbeddingAdapter(api_key="sk-fake", model="text-embedding-3-small", dimension=1536)
    mock_client = AsyncMock()

    class Item:
        def __init__(self, emb):
            self.embedding = emb

    class Response:
        def __init__(self, data):
            self.data = data

    mock_client.embeddings.create.return_value = Response([Item([0.4, 0.5, 0.6])])
    adapter.client = mock_client

    vec = await adapter.embed_query("teste")
    assert vec == [0.4, 0.5, 0.6]
    mock_client.embeddings.create.assert_awaited_once_with(input="teste", model="text-embedding-3-small")

    mock_client.embeddings.create.reset_mock()
    mock_client.embeddings.create.return_value = Response([Item([0.1, 0.2]), Item([0.3, 0.4])])

    docs = await adapter.embed_documents(["t1", "t2"])
    assert docs == [[0.1, 0.2], [0.3, 0.4]]
    assert adapter.dimension == 1536

    empty_docs = await adapter.embed_documents([])
    assert empty_docs == []


# --- OllamaEmbeddingAdapter Tests ---

@pytest.mark.asyncio
async def test_ollama_embedding_adapter():
    adapter = OllamaEmbeddingAdapter(base_url="http://localhost:11434/", model="nomic-embed-text", dimension=768)
    assert adapter.base_url == "http://localhost:11434"
    assert adapter.dimension == 768

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"embedding": [0.7, 0.8, 0.9]}

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        vec = await adapter.embed_query("teste ollama")
        assert vec == [0.7, 0.8, 0.9]
        mock_post.assert_awaited_once_with(
            "http://localhost:11434/api/embeddings",
            json={"model": "nomic-embed-text", "prompt": "teste ollama"}
        )

        mock_post.reset_mock()
        docs = await adapter.embed_documents(["t1", "t2"])
        assert docs == [[0.7, 0.8, 0.9], [0.7, 0.8, 0.9]]
        assert mock_post.await_count == 2


# --- OpenAIAdapter Tests ---

@pytest.mark.asyncio
async def test_openai_llm_adapter_generate_stream():
    adapter = OpenAIAdapter(api_key="sk-fake", model="gpt-4o-mini")

    class ChunkDelta:
        def __init__(self, content):
            self.content = content

    class ChunkChoice:
        def __init__(self, content):
            self.delta = ChunkDelta(content)

    class StreamChunk:
        def __init__(self, content):
            self.choices = [ChunkChoice(content)] if content is not None else []

    async def fake_stream():
        yield StreamChunk("Olá")
        yield StreamChunk(", ")
        yield StreamChunk("Mundo")
        yield StreamChunk(None)  # finish chunk with no content
        yield StreamChunk("")

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = fake_stream()
    adapter.client = mock_client

    history = [ChatMessage(role="assistant", content="Olá anterior")]
    tokens = []
    async for token in adapter.generate_stream(
        prompt="Como vai?",
        system_prompt="Você é um assistente RAG.",
        history=history
    ):
        tokens.append(token)

    assert "".join(tokens) == "Olá, Mundo"
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Você é um assistente RAG."},
            {"role": "assistant", "content": "Olá anterior"},
            {"role": "user", "content": "Como vai?"}
        ],
        stream=True
    )


# --- GeminiAdapter Tests ---

@pytest.mark.asyncio
async def test_gemini_llm_adapter_generate_stream():
    adapter = GeminiAdapter(api_key="gm-fake", model="gemini-1.5-flash")
    assert str(adapter.client.base_url) == "https://generativelanguage.googleapis.com/v1beta/openai/"

    class ChunkDelta:
        def __init__(self, content):
            self.content = content

    class ChunkChoice:
        def __init__(self, content):
            self.delta = ChunkDelta(content)

    class StreamChunk:
        def __init__(self, content):
            self.choices = [ChunkChoice(content)] if content is not None else []

    async def fake_stream():
        yield StreamChunk("Resposta ")
        yield StreamChunk("Gemini")

    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = fake_stream()
    adapter.client = mock_client

    tokens = []
    async for token in adapter.generate_stream(
        prompt="Pergunta Gemini",
        system_prompt="Sistema",
        history=[]
    ):
        tokens.append(token)

    assert "".join(tokens) == "Resposta Gemini"
    mock_client.chat.completions.create.assert_awaited_once_with(
        model="gemini-1.5-flash",
        messages=[
            {"role": "system", "content": "Sistema"},
            {"role": "user", "content": "Pergunta Gemini"}
        ],
        stream=True
    )


# --- OllamaAdapter Tests ---

@pytest.mark.asyncio
async def test_ollama_llm_adapter_generate_stream():
    adapter = OllamaAdapter(base_url="http://localhost:11434/", model="llama3.2")
    assert adapter.base_url == "http://localhost:11434"

    lines = [
        json.dumps({"message": {"content": "Resposta "}}),
        "",  # empty line should be skipped
        json.dumps({"message": {"content": "Ollama"}}),
        json.dumps({"done": True, "message": {"content": ""}})
    ]

    class AsyncLineIterator:
        def __init__(self, items):
            self.items = items

        def __aiter__(self):
            self._iter = iter(self.items)
            return self

        async def __anext__(self):
            try:
                return next(self._iter)
            except StopIteration:
                raise StopAsyncIteration

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines.return_value = AsyncLineIterator(lines)

    class FakeStreamContext:
        async def __aenter__(self):
            return mock_response
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    with patch("httpx.AsyncClient.stream", return_value=FakeStreamContext()) as mock_stream:
        tokens = []
        async for token in adapter.generate_stream(
            prompt="Prompt Ollama",
            system_prompt="System Ollama",
            history=[ChatMessage(role="user", content="msg1")]
        ):
            tokens.append(token)

        assert "".join(tokens) == "Resposta Ollama"
        mock_stream.assert_called_once_with(
            "POST",
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": "System Ollama"},
                    {"role": "user", "content": "msg1"},
                    {"role": "user", "content": "Prompt Ollama"}
                ],
                "stream": True
            }
        )
