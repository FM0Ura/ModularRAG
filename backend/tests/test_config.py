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

def test_settings_custom_values():
    settings = Settings(
        LLM_PROVIDER="openai",
        LLM_MODEL="gpt-4o-mini",
        OPENAI_API_KEY="test-key",
        EMBEDDING_PROVIDER="openai",
        EMBEDDING_MODEL="text-embedding-3-small",
        EMBEDDING_DIMENSION=1536,
        RERANKER_THRESHOLD=0.5,
        RERANKER_TOP_K=10,
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=200,
    )
    assert settings.LLM_PROVIDER == "openai"
    assert settings.LLM_MODEL == "gpt-4o-mini"
    assert settings.OPENAI_API_KEY == "test-key"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.EMBEDDING_MODEL == "text-embedding-3-small"
    assert settings.EMBEDDING_DIMENSION == 1536
    assert settings.RERANKER_THRESHOLD == 0.5
    assert settings.RERANKER_TOP_K == 10
    assert settings.CHUNK_SIZE == 1000
    assert settings.CHUNK_OVERLAP == 200
