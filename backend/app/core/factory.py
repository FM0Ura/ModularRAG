from typing import Optional
from app.core.config import Settings, settings as default_settings
from app.domain.interfaces import LLMProvider, EmbeddingProvider
from app.adapters.llm.openai_adapter import OpenAIAdapter
from app.adapters.llm.ollama_adapter import OllamaAdapter
from app.adapters.llm.gemini_adapter import GeminiAdapter
from app.adapters.embeddings.fastembed_adapter import FastEmbedAdapter
from app.adapters.embeddings.openai_embed import OpenAIEmbeddingAdapter
from app.adapters.embeddings.ollama_embed import OllamaEmbeddingAdapter


def get_llm_provider(config: Optional[Settings] = None) -> LLMProvider:
    cfg = default_settings if config is None else config
    provider = cfg.LLM_PROVIDER.lower()
    if provider == "openai":
        return OpenAIAdapter(api_key=cfg.OPENAI_API_KEY, model=cfg.LLM_MODEL)
    elif provider == "gemini":
        return GeminiAdapter(api_key=cfg.GEMINI_API_KEY, model=cfg.LLM_MODEL)
    elif provider == "ollama":
        return OllamaAdapter(base_url=cfg.OLLAMA_BASE_URL, model=cfg.LLM_MODEL)
    else:
        raise ValueError(f"Provedor LLM desconhecido: '{provider}'")


def get_embedding_provider(config: Optional[Settings] = None) -> EmbeddingProvider:
    cfg = default_settings if config is None else config
    provider = cfg.EMBEDDING_PROVIDER.lower()
    if provider == "openai":
        return OpenAIEmbeddingAdapter(
            api_key=cfg.OPENAI_API_KEY,
            model=cfg.EMBEDDING_MODEL,
            dimension=cfg.EMBEDDING_DIMENSION,
        )
    elif provider == "ollama":
        return OllamaEmbeddingAdapter(
            base_url=cfg.OLLAMA_BASE_URL,
            model=cfg.EMBEDDING_MODEL,
            dimension=cfg.EMBEDDING_DIMENSION,
        )
    elif provider == "fastembed":
        return FastEmbedAdapter(
            model_name=cfg.EMBEDDING_MODEL,
            dimension=cfg.EMBEDDING_DIMENSION,
        )
    else:
        raise ValueError(f"Provedor de Embedding desconhecido: '{provider}'")
