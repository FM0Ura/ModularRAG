from typing import List, Optional, Any
from fastembed import TextEmbedding
from app.domain.interfaces import EmbeddingProvider


class FastEmbedAdapter(EmbeddingProvider):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        model: Optional[Any] = None,
    ):
        self._model = model if model is not None else TextEmbedding(model_name=model_name)
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        embeddings = list(self._model.embed([text]))
        return embeddings[0].tolist()

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = list(self._model.embed(texts))
        return [e.tolist() for e in embeddings]

    @property
    def dimension(self) -> int:
        return self._dim
