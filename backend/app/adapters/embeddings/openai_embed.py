from typing import List, Optional
from openai import AsyncOpenAI
from app.domain.interfaces import EmbeddingProvider


class OpenAIEmbeddingAdapter(EmbeddingProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ):
        self.client = AsyncOpenAI(api_key=api_key or "dummy-key")
        self.model = model
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        res = await self.client.embeddings.create(input=text, model=self.model)
        return res.data[0].embedding

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        res = await self.client.embeddings.create(input=texts, model=self.model)
        return [item.embedding for item in res.data]

    @property
    def dimension(self) -> int:
        return self._dim
