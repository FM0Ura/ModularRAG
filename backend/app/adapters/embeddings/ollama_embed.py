from typing import List
import httpx
from app.domain.interfaces import EmbeddingProvider


class OllamaEmbeddingAdapter(EmbeddingProvider):
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "nomic-embed-text",
        dimension: int = 768,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dim = dimension

    async def embed_query(self, text: str) -> List[float]:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            resp.raise_for_status()
            return resp.json()["embedding"]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        results = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for t in texts:
                resp = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": t},
                )
                resp.raise_for_status()
                results.append(resp.json()["embedding"])
        return results

    @property
    def dimension(self) -> int:
        return self._dim
