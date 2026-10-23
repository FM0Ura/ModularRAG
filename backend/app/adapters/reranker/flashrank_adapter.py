from typing import List, Optional, Any
from flashrank import Ranker, RerankRequest
from app.domain.interfaces import Reranker
from app.domain.models import SearchResult


class FlashRankReranker(Reranker):
    def __init__(
        self,
        model_name: str = "ms-marco-MiniLM-L-12-v2",
        cache_dir: str = "/tmp/flashrank",
        ranker: Optional[Any] = None,
    ):
        if ranker is not None:
            self.ranker = ranker
        else:
            self.ranker = Ranker(model_name=model_name, cache_dir=cache_dir)

    async def rerank(
        self,
        query: str,
        candidates: List[SearchResult],
        top_k: int = 4,
        threshold: float = 0.30,
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
