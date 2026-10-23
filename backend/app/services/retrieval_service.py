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
        threshold: float = 0.30,
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.reranker = reranker
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self.final_top_k = final_top_k
        self.threshold = threshold

    async def retrieve_context(self, query: str) -> List[SearchResult]:
        if not query or not query.strip():
            return []

        # 1. Gera embedding da consulta
        query_vector = await self.embedding_provider.embed_query(query)

        # 2. Executa busca híbrida
        raw_candidates = await self.vector_store.hybrid_search(
            query=query,
            query_vector=query_vector,
            dense_top_k=self.dense_top_k,
            sparse_top_k=self.sparse_top_k,
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
            threshold=self.threshold,
        )

        return reranked
