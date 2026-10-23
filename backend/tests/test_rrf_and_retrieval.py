import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.domain.models import SearchResult
from app.services.retrieval_service import RetrievalService, apply_rrf
from app.adapters.reranker.flashrank_adapter import FlashRankReranker


def test_apply_rrf_prioritizes_common_matches():
    r1 = SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.9, rank_dense=1, rank_sparse=1)
    r2 = SearchResult("chk-2", "doc-1", "doc.pdf", 2, "texto 2", 0.8, rank_dense=2, rank_sparse=None)

    scored = apply_rrf([r1, r2], k=60)
    # r1 está em ambos os rankings, logo seu score RRF deve ser maior
    assert scored[0].chunk_id == "chk-1"
    assert scored[0].score > scored[1].score


def test_apply_rrf_mathematics():
    r1 = SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.0, rank_dense=1, rank_sparse=2)
    r2 = SearchResult("chk-2", "doc-1", "doc.pdf", 2, "texto 2", 0.0, rank_dense=None, rank_sparse=3)
    r3 = SearchResult("chk-3", "doc-1", "doc.pdf", 3, "texto 3", 0.0, rank_dense=4, rank_sparse=None)
    r4 = SearchResult("chk-4", "doc-1", "doc.pdf", 4, "texto 4", 0.0, rank_dense=None, rank_sparse=None)

    scored = apply_rrf([r1, r2, r3, r4], k=60)

    # Verifica o cálculo matemático exato com k=60:
    # r1: 1 / (60 + 1) + 1 / (60 + 2) = 1/61 + 1/62
    expected_r1 = (1.0 / 61.0) + (1.0 / 62.0)
    # r2: 1 / (60 + 3) = 1/63
    expected_r2 = 1.0 / 63.0
    # r3: 1 / (60 + 4) = 1/64
    expected_r3 = 1.0 / 64.0
    # r4: 0.0
    expected_r4 = 0.0

    scored_dict = {item.chunk_id: item.score for item in scored}
    assert pytest.approx(scored_dict["chk-1"], abs=1e-6) == expected_r1
    assert pytest.approx(scored_dict["chk-2"], abs=1e-6) == expected_r2
    assert pytest.approx(scored_dict["chk-3"], abs=1e-6) == expected_r3
    assert pytest.approx(scored_dict["chk-4"], abs=1e-6) == expected_r4

    # Ordenação decrescente por score
    assert [item.chunk_id for item in scored] == ["chk-1", "chk-2", "chk-3", "chk-4"]


def test_apply_rrf_custom_k_and_empty():
    assert apply_rrf([]) == []

    r1 = SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.0, rank_dense=1, rank_sparse=None)
    scored = apply_rrf([r1], k=10)
    assert pytest.approx(scored[0].score, abs=1e-6) == 1.0 / (10 + 1)


@pytest.mark.asyncio
async def test_flashrank_adapter_rerank_with_mock_ranker():
    mock_ranker = MagicMock()
    # Simula resposta do flashrank
    mock_ranker.rerank.return_value = [
        {"id": "chk-2", "score": 0.88888, "text": "texto 2", "meta": SearchResult("chk-2", "doc-1", "doc.pdf", 2, "texto 2", 0.0)},
        {"id": "chk-1", "score": 0.55556, "text": "texto 1", "meta": SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.0)},
    ]

    reranker = FlashRankReranker(ranker=mock_ranker)
    candidates = [
        SearchResult("chk-1", "doc-1", "doc.pdf", 1, "texto 1", 0.0),
        SearchResult("chk-2", "doc-1", "doc.pdf", 2, "texto 2", 0.0),
    ]

    results = await reranker.rerank(query="consulta", candidates=candidates, top_k=2, threshold=0.30)

    assert mock_ranker.rerank.called
    assert len(results) == 2
    assert results[0].chunk_id == "chk-2"
    assert results[0].score == 0.8889  # arredondado para 4 casas decimais
    assert results[1].chunk_id == "chk-1"
    assert results[1].score == 0.5556


@pytest.mark.asyncio
async def test_flashrank_adapter_threshold_filtering():
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": "chk-high", "score": 0.95, "text": "alto", "meta": SearchResult("chk-high", "d", "f", 1, "alto", 0.0)},
        {"id": "chk-edge", "score": 0.30, "text": "limiar", "meta": SearchResult("chk-edge", "d", "f", 2, "limiar", 0.0)},
        {"id": "chk-low", "score": 0.299, "text": "baixo", "meta": SearchResult("chk-low", "d", "f", 3, "baixo", 0.0)},
        {"id": "chk-irrelevant", "score": 0.05, "text": "irrelevante", "meta": SearchResult("chk-irrelevant", "d", "f", 4, "irrelevante", 0.0)},
    ]

    reranker = FlashRankReranker(ranker=mock_ranker)
    candidates = [
        SearchResult("chk-high", "d", "f", 1, "alto", 0.0),
        SearchResult("chk-edge", "d", "f", 2, "limiar", 0.0),
        SearchResult("chk-low", "d", "f", 3, "baixo", 0.0),
        SearchResult("chk-irrelevant", "d", "f", 4, "irrelevante", 0.0),
    ]

    results = await reranker.rerank(query="consulta", candidates=candidates, top_k=4, threshold=0.30)

    # Apenas chk-high e chk-edge atingem o threshold >= 0.30
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["chk-high", "chk-edge"]


@pytest.mark.asyncio
async def test_flashrank_adapter_top_k_limiting():
    mock_ranker = MagicMock()
    mock_ranker.rerank.return_value = [
        {"id": f"chk-{i}", "score": 0.9 - (i * 0.05), "text": f"t-{i}", "meta": SearchResult(f"chk-{i}", "d", "f", i, f"t-{i}", 0.0)}
        for i in range(10)
    ]

    reranker = FlashRankReranker(ranker=mock_ranker)
    candidates = [
        SearchResult(f"chk-{i}", "d", "f", i, f"t-{i}", 0.0)
        for i in range(10)
    ]

    results = await reranker.rerank(query="consulta", candidates=candidates, top_k=3, threshold=0.30)
    assert len(results) == 3


@pytest.mark.asyncio
async def test_flashrank_adapter_empty_candidates():
    mock_ranker = MagicMock()
    reranker = FlashRankReranker(ranker=mock_ranker)
    results = await reranker.rerank(query="consulta", candidates=[])
    assert results == []
    assert not mock_ranker.rerank.called


def test_flashrank_adapter_init_without_ranker():
    with patch("app.adapters.reranker.flashrank_adapter.Ranker") as mock_ranker_cls:
        reranker = FlashRankReranker(model_name="test-model", cache_dir="/tmp/test_cache")
        mock_ranker_cls.assert_called_once_with(model_name="test-model", cache_dir="/tmp/test_cache")
        assert reranker.ranker == mock_ranker_cls.return_value


@pytest.mark.asyncio
async def test_retrieval_service_flow():
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_rerank = AsyncMock()

    mock_embed.embed_query.return_value = [0.1] * 384
    mock_vectorstore.hybrid_search.return_value = [
        SearchResult("chk-1", "doc-1", "doc.pdf", 1, "trecho técnico relevante", 0.9, rank_dense=1, rank_sparse=2)
    ]
    mock_rerank.rerank.return_value = [
        SearchResult("chk-1", "doc-1", "doc.pdf", 1, "trecho técnico relevante", 0.85)
    ]

    service = RetrievalService(
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        reranker=mock_rerank,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=4,
        threshold=0.30
    )

    results = await service.retrieve_context("pergunta de teste")
    assert len(results) == 1
    assert results[0].score == 0.85
    assert mock_rerank.rerank.called


@pytest.mark.asyncio
async def test_retrieval_service_empty_query():
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_rerank = AsyncMock()

    service = RetrievalService(
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        reranker=mock_rerank,
    )

    results = await service.retrieve_context("   ")
    assert results == []
    assert not mock_embed.embed_query.called
    assert not mock_vectorstore.hybrid_search.called
    assert not mock_rerank.rerank.called


@pytest.mark.asyncio
async def test_retrieval_service_empty_hybrid_results():
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_rerank = AsyncMock()

    mock_embed.embed_query.return_value = [0.1] * 384
    mock_vectorstore.hybrid_search.return_value = []

    service = RetrievalService(
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        reranker=mock_rerank,
    )

    results = await service.retrieve_context("pergunta sem resultados")
    assert results == []
    assert not mock_rerank.rerank.called


@pytest.mark.asyncio
async def test_retrieval_service_limits_candidates_to_reranker():
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_rerank = AsyncMock()

    mock_embed.embed_query.return_value = [0.1] * 384
    # Gera 25 candidatos da busca híbrida
    raw_results = [
        SearchResult(f"chk-{i}", "doc-1", "doc.pdf", i, f"texto {i}", 0.5, rank_dense=i + 1, rank_sparse=i + 1)
        for i in range(25)
    ]
    mock_vectorstore.hybrid_search.return_value = raw_results
    mock_rerank.rerank.return_value = [
        SearchResult("chk-0", "doc-1", "doc.pdf", 0, "texto 0", 0.95)
    ]

    service = RetrievalService(
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        reranker=mock_rerank,
        dense_top_k=20,
        sparse_top_k=20,
        final_top_k=4,
        threshold=0.30,
    )

    results = await service.retrieve_context("query")
    assert len(results) == 1

    # Verifica se o reranker recebeu no máximo os top 15 candidatos fundidos por RRF
    call_args = mock_rerank.rerank.call_args
    assert len(call_args.kwargs["candidates"]) == 15
    assert call_args.kwargs["top_k"] == 4
    assert call_args.kwargs["threshold"] == 0.30
