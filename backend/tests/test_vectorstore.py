import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models import DocumentChunk
from app.adapters.vectorstore.pgvector_store import PGVectorStore, _to_uuid
from app.db.models import (
    DBDocument,
    DBDocumentChunk,
    DBChatSession,
    DBChatMessage,
)
from app.db.session import init_db


@pytest.mark.asyncio
async def test_vector_store_add_chunks_mocked():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    session_factory.return_value.__aenter__.return_value = mock_session

    store = PGVectorStore(session_factory=session_factory)
    chunk = DocumentChunk(
        document_id="doc-123",
        chunk_index=0,
        page_number=1,
        content="Conteúdo do chunk teste",
        embedding=[0.1] * 384,
    )

    await store.add_chunks([chunk])
    assert mock_session.execute.called or mock_session.add_all.called
    assert mock_session.commit.called

    # Verifica os objetos criados e passados para add_all
    args, _ = mock_session.add_all.call_args
    added_chunks = args[0]
    assert len(added_chunks) == 1
    assert added_chunks[0].chunk_index == 0
    assert added_chunks[0].page_number == 1
    assert added_chunks[0].content == "Conteúdo do chunk teste"
    assert added_chunks[0].embedding == [0.1] * 384
    assert isinstance(added_chunks[0].id, uuid.UUID)
    assert isinstance(added_chunks[0].document_id, uuid.UUID)


@pytest.mark.asyncio
async def test_vector_store_add_chunks_with_existing_uuid():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.add_all = MagicMock()
    session_factory.return_value.__aenter__.return_value = mock_session

    doc_uuid = str(uuid.uuid4())
    chunk_uuid = str(uuid.uuid4())

    store = PGVectorStore(session_factory=session_factory)
    chunk = DocumentChunk(
        id=chunk_uuid,
        document_id=doc_uuid,
        chunk_index=1,
        page_number=2,
        content="Outro chunk com UUID válido",
        embedding=[0.2] * 384,
    )

    await store.add_chunks([chunk])
    args, _ = mock_session.add_all.call_args
    added_chunk = args[0][0]
    assert added_chunk.id == uuid.UUID(chunk_uuid)
    assert added_chunk.document_id == uuid.UUID(doc_uuid)


@pytest.mark.asyncio
async def test_vector_store_hybrid_search():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session

    chunk_id_1 = str(uuid.uuid4())
    chunk_id_2 = str(uuid.uuid4())
    chunk_id_3 = str(uuid.uuid4())
    doc_id = str(uuid.uuid4())

    # Mock resultado da busca densa: chunk_1 (rank 1), chunk_2 (rank 2)
    dense_result = MagicMock()
    dense_result.fetchall.return_value = [
        (chunk_id_1, doc_id, "doc1.pdf", 1, "Conteúdo 1", 0.95),
        (chunk_id_2, doc_id, "doc1.pdf", 2, "Conteúdo 2", 0.85),
    ]

    # Mock resultado da busca esparsa: chunk_2 (rank 1), chunk_3 (rank 2)
    sparse_result = MagicMock()
    sparse_result.fetchall.return_value = [
        (chunk_id_2, doc_id, "doc1.pdf", 2, "Conteúdo 2", 0.70),
        (chunk_id_3, doc_id, "doc1.pdf", 3, "Conteúdo 3", 0.60),
    ]

    mock_session.execute.side_effect = [dense_result, sparse_result]

    store = PGVectorStore(session_factory=session_factory)
    results = await store.hybrid_search(
        query="teste de busca híbrida",
        query_vector=[0.1] * 384,
        dense_top_k=5,
        sparse_top_k=5,
    )

    assert len(results) == 3
    results_by_id = {r.chunk_id: r for r in results}

    # chunk 1: apenas na busca densa (rank_dense=1, rank_sparse=None)
    assert results_by_id[chunk_id_1].rank_dense == 1
    assert results_by_id[chunk_id_1].rank_sparse is None
    assert results_by_id[chunk_id_1].score == 0.95

    # chunk 2: presente em ambas (rank_dense=2, rank_sparse=1)
    assert results_by_id[chunk_id_2].rank_dense == 2
    assert results_by_id[chunk_id_2].rank_sparse == 1

    # chunk 3: apenas na busca esparsa (rank_dense=None, rank_sparse=2)
    assert results_by_id[chunk_id_3].rank_dense is None
    assert results_by_id[chunk_id_3].rank_sparse == 2
    assert results_by_id[chunk_id_3].score == 0.60

    assert mock_session.execute.call_count == 2


@pytest.mark.asyncio
async def test_hybrid_search_query_compilation_bindparams():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session

    dense_result = MagicMock()
    dense_result.fetchall.return_value = []
    sparse_result = MagicMock()
    sparse_result.fetchall.return_value = []
    mock_session.execute.side_effect = [dense_result, sparse_result]

    store = PGVectorStore(session_factory=session_factory)
    await store.hybrid_search(
        query="teste bindparams",
        query_vector=[0.1] * 384,
        dense_top_k=10,
        sparse_top_k=10,
    )

    assert mock_session.execute.call_count == 2
    dense_call, sparse_call = mock_session.execute.call_args_list

    dense_stmt = dense_call.args[0]
    dense_params = dense_call.args[1]
    compiled_dense_params = set(dense_stmt.compile().params.keys())
    assert "qvec" in compiled_dense_params
    assert "dense_k" in compiled_dense_params
    assert dense_params["dense_k"] == 10
    assert "CAST(:qvec AS vector)" in str(dense_stmt)
    assert "c.embedding IS NOT NULL" in str(dense_stmt)

    sparse_stmt = sparse_call.args[0]
    sparse_params = sparse_call.args[1]
    compiled_sparse_params = set(sparse_stmt.compile().params.keys())
    assert "qtext" in compiled_sparse_params
    assert "sparse_k" in compiled_sparse_params
    assert sparse_params["qtext"] == "teste bindparams"
    assert sparse_params["sparse_k"] == 10


@pytest.mark.asyncio
async def test_vector_store_delete_document():
    session_factory = MagicMock()
    mock_session = AsyncMock()
    session_factory.return_value.__aenter__.return_value = mock_session

    store = PGVectorStore(session_factory=session_factory)
    doc_id = str(uuid.uuid4())

    await store.delete_document(doc_id)
    assert mock_session.execute.called
    assert mock_session.commit.called


def test_to_uuid_utility():
    # UUID instance
    u = uuid.uuid4()
    assert _to_uuid(u) == u

    # Valid UUID string
    assert _to_uuid(str(u)) == u

    # Non-UUID string produces stable uuid5
    u1 = _to_uuid("doc-123")
    u2 = _to_uuid("doc-123")
    assert isinstance(u1, uuid.UUID)
    assert u1 == u2
    assert u1 != _to_uuid("doc-456")


def test_models_metadata_definitions():
    assert DBDocument.__tablename__ == "documents"
    assert DBDocumentChunk.__tablename__ == "document_chunks"
    assert DBChatSession.__tablename__ == "chat_sessions"
    assert DBChatMessage.__tablename__ == "chat_messages"

    # Checa colunas esperadas
    doc_cols = {c.name for c in DBDocument.__table__.columns}
    assert {"id", "filename", "file_hash", "file_type", "file_size_bytes", "total_pages", "created_at"}.issubset(doc_cols)

    chunk_cols = {c.name for c in DBDocumentChunk.__table__.columns}
    assert {"id", "document_id", "chunk_index", "page_number", "content", "embedding", "created_at"}.issubset(chunk_cols)

    chat_session_cols = {c.name for c in DBChatSession.__table__.columns}
    assert {"id", "title", "created_at", "updated_at"}.issubset(chat_session_cols)

    chat_msg_cols = {c.name for c in DBChatMessage.__table__.columns}
    assert {"id", "session_id", "role", "content", "citations", "created_at"}.issubset(chat_msg_cols)


@pytest.mark.asyncio
async def test_init_db():
    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_engine.begin.return_value.__aenter__.return_value = mock_conn

    await init_db(target_engine=mock_engine)

    assert mock_conn.execute.called
    assert mock_conn.run_sync.called
