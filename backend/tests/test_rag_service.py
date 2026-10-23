import hashlib
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.domain.models import SearchResult, ChatMessage, Citation, RawDocumentPage, DocumentChunk
from app.db.models import DBDocument, DBChatMessage
from app.services.rag_service import RAGService, build_prompt_with_sources
from app.services.ingestion_service import IngestionService


def test_build_prompt_with_sources_injects_numbered_citations():
    sources = [
        SearchResult("chk-1", "doc-1", "politicas.pdf", 5, "O prazo de SLA é 24h.", 0.95),
        SearchResult("chk-2", "doc-2", "manual.docx", 2, "Suporte disponível em horário comercial.", 0.88),
    ]
    prompt, citations = build_prompt_with_sources("Qual o prazo de SLA?", sources)

    assert "[FONTE 1] (Arquivo: politicas.pdf | Página: 5)" in prompt
    assert "[FONTE 2] (Arquivo: manual.docx | Página: 2)" in prompt
    assert len(citations) == 2
    assert citations[0].id == 1
    assert citations[0].filename == "politicas.pdf"
    assert citations[0].page_number == 5
    assert citations[0].score == 0.95
    assert citations[0].snippet == "O prazo de SLA é 24h."
    assert citations[1].id == 2
    assert citations[1].filename == "manual.docx"


def test_build_prompt_with_sources_empty():
    prompt, citations = build_prompt_with_sources("Qualquer pergunta?", [])
    assert "Nenhum documento relevante encontrado." in prompt
    assert "Qualquer pergunta?" in prompt
    assert citations == []


@pytest.mark.asyncio
async def test_rag_service_streams_sources_then_tokens():
    mock_retrieval = AsyncMock()
    mock_llm = AsyncMock()
    mock_session_factory = MagicMock()

    mock_retrieval.retrieve_context.return_value = [
        SearchResult("chk-1", "doc-1", "politicas.pdf", 5, "O prazo de SLA é 24h.", 0.95)
    ]

    async def sample_token_stream(*args, **kwargs):
        yield "O "
        yield "prazo "
        yield "é 24h."

    mock_llm.generate_stream.side_effect = sample_token_stream

    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_provider=mock_llm,
        session_factory=mock_session_factory,
    )

    events = []
    async for event in rag.stream_response(session_id="test-session", question="Qual SLA?"):
        events.append(event)

    assert len(events) == 5
    assert events[0]["type"] == "sources"
    assert len(events[0]["citations"]) == 1
    assert events[0]["citations"][0]["id"] == 1
    assert events[0]["citations"][0]["filename"] == "politicas.pdf"
    assert events[1]["type"] == "token"
    assert events[1]["delta"] == "O "
    assert events[2]["type"] == "token"
    assert events[2]["delta"] == "prazo "
    assert events[3]["type"] == "token"
    assert events[3]["delta"] == "é 24h."
    assert events[4]["type"] == "done"
    assert events[4]["session_id"] == "test-session"
    assert "message_id" in events[4]


@pytest.mark.asyncio
async def test_rag_service_loads_history_and_saves_messages():
    mock_retrieval = AsyncMock()
    mock_llm = AsyncMock()
    mock_session_factory = MagicMock()

    mock_retrieval.retrieve_context.return_value = []

    async def sample_stream(*args, **kwargs):
        yield "Resposta teste."

    mock_llm.generate_stream.side_effect = sample_stream

    # Setup session with previous messages
    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    session_uuid = uuid.uuid4()
    past_msg1 = DBChatMessage(id=uuid.uuid4(), session_id=session_uuid, role="user", content="Oi")
    past_msg2 = DBChatMessage(id=uuid.uuid4(), session_id=session_uuid, role="assistant", content="Olá!")

    mock_result = MagicMock()
    # In desc order, so latest first
    mock_result.scalars.return_value.all.return_value = [past_msg2, past_msg1]
    mock_session.execute.return_value = mock_result

    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_provider=mock_llm,
        session_factory=mock_session_factory,
    )

    events = [e async for e in rag.stream_response(session_id=str(session_uuid), question="Nova pergunta")]

    assert events[-1]["type"] == "done"
    assert mock_llm.generate_stream.called
    call_kwargs = mock_llm.generate_stream.call_args.kwargs
    history = call_kwargs["history"]
    # Reversing desc order should give chronological [past_msg1, past_msg2]
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[0].content == "Oi"
    assert history[1].role == "assistant"
    assert history[1].content == "Olá!"

    # Verify session.add_all and session.commit were called to persist new messages
    assert mock_session.add_all.called
    added_messages = mock_session.add_all.call_args[0][0]
    assert len(added_messages) == 2
    assert added_messages[0].role == "user"
    assert added_messages[0].content == "Nova pergunta"
    assert added_messages[1].role == "assistant"
    assert added_messages[1].content == "Resposta teste."
    assert mock_session.commit.called


@pytest.mark.asyncio
async def test_rag_service_without_session_factory():
    mock_retrieval = AsyncMock()
    mock_llm = AsyncMock()

    mock_retrieval.retrieve_context.return_value = []

    async def sample_stream(*args, **kwargs):
        yield "Sem DB"

    mock_llm.generate_stream.side_effect = sample_stream

    rag = RAGService(
        retrieval_service=mock_retrieval,
        llm_provider=mock_llm,
        session_factory=None,
    )

    events = [e async for e in rag.query_stream(session_id="session-123", question="Pergunta")]
    assert len(events) == 3
    assert events[0]["type"] == "sources"
    assert events[1]["type"] == "token"
    assert events[2]["type"] == "done"


@pytest.mark.asyncio
async def test_ingestion_service_ingest_file_success():
    mock_chunking = MagicMock()
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_session_factory = MagicMock()

    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    # Document does not exist yet
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # Mock chunking
    mock_chunking.split_pages.return_value = [
        DocumentChunk(document_id="doc-1", chunk_index=0, page_number=1, content="Parte 1"),
        DocumentChunk(document_id="doc-1", chunk_index=1, page_number=1, content="Parte 2"),
    ]
    # Mock embedding provider
    mock_embed.embed_documents.return_value = [[0.1] * 10, [0.2] * 10]

    service = IngestionService(
        chunking_service=mock_chunking,
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        session_factory=mock_session_factory,
    )

    file_bytes = b"Texto simples de documento para teste"
    doc = await service.ingest_file("teste.txt", file_bytes)

    assert doc.filename == "teste.txt"
    assert doc.file_type == "txt"
    assert doc.file_size_bytes == len(file_bytes)
    assert doc.file_hash == hashlib.sha256(file_bytes).hexdigest()

    assert mock_session.add.called
    assert mock_session.commit.called
    assert mock_chunking.split_pages.called
    assert mock_embed.embed_documents.called
    assert mock_vectorstore.add_chunks.called

    added_chunks = mock_vectorstore.add_chunks.call_args[0][0]
    assert len(added_chunks) == 2
    assert added_chunks[0].embedding == [0.1] * 10
    assert added_chunks[1].embedding == [0.2] * 10


@pytest.mark.asyncio
async def test_ingestion_service_idempotency_raises_error():
    mock_chunking = MagicMock()
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_session_factory = MagicMock()

    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session

    # Existing document found
    existing_doc = DBDocument(id=uuid.uuid4(), filename="antigo.pdf", file_hash="hash-existente")
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = existing_doc
    mock_session.execute.return_value = mock_result

    service = IngestionService(
        chunking_service=mock_chunking,
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        session_factory=mock_session_factory,
    )

    with pytest.raises(ValueError, match="já foi indexado previamente"):
        await service.ingest_file("novo.pdf", b"conteudo repetido")

    assert not mock_vectorstore.add_chunks.called


@pytest.mark.asyncio
async def test_ingestion_service_batch_embedding_batches_at_64():
    mock_chunking = MagicMock()
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_session_factory = MagicMock()

    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    # 70 chunks -> 2 batches (64 and 6)
    chunks = [
        DocumentChunk(document_id="doc-1", chunk_index=i, page_number=1, content=f"Chunk {i}")
        for i in range(70)
    ]
    mock_chunking.split_pages.return_value = chunks

    def mock_embed_docs(texts):
        return [[0.1] * 5 for _ in texts]

    mock_embed.embed_documents.side_effect = mock_embed_docs

    service = IngestionService(
        chunking_service=mock_chunking,
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        session_factory=mock_session_factory,
    )

    await service.ingest_file("relatorio.txt", b"Texto longo")

    assert mock_embed.embed_documents.call_count == 2
    first_call_len = len(mock_embed.embed_documents.call_args_list[0][0][0])
    second_call_len = len(mock_embed.embed_documents.call_args_list[1][0][0])
    assert first_call_len == 64
    assert second_call_len == 6
    assert all(c.embedding is not None for c in chunks)


@pytest.mark.asyncio
async def test_ingestion_service_cleanup_on_add_chunks_failure():
    mock_chunking = MagicMock()
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_session_factory = MagicMock()

    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    mock_chunking.split_pages.return_value = [
        DocumentChunk(document_id="doc-1", chunk_index=0, page_number=1, content="Texto teste")
    ]
    mock_embed.embed_documents.return_value = [[0.1] * 10]
    mock_vectorstore.add_chunks.side_effect = RuntimeError("Falha ao salvar no VectorStore")

    service = IngestionService(
        chunking_service=mock_chunking,
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        session_factory=mock_session_factory,
    )

    with pytest.raises(RuntimeError, match="Falha ao salvar no VectorStore"):
        await service.ingest_file("arquivo_com_erro.txt", b"Texto de teste")

    # Documento deve ter sido removido na sessão de limpeza
    assert mock_session.delete.called or mock_session.execute.call_count >= 2
    assert mock_session.commit.call_count >= 2


@pytest.mark.asyncio
async def test_ingestion_service_cleanup_on_embedding_failure():
    mock_chunking = MagicMock()
    mock_vectorstore = AsyncMock()
    mock_embed = AsyncMock()
    mock_session_factory = MagicMock()

    mock_session = AsyncMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_session.execute.return_value = mock_result

    mock_chunking.split_pages.return_value = [
        DocumentChunk(document_id="doc-1", chunk_index=0, page_number=1, content="Texto teste")
    ]
    mock_embed.embed_documents.side_effect = RuntimeError("Provedor de embeddings offline")

    service = IngestionService(
        chunking_service=mock_chunking,
        vector_store=mock_vectorstore,
        embedding_provider=mock_embed,
        session_factory=mock_session_factory,
    )

    with pytest.raises(RuntimeError, match="Provedor de embeddings offline"):
        await service.ingest_file("arquivo_falha_emb.txt", b"Texto de teste")

    # Vector store nem deve ser chamado, mas o cleanup deve ser executado
    assert not mock_vectorstore.add_chunks.called
    assert mock_session.delete.called or mock_session.execute.call_count >= 2
    assert mock_session.commit.call_count >= 2
