import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone
import uuid
import json

from app.main import app
from app.db.models import DBDocument, DBChatSession, DBChatMessage
from app.api.schemas import (
    DocumentResponse,
    ChatSessionCreate,
    ChatSessionResponse,
    CitationSchema,
    ChatMessageResponse,
    ChatPromptRequest,
)


@pytest.mark.asyncio
async def test_health_check_endpoint():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "llm_provider" in data
        assert "embedding_provider" in data
        assert "reranker_threshold" in data


@pytest.mark.asyncio
async def test_upload_document_success():
    doc_id = uuid.uuid4()
    mock_doc = DBDocument(
        id=doc_id,
        filename="test_manual.pdf",
        file_hash="hash123",
        file_type="pdf",
        file_size_bytes=1024,
        total_pages=5,
        created_at=datetime.now(timezone.utc),
    )

    with patch("app.api.routes.documents.get_ingestion_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.ingest_file = AsyncMock(return_value=mock_doc)
        mock_get_svc.return_value = mock_svc

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("test_manual.pdf", b"conteudo falso do pdf", "application/pdf")}
            response = await client.post("/api/v1/documents/upload", files=files)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(doc_id)
            assert data["filename"] == "test_manual.pdf"
            assert data["file_type"] == "pdf"
            assert data["file_size_bytes"] == 1024
            assert data["total_pages"] == 5
            assert "created_at" in data
            mock_svc.ingest_file.assert_awaited_once_with(
                filename="test_manual.pdf",
                file_bytes=b"conteudo falso do pdf"
            )


@pytest.mark.asyncio
async def test_upload_document_empty_file():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("empty.pdf", b"", "application/pdf")}
        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 400
        assert "vazio" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_upload_document_duplicate_conflict_409():
    with patch("app.api.routes.documents.get_ingestion_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.ingest_file = AsyncMock(
            side_effect=ValueError("O documento 'doc.pdf' já foi indexado previamente.")
        )
        mock_get_svc.return_value = mock_svc

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("doc.pdf", b"conteudo", "application/pdf")}
            response = await client.post("/api/v1/documents/upload", files=files)
            assert response.status_code == 409
            assert "já foi indexado" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_document_parse_error_400():
    with patch("app.api.routes.documents.get_ingestion_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.ingest_file = AsyncMock(side_effect=ValueError("Formato invalido ou corrompido"))
        mock_get_svc.return_value = mock_svc

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            files = {"file": ("corrupt.pdf", b"corrompido", "application/pdf")}
            response = await client.post("/api/v1/documents/upload", files=files)
            assert response.status_code == 400
            assert "corrompido" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_documents():
    doc1 = DBDocument(
        id=uuid.uuid4(),
        filename="doc1.pdf",
        file_hash="h1",
        file_type="pdf",
        file_size_bytes=500,
        total_pages=2,
        created_at=datetime.now(timezone.utc),
    )
    doc2 = DBDocument(
        id=uuid.uuid4(),
        filename="doc2.txt",
        file_hash="h2",
        file_type="txt",
        file_size_bytes=300,
        total_pages=1,
        created_at=datetime.now(timezone.utc),
    )

    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [doc1, doc2]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    with patch("app.api.routes.documents.AsyncSessionLocal", return_value=mock_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/documents")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["filename"] == "doc1.pdf"
            assert data[1]["filename"] == "doc2.txt"


@pytest.mark.asyncio
async def test_delete_document():
    doc_id = str(uuid.uuid4())
    with patch("app.api.routes.documents.PGVectorStore") as mock_vstore_cls:
        mock_vstore = MagicMock()
        mock_vstore.delete_document = AsyncMock(return_value=None)
        mock_vstore_cls.return_value = mock_vstore

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete(f"/api/v1/documents/{doc_id}")
            assert response.status_code == 200
            data = response.json()
            assert data["document_id"] == doc_id
            assert "removido" in data["message"].lower()
            mock_vstore.delete_document.assert_awaited_once_with(doc_id)


@pytest.mark.asyncio
async def test_create_chat_session():
    sess_id = uuid.uuid4()
    mock_session = AsyncMock()
    
    async def fake_refresh(obj):
        obj.id = sess_id
        obj.created_at = datetime.now(timezone.utc)
        obj.updated_at = datetime.now(timezone.utc)

    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock(side_effect=fake_refresh)
    mock_session.add = MagicMock()

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    with patch("app.api.routes.chat.AsyncSessionLocal", return_value=mock_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/api/v1/chat/sessions", json={"title": "Sessão de Suporte"})
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == str(sess_id)
            assert data["title"] == "Sessão de Suporte"
            assert "created_at" in data


@pytest.mark.asyncio
async def test_list_chat_sessions():
    s1 = DBChatSession(
        id=uuid.uuid4(),
        title="Conversa 1",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    s2 = DBChatSession(
        id=uuid.uuid4(),
        title="Conversa 2",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [s1, s2]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    with patch("app.api.routes.chat.AsyncSessionLocal", return_value=mock_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/chat/sessions")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["title"] == "Conversa 1"
            assert data[1]["title"] == "Conversa 2"


@pytest.mark.asyncio
async def test_get_session_messages():
    sess_id = uuid.uuid4()
    m1 = DBChatMessage(
        id=uuid.uuid4(),
        session_id=sess_id,
        role="user",
        content="Olá",
        citations=[],
        created_at=datetime.now(timezone.utc),
    )
    m2 = DBChatMessage(
        id=uuid.uuid4(),
        session_id=sess_id,
        role="assistant",
        content="Como posso ajudar?",
        citations=[
            {
                "id": 1,
                "filename": "doc.pdf",
                "page_number": 1,
                "score": 0.95,
                "snippet": "trecho do documento",
            }
        ],
        created_at=datetime.now(timezone.utc),
    )

    mock_session = AsyncMock()
    mock_scalars = MagicMock()
    mock_scalars.all.return_value = [m1, m2]
    mock_result = MagicMock()
    mock_result.scalars.return_value = mock_scalars
    mock_session.execute = AsyncMock(return_value=mock_result)

    mock_ctx = AsyncMock()
    mock_ctx.__aenter__.return_value = mock_session
    mock_ctx.__aexit__.return_value = None

    with patch("app.api.routes.chat.AsyncSessionLocal", return_value=mock_ctx):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/v1/chat/sessions/{sess_id}/messages")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert data[0]["role"] == "user"
            assert data[0]["content"] == "Olá"
            assert data[1]["role"] == "assistant"
            assert data[1]["content"] == "Como posso ajudar?"
            assert len(data[1]["citations"]) == 1
            assert data[1]["citations"][0]["filename"] == "doc.pdf"


@pytest.mark.asyncio
async def test_chat_stream_sse_success():
    sess_id = str(uuid.uuid4())

    async def fake_stream_response(session_id, question):
        yield {
            "type": "sources",
            "citations": [
                {
                    "id": 1,
                    "filename": "manual.pdf",
                    "page_number": 3,
                    "score": 0.88,
                    "snippet": "texto explicativo",
                }
            ],
        }
        yield {"type": "token", "delta": "Olá, "}
        yield {"type": "token", "delta": "mundo!"}
        yield {"type": "done", "message_id": "msg-123", "session_id": session_id}

    with patch("app.api.routes.chat.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.stream_response = fake_stream_response
        mock_get_rag.return_value = mock_rag

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/chat/{sess_id}/stream",
                json={"question": "Olá tudo bem?"},
            )
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            lines = response.text.strip().split("\n\n")
            events = []
            for block in lines:
                if not block.strip():
                    continue
                parts = block.strip().split("\n")
                event_type = parts[0].replace("event: ", "").strip()
                data_val = json.loads(parts[1].replace("data: ", "").strip())
                events.append((event_type, data_val))

            assert len(events) == 4
            assert events[0][0] == "sources"
            assert events[0][1]["citations"][0]["filename"] == "manual.pdf"
            assert events[1][0] == "token"
            assert events[1][1]["delta"] == "Olá, "
            assert events[2][0] == "token"
            assert events[2][1]["delta"] == "mundo!"
            assert events[3][0] == "done"
            assert events[3][1]["message_id"] == "msg-123"


@pytest.mark.asyncio
async def test_chat_stream_sse_error_handling():
    sess_id = str(uuid.uuid4())

    async def failing_stream(session_id, question):
        if False:
            yield
        raise RuntimeError("Falha de conexão com o LLM")

    with patch("app.api.routes.chat.get_rag_service") as mock_get_rag:
        mock_rag = MagicMock()
        mock_rag.stream_response = failing_stream
        mock_get_rag.return_value = mock_rag

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/chat/{sess_id}/stream",
                json={"question": "Pergunta que falha"},
            )
            assert response.status_code == 200
            assert "event: error" in response.text
            assert "Falha de conexão com o LLM" in response.text


@pytest.mark.asyncio
async def test_schemas_instantiation():
    doc_id = uuid.uuid4()
    doc = DocumentResponse(
        id=doc_id,
        filename="teste.txt",
        file_type="txt",
        file_size_bytes=100,
        total_pages=1,
        created_at=datetime.now(timezone.utc),
    )
    assert doc.id == doc_id

    chat_req = ChatPromptRequest(question="Pergunta?")
    assert chat_req.question == "Pergunta?"

    chat_sess_create = ChatSessionCreate()
    assert chat_sess_create.title == "Nova Conversa"


@pytest.mark.asyncio
async def test_upload_document_no_filename():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        files = {"file": ("   ", b"algum conteudo", "application/pdf")}
        response = await client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 400
        assert "sem nome" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_cors_headers():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        # Com allow_credentials=True, o middleware CORS reflete a origem
        assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_lifespan_events():
    from app.main import lifespan
    
    with patch("app.main.init_db", new_callable=AsyncMock) as mock_init:
        async with lifespan(app):
            pass
        mock_init.assert_awaited_once()

    with patch("app.main.init_db", side_effect=Exception("DB offline")):
        async with lifespan(app):
            pass  # Should not raise exception

