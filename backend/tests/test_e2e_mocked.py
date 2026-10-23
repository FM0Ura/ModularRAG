import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.db.models import DBDocument, DBChatSession, DBChatMessage


@pytest.mark.asyncio
async def test_complete_upload_and_chat_e2e():
    """
    End-to-End simulation testing the entire pipeline:
    1. Health check verification
    2. Document upload and ingestion
    3. Document listing
    4. Chat session creation
    5. Session message history verification (initial empty)
    6. SSE streaming chat with citations and token generation
    7. Post-chat message history verification
    8. Document deletion
    """
    doc_id = uuid.uuid4()
    session_id = uuid.uuid4()
    msg_user_id = uuid.uuid4()
    msg_asst_id = uuid.uuid4()

    mock_db_doc = DBDocument(
        id=doc_id,
        filename="manual_arquitetura.pdf",
        file_hash="e2e_hash_98765",
        file_type="pdf",
        file_size_bytes=2048,
        total_pages=4,
        created_at=datetime.now(timezone.utc),
    )

    mock_chat_session = DBChatSession(
        id=session_id,
        title="Sessão de Teste E2E",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )

    mock_user_message = DBChatMessage(
        id=msg_user_id,
        session_id=session_id,
        role="user",
        content="Como funciona a arquitetura do sistema?",
        citations=[],
        created_at=datetime.now(timezone.utc),
    )

    mock_assistant_message = DBChatMessage(
        id=msg_asst_id,
        session_id=session_id,
        role="assistant",
        content="A arquitetura é modular seguindo Ports & Adapters [1].",
        citations=[
            {
                "id": 1,
                "filename": "manual_arquitetura.pdf",
                "page_number": 2,
                "score": 0.95,
                "snippet": "A arquitetura do sistema é estritamente desacoplada seguindo Ports & Adapters.",
            }
        ],
        created_at=datetime.now(timezone.utc),
    )

    # 1. Mock DB Session Local
    mock_db_session = AsyncMock()
    mock_db_session.add = MagicMock()
    mock_db_ctx = AsyncMock()
    mock_db_ctx.__aenter__.return_value = mock_db_session
    mock_db_ctx.__aexit__.return_value = None

    # Track messages state
    session_messages = []

    async def mock_execute(query, *args, **kwargs):
        query_str = str(query)
        result = MagicMock()
        scalars = MagicMock()

        if "FROM documents" in query_str:
            scalars.all.return_value = [mock_db_doc]
        elif "FROM chat_sessions" in query_str:
            scalars.all.return_value = [mock_chat_session]
        elif "FROM chat_messages" in query_str:
            scalars.all.return_value = list(session_messages)
        else:
            scalars.all.return_value = []
        result.scalars.return_value = scalars
        return result

    mock_db_session.execute = AsyncMock(side_effect=mock_execute)

    async def mock_refresh(instance):
        if isinstance(instance, DBChatSession) and not getattr(instance, "id", None):
            instance.id = session_id
            instance.created_at = datetime.now(timezone.utc)
            instance.updated_at = datetime.now(timezone.utc)

    mock_db_session.refresh = AsyncMock(side_effect=mock_refresh)

    # 2. Mock Ingestion Service
    mock_ingestion_svc = MagicMock()
    mock_ingestion_svc.ingest_file = AsyncMock(return_value=mock_db_doc)

    # 3. Mock RAG Service for SSE Stream
    mock_rag_svc = MagicMock()

    async def fake_stream_response(sess_id: str, question: str):
        # Emit citations
        yield {
            "type": "sources",
            "citations": [
                {
                    "id": 1,
                    "filename": "manual_arquitetura.pdf",
                    "page_number": 2,
                    "score": 0.95,
                    "snippet": "A arquitetura do sistema é estritamente desacoplada seguindo Ports & Adapters.",
                }
            ],
        }
        # Emit tokens
        yield {"type": "token", "delta": "A arquitetura "}
        yield {"type": "token", "delta": "é modular seguindo Ports & Adapters [1]."}
        # Emit done
        yield {"type": "done", "message_id": str(msg_asst_id), "session_id": str(sess_id)}

        # In a real run, RAGService persists the messages in the DB
        session_messages.append(mock_user_message)
        session_messages.append(mock_assistant_message)

    mock_rag_svc.stream_response = fake_stream_response

    # 4. Mock VectorStore for document deletion
    mock_vstore = MagicMock()
    mock_vstore.delete_document = AsyncMock()

    with patch("app.api.routes.documents.AsyncSessionLocal", return_value=mock_db_ctx), \
         patch("app.api.routes.chat.AsyncSessionLocal", return_value=mock_db_ctx), \
         patch("app.api.routes.documents.get_ingestion_service", return_value=mock_ingestion_svc), \
         patch("app.api.routes.chat.get_rag_service", return_value=mock_rag_svc), \
         patch("app.api.routes.documents.PGVectorStore", return_value=mock_vstore):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # Step 1: Health check
            health_resp = await client.get("/api/v1/health")
            assert health_resp.status_code == 200
            health_data = health_resp.json()
            assert health_data["status"] == "healthy"
            assert "llm_provider" in health_data
            assert "embedding_provider" in health_data
            assert "reranker_threshold" in health_data

            # Step 2: Ingest Document
            upload_files = {
                "file": ("manual_arquitetura.pdf", b"%PDF-1.4 Fake PDF Content for E2E", "application/pdf")
            }
            upload_resp = await client.post("/api/v1/documents/upload", files=upload_files)
            assert upload_resp.status_code == 200
            upload_data = upload_resp.json()
            assert upload_data["id"] == str(doc_id)
            assert upload_data["filename"] == "manual_arquitetura.pdf"
            assert upload_data["file_type"] == "pdf"
            assert upload_data["file_size_bytes"] == 2048
            assert upload_data["total_pages"] == 4
            mock_ingestion_svc.ingest_file.assert_awaited_once_with(
                filename="manual_arquitetura.pdf",
                file_bytes=b"%PDF-1.4 Fake PDF Content for E2E",
            )

            # Step 3: List Documents
            docs_list_resp = await client.get("/api/v1/documents")
            assert docs_list_resp.status_code == 200
            docs_list = docs_list_resp.json()
            assert len(docs_list) == 1
            assert docs_list[0]["id"] == str(doc_id)
            assert docs_list[0]["filename"] == "manual_arquitetura.pdf"

            # Step 4: Create Chat Session
            create_sess_resp = await client.post(
                "/api/v1/chat/sessions", json={"title": "Sessão de Teste E2E"}
            )
            assert create_sess_resp.status_code == 200
            sess_data = create_sess_resp.json()
            assert sess_data["title"] == "Sessão de Teste E2E"
            created_session_id = sess_data["id"]

            # Step 5: Check initial empty messages
            msgs_resp_init = await client.get(f"/api/v1/chat/sessions/{created_session_id}/messages")
            assert msgs_resp_init.status_code == 200
            assert len(msgs_resp_init.json()) == 0

            # Step 6: Stream Chat via Server-Sent Events (SSE)
            chat_stream_resp = await client.post(
                f"/api/v1/chat/{created_session_id}/stream",
                json={"question": "Como funciona a arquitetura do sistema?"},
            )
            assert chat_stream_resp.status_code == 200
            assert "text/event-stream" in chat_stream_resp.headers["content-type"]

            # Parse SSE events
            sse_blocks = chat_stream_resp.text.strip().split("\n\n")
            parsed_events = []
            for block in sse_blocks:
                if not block.strip():
                    continue
                lines = block.strip().split("\n")
                evt_type = lines[0].replace("event: ", "").strip()
                evt_payload = json.loads(lines[1].replace("data: ", "").strip())
                parsed_events.append((evt_type, evt_payload))

            assert len(parsed_events) == 4
            # Event 1: Sources / Citations
            assert parsed_events[0][0] == "sources"
            citations = parsed_events[0][1]["citations"]
            assert len(citations) == 1
            assert citations[0]["id"] == 1
            assert citations[0]["filename"] == "manual_arquitetura.pdf"
            assert citations[0]["page_number"] == 2
            assert citations[0]["score"] == 0.95
            assert "Ports & Adapters" in citations[0]["snippet"]

            # Event 2 & 3: Token streaming
            assert parsed_events[1][0] == "token"
            assert parsed_events[1][1]["delta"] == "A arquitetura "
            assert parsed_events[2][0] == "token"
            assert parsed_events[2][1]["delta"] == "é modular seguindo Ports & Adapters [1]."

            # Event 4: Completion Done event
            assert parsed_events[3][0] == "done"
            assert parsed_events[3][1]["message_id"] == str(msg_asst_id)
            assert parsed_events[3][1]["session_id"] == str(created_session_id)

            # Step 7: Post-chat history verification
            msgs_resp_post = await client.get(f"/api/v1/chat/sessions/{created_session_id}/messages")
            assert msgs_resp_post.status_code == 200
            post_msgs = msgs_resp_post.json()
            assert len(post_msgs) == 2
            assert post_msgs[0]["role"] == "user"
            assert post_msgs[0]["content"] == "Como funciona a arquitetura do sistema?"
            assert post_msgs[1]["role"] == "assistant"
            assert "Ports & Adapters [1]" in post_msgs[1]["content"]
            assert len(post_msgs[1]["citations"]) == 1
            assert post_msgs[1]["citations"][0]["filename"] == "manual_arquitetura.pdf"

            # Step 8: Document Deletion
            del_resp = await client.delete(f"/api/v1/documents/{doc_id}")
            assert del_resp.status_code == 200
            del_data = del_resp.json()
            assert del_data["document_id"] == str(doc_id)
            mock_vstore.delete_document.assert_awaited_once_with(str(doc_id))


@pytest.mark.asyncio
async def test_e2e_edge_cases_and_error_handling():
    """
    End-to-End edge cases:
    - Empty file rejected with 400
    - Duplicate file upload returns 409
    - Streaming error handled gracefully as SSE error event
    """
    mock_db_session = AsyncMock()
    mock_db_ctx = AsyncMock()
    mock_db_ctx.__aenter__.return_value = mock_db_session
    mock_db_ctx.__aexit__.return_value = None

    # Ingestion mock throwing duplicate error
    mock_ingestion_svc = MagicMock()
    mock_ingestion_svc.ingest_file = AsyncMock(
        side_effect=ValueError("O documento 'duplicate.pdf' já foi indexado previamente.")
    )

    # RAG service mock failing during stream
    mock_rag_svc = MagicMock()

    async def failing_stream(sess_id: str, question: str):
        if False:
            yield
        raise RuntimeError("Provedor LLM fora do ar (Connection refused)")

    mock_rag_svc.stream_response = failing_stream

    with patch("app.api.routes.documents.AsyncSessionLocal", return_value=mock_db_ctx), \
         patch("app.api.routes.chat.AsyncSessionLocal", return_value=mock_db_ctx), \
         patch("app.api.routes.documents.get_ingestion_service", return_value=mock_ingestion_svc), \
         patch("app.api.routes.chat.get_rag_service", return_value=mock_rag_svc):

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # 1. Empty file rejection
            empty_resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("empty.pdf", b"", "application/pdf")},
            )
            assert empty_resp.status_code == 400
            assert "vazio" in empty_resp.json()["detail"].lower()

            # 2. Duplicate file conflict
            dup_resp = await client.post(
                "/api/v1/documents/upload",
                files={"file": ("duplicate.pdf", b"dup content", "application/pdf")},
            )
            assert dup_resp.status_code == 409
            assert "já foi indexado" in dup_resp.json()["detail"].lower()

            # 3. Streaming error SSE event
            stream_err_resp = await client.post(
                "/api/v1/chat/any-sess-id/stream",
                json={"question": "Pergunta que falha no LLM"},
            )
            assert stream_err_resp.status_code == 200
            assert "text/event-stream" in stream_err_resp.headers["content-type"]

            lines = stream_err_resp.text.strip().split("\n")
            assert lines[0] == "event: error"
            err_payload = json.loads(lines[1].replace("data: ", "").strip())
            assert "Provedor LLM fora do ar" in err_payload["error"]
