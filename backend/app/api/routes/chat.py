import json
from typing import List, Optional
import uuid
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from app.api.schemas import (
    ChatSessionCreate,
    ChatSessionResponse,
    ChatMessageResponse,
    ChatPromptRequest,
)
from app.db.session import AsyncSessionLocal
from app.db.models import DBChatSession, DBChatMessage
from app.core.config import settings
from app.core.factory import get_llm_provider, get_embedding_provider
from app.adapters.vectorstore.pgvector_store import PGVectorStore
from app.adapters.reranker.flashrank_adapter import FlashRankReranker
from app.services.retrieval_service import RetrievalService
from app.services.rag_service import RAGService

router = APIRouter(prefix="/chat", tags=["Chat"])


def get_rag_service() -> RAGService:
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    embed = get_embedding_provider()
    reranker = FlashRankReranker()
    retrieval = RetrievalService(
        vector_store=vstore,
        embedding_provider=embed,
        reranker=reranker,
        dense_top_k=settings.DENSE_TOP_K,
        sparse_top_k=settings.SPARSE_TOP_K,
        final_top_k=settings.RERANKER_TOP_K,
        threshold=settings.RERANKER_THRESHOLD,
    )
    llm = get_llm_provider()
    return RAGService(
        retrieval_service=retrieval,
        llm_provider=llm,
        session_factory=AsyncSessionLocal,
    )


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_chat_session(payload: Optional[ChatSessionCreate] = None):
    title = payload.title if payload and payload.title else "Nova Conversa"
    async with AsyncSessionLocal() as session:
        chat_sess = DBChatSession(title=title)
        session.add(chat_sess)
        await session.commit()
        await session.refresh(chat_sess)
        return chat_sess


@router.get("/sessions", response_model=List[ChatSessionResponse], include_in_schema=True)
@router.get("/sessions/", response_model=List[ChatSessionResponse], include_in_schema=False)
async def list_chat_sessions():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(DBChatSession).order_by(DBChatSession.updated_at.desc()))
        return res.scalars().all()


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
async def get_session_messages(session_id: str):
    try:
        session_uuid = uuid.UUID(session_id)
    except (ValueError, AttributeError):
        session_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(session_id))

    async with AsyncSessionLocal() as session:
        res = await session.execute(
            select(DBChatMessage)
            .where(DBChatMessage.session_id == session_uuid)
            .order_by(DBChatMessage.created_at.asc())
        )
        return res.scalars().all()


@router.post("/{session_id}/stream")
async def chat_stream(session_id: str, payload: ChatPromptRequest):
    rag_service = get_rag_service()

    async def sse_event_generator():
        try:
            async for event in rag_service.stream_response(session_id, payload.question):
                event_type = event.get("type", "message")
                data_json = json.dumps(event, ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data_json}\n\n"
        except Exception as e:
            err_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"event: error\ndata: {err_data}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
