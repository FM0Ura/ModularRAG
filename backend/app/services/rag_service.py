import inspect
import uuid
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.models import DBChatMessage
from app.domain.interfaces import LLMProvider
from app.domain.models import ChatMessage, Citation, SearchResult
from app.services.retrieval_service import RetrievalService

SYSTEM_PROMPT = """Você é um assistente técnico especialista e confiável.
Responda à pergunta do usuário baseando-se estritamente nas FONTES fornecidas.
Regras:
1. Responda apenas com o que estiver respaldado nas fontes. Não invente detalhes.
2. Cite cada informação com referências no formato [1], [2], correspondente à fonte consultada.
3. Se a informação não constar nas fontes, declare educadamente que os documentos cadastrados não contêm essa informação."""


def _parse_uuid(val: Any) -> uuid.UUID:
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, AttributeError):
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))


def build_prompt_with_sources(question: str, sources: List[SearchResult]) -> Tuple[str, List[Citation]]:
    citations: List[Citation] = []
    if not sources:
        prompt = f"Nenhum documento relevante encontrado.\n\nPERGUNTA: {question}"
        return prompt, citations

    sources_text = "FONTES CONSULTADAS:\n"
    for idx, s in enumerate(sources, start=1):
        citations.append(
            Citation(
                id=idx,
                filename=s.filename,
                page_number=s.page_number,
                score=s.score,
                snippet=s.content[:250] + ("..." if len(s.content) > 250 else ""),
            )
        )
        sources_text += f"\n[FONTE {idx}] (Arquivo: {s.filename} | Página: {s.page_number})\n{s.content}\n"

    prompt = f"{sources_text}\n\nPERGUNTA DO USUÁRIO:\n{question}"
    return prompt, citations


class RAGService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        session_factory: Optional[Callable[[], AsyncSession]] = None,
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.session_factory = session_factory

    async def stream_response(self, session_id: str, question: str) -> AsyncIterator[Dict[str, Any]]:
        # 1. Recupera chunks com busca híbrida + rerank
        sources = await self.retrieval_service.retrieve_context(question)
        prompt, citations = build_prompt_with_sources(question, sources)

        # 2. Emite evento de fontes imediatamente
        yield {
            "type": "sources",
            "citations": [
                {
                    "id": c.id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "score": c.score,
                    "snippet": c.snippet,
                }
                for c in citations
            ],
        }

        # 3. Carrega histórico recente da sessão (limit 6)
        history: List[ChatMessage] = []
        session_uuid = _parse_uuid(session_id)
        if self.session_factory:
            async with self.session_factory() as session:
                stmt = (
                    select(DBChatMessage)
                    .where(DBChatMessage.session_id == session_uuid)
                    .order_by(DBChatMessage.created_at.desc())
                    .limit(6)
                )
                res = await session.execute(stmt)
                scalars_fn = getattr(res, "scalars", None)
                if callable(scalars_fn):
                    scalars_res = scalars_fn()
                    if hasattr(scalars_res, "all") and callable(scalars_res.all):
                        all_res = scalars_res.all()
                        db_msgs = await all_res if inspect.isawaitable(all_res) else all_res
                    elif inspect.isawaitable(scalars_res):
                        scalars_res.close()
                        db_msgs = []
                    else:
                        db_msgs = []
                else:
                    db_msgs = []

                for m in reversed(db_msgs):
                    history.append(ChatMessage(role=m.role, content=m.content))

        # 4. Stream dos tokens do LLM
        stream = self.llm_provider.generate_stream(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            history=history,
        )
        if inspect.iscoroutine(stream):
            stream = await stream

        full_response: List[str] = []
        async for token in stream:
            full_response.append(token)
            yield {"type": "token", "delta": token}

        complete_text = "".join(full_response)

        # 5. Salva mensagens no histórico do PostgreSQL
        assistant_id = uuid.uuid4()
        if self.session_factory:
            async with self.session_factory() as session:
                user_msg = DBChatMessage(
                    id=uuid.uuid4(),
                    session_id=session_uuid,
                    role="user",
                    content=question,
                    citations=[],
                )
                assistant_msg = DBChatMessage(
                    id=assistant_id,
                    session_id=session_uuid,
                    role="assistant",
                    content=complete_text,
                    citations=[
                        {
                            "id": c.id,
                            "filename": c.filename,
                            "page_number": c.page_number,
                            "score": c.score,
                            "snippet": c.snippet,
                        }
                        for c in citations
                    ],
                )
                add_res = session.add_all([user_msg, assistant_msg])
                if inspect.isawaitable(add_res):
                    add_res.close()
                await session.commit()
                msg_id = str(assistant_msg.id)
        else:
            msg_id = str(assistant_id)

        # 6. Evento de conclusão
        yield {"type": "done", "message_id": msg_id, "session_id": session_id}

    # Alias for interface compatibility
    query_stream = stream_response
