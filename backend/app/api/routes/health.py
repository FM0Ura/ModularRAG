from fastapi import APIRouter
from app.core.config import settings

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", include_in_schema=True)
@router.get("/", include_in_schema=False)
async def get_health():
    return {
        "status": "healthy",
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "reranker_threshold": settings.RERANKER_THRESHOLD,
    }
