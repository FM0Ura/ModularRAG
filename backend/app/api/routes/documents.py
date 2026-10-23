from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from sqlalchemy import select
from app.api.schemas import DocumentResponse
from app.db.session import AsyncSessionLocal
from app.db.models import DBDocument
from app.core.config import settings
from app.core.factory import get_embedding_provider
from app.adapters.vectorstore.pgvector_store import PGVectorStore
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents"])


def get_ingestion_service() -> IngestionService:
    chunking = ChunkingService(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    embed = get_embedding_provider()
    return IngestionService(
        chunking_service=chunking,
        vector_store=vstore,
        embedding_provider=embed,
        session_factory=AsyncSessionLocal,
    )


@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.strip():
        raise HTTPException(status_code=400, detail="Arquivo sem nome fornecido.")

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="O arquivo enviado está vazio (0 bytes).")

    service = get_ingestion_service()
    try:
        db_doc = await service.ingest_file(filename=file.filename, file_bytes=contents)
        return db_doc
    except ValueError as e:
        if "já foi indexado" in str(e):
            raise HTTPException(status_code=409, detail=str(e))
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[DocumentResponse], include_in_schema=True)
@router.get("/", response_model=List[DocumentResponse], include_in_schema=False)
async def list_documents():
    async with AsyncSessionLocal() as session:
        res = await session.execute(select(DBDocument).order_by(DBDocument.created_at.desc()))
        return res.scalars().all()


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    vstore = PGVectorStore(session_factory=AsyncSessionLocal)
    await vstore.delete_document(document_id)
    return {"message": "Documento removido com sucesso.", "document_id": document_id}
