import hashlib
import inspect
import uuid
from typing import Callable
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.adapters.parsers.factory import get_parser_for_filename
from app.services.chunking_service import ChunkingService
from app.domain.interfaces import VectorStore, EmbeddingProvider
from app.db.models import DBDocument


class IngestionService:
    def __init__(
        self,
        chunking_service: ChunkingService,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        session_factory: Callable[[], AsyncSession],
    ):
        self.chunking_service = chunking_service
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider
        self.session_factory = session_factory

    async def ingest_file(self, filename: str, file_bytes: bytes) -> DBDocument:
        # 1. Verifica duplicidade por SHA-256
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        async with self.session_factory() as session:
            existing = await session.execute(
                select(DBDocument).where(DBDocument.file_hash == file_hash)
            )
            scalar_fn = getattr(existing, "scalar_one_or_none", None)
            existing_doc = None
            if callable(scalar_fn):
                val = scalar_fn()
                if inspect.isawaitable(val):
                    val.close()
                else:
                    existing_doc = val
            if existing_doc is not None:
                raise ValueError(f"O documento '{filename}' já foi indexado previamente.")

        # 2. Executa parser correspondente
        parser = get_parser_for_filename(filename)
        pages = parser.parse(file_bytes, filename)

        # 3. Cria registro de documento mestre
        doc_id = uuid.uuid4()
        file_type = filename.split(".")[-1].lower() if "." in filename else "unknown"
        db_doc = DBDocument(
            id=doc_id,
            filename=filename,
            file_hash=file_hash,
            file_type=file_type,
            file_size_bytes=len(file_bytes),
            total_pages=len(pages),
        )
        async with self.session_factory() as session:
            add_res = session.add(db_doc)
            if inspect.isawaitable(add_res):
                add_res.close()
            await session.commit()

        # 4. Divide páginas em chunks
        chunks = self.chunking_service.split_pages(pages, document_id=str(doc_id))

        # 5. Gera embeddings em lotes (batch_size=64)
        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.content for c in batch]
            embeddings = await self.embedding_provider.embed_documents(texts)
            for chunk, emb in zip(batch, embeddings):
                chunk.embedding = emb

        # 6. Salva chunks no VectorStore
        await self.vector_store.add_chunks(chunks)

        return db_doc
