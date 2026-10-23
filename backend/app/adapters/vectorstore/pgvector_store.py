from typing import List, Callable, Any
import uuid
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession
from app.domain.interfaces import VectorStore
from app.domain.models import DocumentChunk, SearchResult
from app.db.models import DBDocumentChunk, DBDocument


def _to_uuid(val: Any) -> uuid.UUID:
    """Converte string ou valor para UUID de forma segura.
    Se já for UUID, retorna o próprio objeto.
    Se for string UUID válida, converte diretamente.
    Caso contrário, gera UUIDv5 determinístico a partir da string para compatibilidade.
    """
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except (ValueError, AttributeError):
        return uuid.uuid5(uuid.NAMESPACE_DNS, str(val))


class PGVectorStore(VectorStore):
    def __init__(self, session_factory: Callable[[], AsyncSession]):
        self.session_factory = session_factory

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        async with self.session_factory() as session:
            db_chunks = [
                DBDocumentChunk(
                    id=_to_uuid(c.id) if c.id else uuid.uuid4(),
                    document_id=_to_uuid(c.document_id),
                    chunk_index=c.chunk_index,
                    page_number=c.page_number,
                    content=c.content,
                    embedding=c.embedding,
                )
                for c in chunks
            ]
            session.add_all(db_chunks)
            await session.commit()

    async def hybrid_search(
        self,
        query: str,
        query_vector: List[float],
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> List[SearchResult]:
        async with self.session_factory() as session:
            # 1. Busca Densa (Cosseno HNSW)
            dense_stmt = text("""
                SELECT c.id, c.document_id, d.filename, c.page_number, c.content,
                       (1 - (c.embedding <=> CAST(:qvec AS vector))) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.embedding IS NOT NULL
                ORDER BY c.embedding <=> CAST(:qvec AS vector)
                LIMIT :dense_k
            """)
            dense_res = await session.execute(
                dense_stmt,
                {"qvec": str(query_vector), "dense_k": dense_top_k},
            )
            dense_rows = dense_res.fetchall()

            # 2. Busca Esparsa (FTS com to_tsvector)
            sparse_stmt = text("""
                SELECT c.id, c.document_id, d.filename, c.page_number, c.content,
                       ts_rank_cd(to_tsvector('portuguese', c.content), plainto_tsquery('portuguese', :qtext)) AS score
                FROM document_chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE to_tsvector('portuguese', c.content) @@ plainto_tsquery('portuguese', :qtext)
                ORDER BY score DESC
                LIMIT :sparse_k
            """)
            sparse_res = await session.execute(
                sparse_stmt,
                {"qtext": query, "sparse_k": sparse_top_k},
            )
            sparse_rows = sparse_res.fetchall()

            # Mapeia resultados com ranks para o RRF
            results_map = {}
            for rank, row in enumerate(dense_rows):
                chunk_id = str(row[0])
                results_map[chunk_id] = SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(row[1]),
                    filename=row[2],
                    page_number=row[3],
                    content=row[4],
                    score=float(row[5]),
                    rank_dense=rank + 1,
                )

            for rank, row in enumerate(sparse_rows):
                chunk_id = str(row[0])
                if chunk_id in results_map:
                    results_map[chunk_id].rank_sparse = rank + 1
                else:
                    results_map[chunk_id] = SearchResult(
                        chunk_id=chunk_id,
                        document_id=str(row[1]),
                        filename=row[2],
                        page_number=row[3],
                        content=row[4],
                        score=float(row[5]),
                        rank_sparse=rank + 1,
                    )

            return list(results_map.values())

    async def delete_document(self, document_id: str) -> None:
        async with self.session_factory() as session:
            await session.execute(
                delete(DBDocument).where(DBDocument.id == _to_uuid(document_id))
            )
            await session.commit()
