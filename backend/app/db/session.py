from typing import Optional
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy import text
from app.core.config import settings
from app.db.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)


async def init_db(target_engine: Optional[AsyncEngine] = None) -> None:
    db_engine = target_engine or engine
    async with db_engine.begin() as conn:
        # Ativa extensão pgvector
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        # Cria tabelas se não existirem
        await conn.run_sync(Base.metadata.create_all)
