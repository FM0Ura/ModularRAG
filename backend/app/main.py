from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, documents, chat
from app.db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa banco e extensões no startup se a URL for válida
    try:
        await init_db()
    except Exception as e:
        print(f"Aviso de inicialização do DB: {e}")
    yield


app = FastAPI(
    title="Modular RAG API",
    description="API assíncrona, desacoplada e modular para RAG corporativo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
