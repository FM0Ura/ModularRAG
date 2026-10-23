from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime
import uuid


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    file_type: str
    file_size_bytes: int
    total_pages: int
    created_at: datetime


class ChatSessionCreate(BaseModel):
    title: Optional[str] = "Nova Conversa"


class ChatSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime


class CitationSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    filename: str
    page_number: int
    score: float
    snippet: str


class ChatMessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: Optional[List[CitationSchema]] = []
    created_at: datetime


class ChatPromptRequest(BaseModel):
    question: str
