from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RawDocumentPage:
    page_number: int
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    document_id: str
    chunk_index: int
    page_number: int
    content: str
    embedding: Optional[List[float]] = None
    id: Optional[str] = None
    token_count: Optional[int] = None


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    content: str
    score: float
    rank_dense: Optional[int] = None
    rank_sparse: Optional[int] = None


@dataclass
class Citation:
    id: int
    filename: str
    page_number: int
    score: float
    snippet: str


@dataclass
class ChatMessage:
    role: str  # "user" ou "assistant"
    content: str
    citations: List[Citation] = field(default_factory=list)
