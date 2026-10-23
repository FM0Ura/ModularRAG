import pytest
from app.domain.models import (
    RawDocumentPage,
    DocumentChunk,
    SearchResult,
    Citation,
    ChatMessage,
)
from app.domain.interfaces import (
    DocumentParser,
    EmbeddingProvider,
    VectorStore,
    Reranker,
    LLMProvider,
)


def test_document_chunk_creation():
    chunk = DocumentChunk(
        document_id="doc-123",
        chunk_index=0,
        page_number=1,
        content="Texto de exemplo",
        embedding=[0.1, 0.2, 0.3],
    )
    assert chunk.document_id == "doc-123"
    assert chunk.page_number == 1
    assert len(chunk.embedding) == 3
    assert chunk.id is None
    assert chunk.token_count is None


def test_search_result_formatting():
    result = SearchResult(
        chunk_id="chk-1",
        document_id="doc-1",
        filename="manual.pdf",
        page_number=3,
        content="Instrução técnica",
        score=0.92,
    )
    assert result.filename == "manual.pdf"
    assert result.score == 0.92
    assert result.rank_dense is None
    assert result.rank_sparse is None


def test_raw_document_page_creation():
    page = RawDocumentPage(
        page_number=1,
        content="Conteúdo da página 1",
    )
    assert page.page_number == 1
    assert page.content == "Conteúdo da página 1"
    assert page.metadata == {}


def test_citation_and_chat_message_creation():
    citation = Citation(
        id=1,
        filename="relatorio.pdf",
        page_number=2,
        score=0.88,
        snippet="Trecho relevante",
    )
    assert citation.id == 1
    assert citation.filename == "relatorio.pdf"

    message = ChatMessage(
        role="assistant",
        content="Aqui está a resposta.",
        citations=[citation],
    )
    assert message.role == "assistant"
    assert len(message.citations) == 1
    assert message.citations[0].snippet == "Trecho relevante"


def test_interfaces_cannot_be_instantiated_directly():
    for interface in [
        DocumentParser,
        EmbeddingProvider,
        VectorStore,
        Reranker,
        LLMProvider,
    ]:
        with pytest.raises(TypeError):
            interface()
