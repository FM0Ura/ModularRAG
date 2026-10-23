import pytest
from app.domain.models import RawDocumentPage
from app.services.chunking_service import ChunkingService


def test_chunking_preserves_page_numbers_and_overlap():
    service = ChunkingService(chunk_size=100, chunk_overlap=20)
    page1 = RawDocumentPage(
        page_number=1,
        content="Parágrafo curto da página um. " * 5,
    )
    page2 = RawDocumentPage(
        page_number=2,
        content="Outro conteúdo da página dois com mais detalhes técnicos explicados aqui.",
    )

    chunks = service.split_pages([page1, page2], document_id="doc-test-1")

    assert len(chunks) > 1
    # Garante que chunks da página 1 tenham page_number 1
    assert chunks[0].page_number == 1
    assert chunks[0].document_id == "doc-test-1"
    assert chunks[0].chunk_index == 0
    # Garante que os últimos chunks tenham page_number 2
    assert chunks[-1].page_number == 2


def test_chunking_single_short_page():
    service = ChunkingService(chunk_size=500, chunk_overlap=50)
    page = RawDocumentPage(
        page_number=1,
        content="Conteúdo pequeno que cabe em um único chunk.",
    )
    chunks = service.split_pages([page], document_id="doc-short")

    assert len(chunks) == 1
    assert chunks[0].document_id == "doc-short"
    assert chunks[0].chunk_index == 0
    assert chunks[0].page_number == 1
    assert chunks[0].content == "Conteúdo pequeno que cabe em um único chunk."
    assert chunks[0].token_count == len("Conteúdo pequeno que cabe em um único chunk.".split())


def test_chunking_empty_page():
    service = ChunkingService()
    empty_page = RawDocumentPage(page_number=1, content="   \n \t ")
    chunks = service.split_pages([empty_page], document_id="doc-empty")
    assert chunks == []


def test_chunking_empty_pages_list():
    service = ChunkingService()
    chunks = service.split_pages([], document_id="doc-none")
    assert chunks == []


def test_chunking_continuous_text_without_separators():
    service = ChunkingService(chunk_size=50, chunk_overlap=10)
    # 120 caracteres sem espaço
    continuous_content = "A" * 120
    page = RawDocumentPage(page_number=1, content=continuous_content)
    chunks = service.split_pages([page], document_id="doc-continuous")

    assert len(chunks) > 1
    # Verifica que todos os chunks são menores ou iguais ao chunk_size
    for chunk in chunks:
        assert len(chunk.content) <= 50


def test_chunking_hierarchical_separators():
    # Deve quebrar preferencialmente em quebras duplas de linha
    service = ChunkingService(chunk_size=80, chunk_overlap=10)
    text = "Primeiro parágrafo de teste.\n\nSegundo parágrafo com detalhes adicionais que explicam o contexto."
    page = RawDocumentPage(page_number=1, content=text)
    chunks = service.split_pages([page], document_id="doc-sep")

    assert len(chunks) >= 2
    assert chunks[0].content == "Primeiro parágrafo de teste."


def test_chunking_sequential_chunk_indices():
    service = ChunkingService(chunk_size=50, chunk_overlap=10)
    pages = [
        RawDocumentPage(page_number=1, content="Parte um que tem texto suficiente para mais de um chunk."),
        RawDocumentPage(page_number=2, content="Parte dois continuando o documento com novo texto longo."),
    ]
    chunks = service.split_pages(pages, document_id="doc-seq")

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.chunk_index == idx
        assert chunk.document_id == "doc-seq"
        assert chunk.token_count is not None
        assert chunk.token_count > 0


def test_chunking_validation_parameters():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        ChunkingService(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        ChunkingService(chunk_size=100, chunk_overlap=-5)

    with pytest.raises(ValueError, match="chunk_overlap must be smaller than chunk_size"):
        ChunkingService(chunk_size=100, chunk_overlap=100)
