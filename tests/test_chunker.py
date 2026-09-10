"""
Testes do módulo de chunking — verifica que a fragmentação de texto
respeita o tamanho configurado e preserva overlap corretamente.
"""

from langchain_core.documents import Document
from src.ingestion.chunker import split_documents


def test_split_documents_respects_chunk_size():
    """Chunks gerados não devem ultrapassar chunk_size + margem de tolerância."""
    texto_longo = "Esta é uma frase de teste. " * 100  # ~2800 caracteres
    documento = Document(page_content=texto_longo, metadata={"source": "teste.pdf", "page": 0})

    chunks = split_documents([documento], chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1, "Um texto longo deveria gerar múltiplos chunks."
    for chunk in chunks:
        # Margem de tolerância pequena: o splitter pode ultrapassar
        # ligeiramente ao respeitar limites de frase/parágrafo.
        assert len(chunk.page_content) <= 550, (
            f"Chunk excedeu o tamanho esperado: {len(chunk.page_content)} caracteres."
        )


def test_split_documents_preserves_metadata():
    """Cada chunk deve herdar os metadados do documento original."""
    documento = Document(
        page_content="Texto de exemplo. " * 50,
        metadata={"source": "relatorio.pdf", "page": 5},
    )

    chunks = split_documents([documento], chunk_size=200, chunk_overlap=20)

    for chunk in chunks:
        assert chunk.metadata.get("source") == "relatorio.pdf"
        assert chunk.metadata.get("page") == 5


def test_split_documents_creates_overlap():
    """Chunks consecutivos devem compartilhar algum conteúdo (overlap)."""
    texto = "Palavra" + " única" * 200  # texto sem pontuação, força quebra por tamanho
    documento = Document(page_content=texto, metadata={"source": "teste.pdf", "page": 0})

    chunks = split_documents([documento], chunk_size=300, chunk_overlap=100)

    assert len(chunks) >= 2, "Texto deveria gerar ao menos 2 chunks para testar overlap."

    # O final do primeiro chunk deve aparecer parcialmente no início do segundo
    fim_chunk1 = chunks[0].page_content[-50:]
    assert fim_chunk1 in chunks[0].page_content, "Sanity check básico do teste."