"""
Testes do módulo de prompt — verifica que a formatação de contexto
inclui corretamente a fonte e a página de cada chunk.
"""

from langchain_core.documents import Document
from src.retrieval.prompt import format_docs_for_context, build_rag_prompt


def test_format_docs_includes_source_and_page():
    """O contexto formatado deve conter o nome do arquivo e o número da página."""
    docs = [
        Document(
            page_content="O lucro líquido foi de R$ 39,9 bilhões.",
            metadata={"source": "data/raw/relatorio.pdf", "page": 8},
        )
    ]

    contexto = format_docs_for_context(docs)

    assert "relatorio.pdf" in contexto, "Nome do arquivo deveria aparecer no contexto formatado."
    assert "8" in contexto, "Número da página deveria aparecer no contexto formatado."
    assert "R$ 39,9 bilhões" in contexto, "Conteúdo original do chunk deve ser preservado."


def test_format_docs_separates_multiple_chunks():
    """Múltiplos chunks devem aparecer separados por um delimitador claro."""
    docs = [
        Document(page_content="Primeiro chunk.", metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Segundo chunk.", metadata={"source": "a.pdf", "page": 2}),
    ]

    contexto = format_docs_for_context(docs)

    assert "Primeiro chunk." in contexto
    assert "Segundo chunk." in contexto
    assert contexto.index("Primeiro chunk.") < contexto.index("Segundo chunk.")


def test_build_rag_prompt_contains_restriction_instruction():
    """O prompt deve conter a instrução de não usar conhecimento externo."""
    prompt = build_rag_prompt(lang="pt")
    prompt_text = prompt.messages[0].prompt.template

    assert "EXCLUSIVAMENTE" in prompt_text or "ONLY" in prompt_text.upper()


def test_build_rag_prompt_respects_language():
    """O prompt em inglês deve conter instruções em inglês, não em português."""
    prompt_pt = build_rag_prompt(lang="pt")
    prompt_en = build_rag_prompt(lang="en")

    texto_pt = prompt_pt.messages[0].prompt.template
    texto_en = prompt_en.messages[0].prompt.template

    assert "português" in texto_pt.lower() or "portugues" in texto_pt.lower()
    assert "english" in texto_en.lower()