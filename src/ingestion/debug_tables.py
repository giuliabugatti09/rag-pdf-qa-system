"""
Script temporário de debug: inspeciona todos os chunks gerados a partir
de uma página específica do PDF original, para verificar como uma
tabela foi fragmentada.
"""

from src.ingestion.loader import load_all_pdfs
from src.ingestion.chunker import split_documents


def find_chunks_by_page(chunks, page_number: int):
    """Imprime todos os chunks originados de uma página específica."""
    found = [c for c in chunks if c.metadata.get("page") == page_number]

    if not found:
        print(f"Nenhum chunk encontrado para a página {page_number}.")
        return

    for i, chunk in enumerate(found):
        print(f"--- chunk {i+1}/{len(found)} da página {page_number} ---")
        print(chunk.page_content)
        print()


if __name__ == "__main__":
    pages = load_all_pdfs()
    chunks = split_documents(pages)
    find_chunks_by_page(chunks, page_number=90)  