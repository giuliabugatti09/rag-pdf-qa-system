"""
Módulo responsável por fragmentar documentos em chunks menores,
prontos para serem convertidos em embeddings.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Divide uma lista de Documents (ex: páginas de PDF) em chunks menores.

    Args:
        documents: lista de Documents a serem fragmentados (ex: saída do loader).
        chunk_size: tamanho máximo de cada chunk, em caracteres.
        chunk_overlap: quantidade de caracteres repetidos entre chunks
            consecutivos, para preservar contexto em fronteiras
            (ex: cabeçalhos de tabela separados dos valores).

    Returns:
        Lista de Documents fragmentados, preservando os metadados
        originais (fonte, página) em cada chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Ordem de tentativa: quebra parágrafo > linha > frase > palavra.
        # Isso evita cortar uma tabela ou frase no pior lugar possível.
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)
    print(f"Documentos originais: {len(documents)} páginas")
    print(f"Chunks gerados: {len(chunks)}")

    return chunks


# Bloco de teste manual
if __name__ == "__main__":
    from src.ingestion.loader import load_all_pdfs

    pages = load_all_pdfs()
    chunks = split_documents(pages)

    # Inspeciona um chunk do meio do documento (região das tabelas)
    print("\n--- Exemplo de chunk (índice 150) ---")
    print(f"Fonte: {chunks[150].metadata.get('source')}")
    print(f"Página: {chunks[150].metadata.get('page')}")
    print(f"Tamanho: {len(chunks[150].page_content)} caracteres")
    print(f"Conteúdo:\n{chunks[150].page_content}")