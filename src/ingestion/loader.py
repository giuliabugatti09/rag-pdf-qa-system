"""
Módulo responsável por carregar PDFs e extrair seu conteúdo textual.
Cada PDF é convertido em uma lista de objetos Document (um por página),
já com metadados úteis para rastreabilidade (fonte + página).
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def load_pdf(file_path: str) -> list[Document]:
    """
    Carrega um único PDF e retorna a lista de páginas como Documents.

    Args:
        file_path: caminho para o arquivo PDF.

    Returns:
        Lista de Document, um por página, com metadata contendo
        'source' (nome do arquivo) e 'page' (número da página).
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages


def load_all_pdfs(folder_path: str = "data/raw") -> list[Document]:
    """
    Carrega todos os PDFs de uma pasta.

    Args:
        folder_path: pasta contendo os arquivos .pdf.

    Returns:
        Lista combinada de Documents de todos os PDFs encontrados.
    """
    folder = Path(folder_path)
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"Nenhum PDF encontrado em '{folder_path}'. "
            "Confira se o arquivo está na pasta correta."
        )

    all_documents = []
    for pdf_file in pdf_files:
        print(f"Carregando: {pdf_file.name}")
        documents = load_pdf(str(pdf_file))
        all_documents.extend(documents)

    print(f"\nTotal de páginas carregadas: {len(all_documents)}")
    return all_documents


# Bloco de teste manual — rode este arquivo diretamente para verificar
if __name__ == "__main__":
    docs = load_all_pdfs()

    # Inspeciona a primeira página carregada
    if docs:
        print("\n--- Exemplo da primeira página ---")
        print(f"Fonte: {docs[0].metadata.get('source')}")
        print(f"Página: {docs[0].metadata.get('page')}")
        print(f"Conteúdo (primeiros 300 caracteres):\n{docs[0].page_content[:300]}")