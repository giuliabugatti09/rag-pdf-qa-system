"""
Módulo responsável por criar, persistir e consultar o banco de dados
vetorial (ChromaDB) que armazena os embeddings dos chunks.
"""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.retrieval.embeddings import get_embedding_model

PERSIST_DIRECTORY = "vectorstore"
COLLECTION_NAME = "rag_documents"


import time


def build_vectorstore(chunks: list[Document], max_tentativas: int = 5) -> Chroma:
    """
    Cria um novo vector store a partir dos chunks fornecidos, gera os
    embeddings de cada um e persiste tudo em disco.

    Remove qualquer índice existente antes de criar um novo, garantindo
    que a função seja idempotente (chamadas repetidas com os mesmos
    chunks não geram duplicatas).

    Args:
        chunks: lista de Documents fragmentados (saída do chunker).
        max_tentativas: número de tentativas ao remover o índice anterior,
            necessário porque no Windows o SQLite do ChromaDB pode manter
            o arquivo brevemente travado após o objeto anterior ser liberado.
    """
    if Path(PERSIST_DIRECTORY).exists():
        for tentativa in range(max_tentativas):
            try:
                shutil.rmtree(PERSIST_DIRECTORY)
                print(f"Índice anterior removido de: {PERSIST_DIRECTORY}")
                break
            except PermissionError:
                if tentativa == max_tentativas - 1:
                    raise
                time.sleep(1.0)

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIRECTORY,
    )

    print(f"Vector store criado com {len(chunks)} chunks.")
    print(f"Persistido em: {Path(PERSIST_DIRECTORY).resolve()}")
    return vectorstore

def load_vectorstore() -> Chroma:
    """
    Carrega um vector store já existente do disco, sem reprocessar
    os documentos originais. Use isso sempre que só precisar CONSULTAR
    dados já indexados anteriormente.

    Returns:
        Instância do Chroma vector store carregada do disco.
    """
    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )
    return vectorstore


def add_documents_to_vectorstore(vectorstore: Chroma, chunks: list[Document]) -> None:
    """
    Adiciona novos chunks a um vector store já existente, sem apagar
    ou reprocessar o conteúdo já indexado.

    Args:
        vectorstore: instância do Chroma já carregada (ver load_vectorstore).
        chunks: lista de novos Documents (chunks) a serem adicionados.
    """
    vectorstore.add_documents(chunks)
    print(f"{len(chunks)} novos chunks adicionados ao vector store.")


def get_vectorstore_stats(vectorstore: Chroma) -> dict:
    """
    Retorna estatísticas básicas do vector store: total de chunks
    e lista de documentos únicos indexados (pelo nome do arquivo).
    """
    collection = vectorstore._collection
    total_chunks = collection.count()

    resultado = collection.get(include=["metadatas"])
    arquivos = set()
    for metadata in resultado.get("metadatas", []):
        source = metadata.get("source", "")
        arquivos.add(Path(source).name)

    return {
        "total_chunks": total_chunks,
        "total_documentos": len(arquivos),
        "documentos": sorted(arquivos),
    }


# Bloco de teste manual: cria o índice do zero e faz uma busca de teste
if __name__ == "__main__":
    from src.ingestion.loader import load_all_pdfs
    from src.ingestion.chunker import split_documents

    pages = load_all_pdfs()
    chunks = split_documents(pages)
    vectorstore = build_vectorstore(chunks)

    query = "net income attributable to shareholders 2023"
    print(f"\n--- Busca de teste: '{query}' ---\n")
    results = vectorstore.similarity_search(query, k=6)

    for i, doc in enumerate(results, 1):
        print(f"Resultado {i} (página {doc.metadata.get('page')}):")
        print(doc.page_content[:300])
        print()