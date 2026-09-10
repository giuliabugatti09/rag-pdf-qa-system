"""
Módulo responsável por configurar o retriever — a interface que busca
os chunks mais relevantes no vector store para uma dada pergunta.
"""

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever


def get_retriever(
    vectorstore: Chroma,
    k: int = 4,
    use_mmr: bool = True,
    lambda_mult: float = 0.5,
) -> VectorStoreRetriever:
    """
    Cria um retriever configurado a partir de um vector store existente.

    Args:
        vectorstore: instância do Chroma já populada.
        k: número de chunks a retornar por busca.
        use_mmr: se True, usa Maximal Marginal Relevance para reduzir
            redundância entre os chunks retornados. Se False, usa
            similarity_search puro (só relevância).
        lambda_mult: equilíbrio entre relevância (1.0) e diversidade (0.0)
            quando use_mmr=True. 0.5 é um bom ponto de partida.

    Returns:
        Um VectorStoreRetriever configurado, pronto para ser usado
        em uma chain de RAG.
    """
    search_type = "mmr" if use_mmr else "similarity"

    search_kwargs = {"k": k}
    if use_mmr:
        # fetch_k: quantos candidatos considerar antes de aplicar MMR.
        # Precisa ser maior que k para o MMR ter opções entre as quais escolher diversidade.
        search_kwargs["fetch_k"] = k * 3
        search_kwargs["lambda_mult"] = lambda_mult

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )
    return retriever


# Bloco de teste manual: compara similarity_search puro vs. MMR lado a lado
if __name__ == "__main__":
    from src.retrieval.vectorstore import load_vectorstore

    vectorstore = load_vectorstore()
    query = "net income attributable to shareholders 2023"

    print(f"--- Busca: '{query}' ---\n")

    print("=== Sem MMR (similarity_search puro) ===")
    retriever_simples = get_retriever(vectorstore, k=4, use_mmr=False)
    for i, doc in enumerate(retriever_simples.invoke(query), 1):
        print(f"{i}. (página {doc.metadata.get('page')}): {doc.page_content[:120]}...")

    print("\n=== Com MMR ===")
    retriever_mmr = get_retriever(vectorstore, k=4, use_mmr=True, lambda_mult=0.8)
    for i, doc in enumerate(retriever_mmr.invoke(query), 1):
        print(f"{i}. (página {doc.metadata.get('page')}): {doc.page_content[:120]}...")