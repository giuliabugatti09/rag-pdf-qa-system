"""
Ponto central da aplicação RAG.

Monta o pipeline completo (vector store -> retriever -> chain) apenas uma vez
na inicialização, e expõe uma função simples para responder perguntas.
Este módulo é o que a API (FastAPI) e a interface (Streamlit) vão usar,
evitando duplicar lógica de negócio em cada interface.
"""

from src.retrieval.vectorstore import load_vectorstore
from src.retrieval.retriever import get_retriever
from src.retrieval.llm import build_rag_chain
from pathlib import Path

class RAGApplication:
    """
    Encapsula o pipeline RAG completo, já inicializado e pronto para uso.

    A inicialização (__init__) é a parte "cara" — carrega o modelo de
    embeddings e o vector store. Deve acontecer uma única vez, na
    subida da aplicação (não a cada pergunta).
    """

    def __init__(self, k: int = 4, use_mmr: bool = False):
        print("Inicializando RAG Application...")

        vectorstore = load_vectorstore()
        self.retriever = get_retriever(vectorstore, k=k, use_mmr=use_mmr)
        self.chain = build_rag_chain(self.retriever)

        print("RAG Application pronta.")

    def perguntar(self, pergunta: str) -> dict:
        resposta = self.chain.invoke(pergunta)
        documentos_fonte = self.retriever.invoke(pergunta)
        fontes = sorted(set(
            (Path(doc.metadata.get("source", "desconhecido")).name, doc.metadata.get("page"))
            for doc in documentos_fonte
        ))

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "fontes": [{"arquivo": arquivo, "pagina": pagina} for arquivo, pagina in fontes],
        }

# Bloco de teste manual
if __name__ == "__main__":
    app = RAGApplication()

    resultado = app.perguntar("Qual foi o lucro líquido atribuível aos acionistas em 2023?")

    print(f"\nPergunta: {resultado['pergunta']}")
    print(f"Resposta: {resultado['resposta']}")
    print(f"Fontes: {resultado['fontes']}")