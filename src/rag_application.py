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
        """
        Responde a uma pergunta usando o pipeline RAG completo.

        Args:
            pergunta: a pergunta do usuário, em texto livre.

        Returns:
            Um dicionário com a resposta e os metadados das fontes
            usadas (páginas), para permitir rastreabilidade na interface.
        """
        resposta = self.chain.invoke(pergunta)

        # Recupera os documentos-fonte separadamente, para expor
        # as páginas usadas junto com a resposta (útil na API/UI)
        documentos_fonte = self.retriever.invoke(pergunta)
        paginas = sorted(set(
            doc.metadata.get("page") for doc in documentos_fonte
        ))

        return {
            "pergunta": pergunta,
            "resposta": resposta,
            "paginas_fonte": paginas,
        }


# Bloco de teste manual
if __name__ == "__main__":
    app = RAGApplication()

    resultado = app.perguntar("Qual foi o lucro líquido atribuível aos acionistas em 2023?")

    print(f"\nPergunta: {resultado['pergunta']}")
    print(f"Resposta: {resultado['resposta']}")
    print(f"Páginas fonte: {resultado['paginas_fonte']}")