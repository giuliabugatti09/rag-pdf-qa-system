"""
Módulo responsável por compor a chain de RAG usando LCEL:
retriever -> formatação de contexto -> prompt final.

"""

from langchain_core.runnables import RunnablePassthrough
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.prompts import ChatPromptTemplate

from src.retrieval.prompt import build_rag_prompt, format_docs_for_context

def build_prompt_chain(retriever: VectorStoreRetriever, lang: str = "pt"):
    
    prompt_template = build_rag_prompt(lang=lang)
    chain = (
        {
            "context": retriever | format_docs_for_context,
            "question": RunnablePassthrough(),
        }
        | prompt_template
    )
    return chain

# Bloco de teste manual
if __name__ == "__main__":
    from src.retrieval.vectorstore import load_vectorstore
    from src.retrieval.retriever import get_retriever

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4, use_mmr=False)

    chain = build_prompt_chain(retriever)

    question = "Qual foi o lucro líquido atribuível aos acionistas em 2023?"
    resultado = chain.invoke(question)

    print("=== Prompt montado automaticamente pela chain ===\n")
    print(resultado.to_string())