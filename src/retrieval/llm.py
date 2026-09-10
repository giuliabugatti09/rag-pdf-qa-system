"""
Módulo responsável por configurar o modelo de linguagem (LLM) via Groq
e compor a chain final de RAG: retriever -> prompt -> LLM -> resposta.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever

from src.retrieval.chain import build_prompt_chain

load_dotenv()


def get_llm(temperature: float = 0.1) -> ChatGroq:
    """
    Cria e retorna o cliente do modelo de linguagem via Groq.

    Args:
        temperature: controla a aleatoriedade da geração. Valores baixos
            (perto de 0) tornam a resposta mais determinística e fiel
            ao contexto — importante para reduzir alucinação em RAG.

    Returns:
        Instância configurada de ChatGroq.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY não encontrada. Verifique se o arquivo .env "
            "existe na raiz do projeto e contém a chave."
        )

    llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=temperature,
        api_key=api_key,
    )
    return llm


def build_rag_chain(retriever: VectorStoreRetriever, lang: str = "pt"):
    """
    Monta a chain completa de RAG: pergunta -> contexto + prompt -> LLM -> resposta em texto.

    Args:
        retriever: retriever configurado (ver src/retrieval/retriever.py).
        lang: idioma da resposta gerada ('pt' ou 'en').

    Returns:
        Uma Runnable que recebe uma pergunta (string) e retorna a
        resposta final da LLM já como string simples.
    """
    prompt_chain = build_prompt_chain(retriever, lang=lang)
    llm = get_llm()

    full_chain = prompt_chain | llm | StrOutputParser()

    return full_chain

if __name__ == "__main__":
    from src.retrieval.vectorstore import load_vectorstore
    from src.retrieval.retriever import get_retriever

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4, use_mmr=False)

    rag_chain = build_rag_chain(retriever)

    question = "Qual foi o lucro líquido atribuível aos acionistas em 2023?"
    print(f"Pergunta: {question}\n")

    resposta = rag_chain.invoke(question)
    print(f"Resposta:\n{resposta}")