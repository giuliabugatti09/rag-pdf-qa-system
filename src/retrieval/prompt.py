"""
Módulo responsável por definir o template de prompt usado para
gerar respostas ancoradas no contexto recuperado (RAG prompt).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document


RAG_PROMPT_TEMPLATE = """Você é um assistente especializado em responder perguntas com base em documentos técnicos e financeiros.

Use EXCLUSIVAMENTE as informações do contexto abaixo para responder à pergunta. Não utilize conhecimento prévio ou externo ao contexto fornecido.

Se a informação necessária não estiver presente no contexto, responda claramente: "Não encontrei essa informação nos documentos fornecidos." Não tente adivinhar ou complementar com suposições.

Ao responder, cite a página de origem de cada informação relevante, no formato (página X).

Contexto:
{context}

Pergunta: {question}

Resposta:"""


def build_rag_prompt() -> ChatPromptTemplate:
    """
    Constrói o template de prompt para a chain de RAG.

    Returns:
        ChatPromptTemplate pronto para ser usado com um LLM,
        esperando as variáveis 'context' e 'question'.
    """
    return ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)


def format_docs_for_context(docs: list[Document]) -> str:
    """
    Formata uma lista de Documents recuperados em uma única string
    de contexto, incluindo o número da página de cada chunk para
    permitir que a LLM cite a fonte corretamente.

    Args:
        docs: lista de Documents retornados pelo retriever.

    Returns:
        String formatada, pronta para ser inserida na variável
        'context' do prompt.
    """
    formatted_chunks = []
    for doc in docs:
        page = doc.metadata.get("page", "desconhecida")
        formatted_chunks.append(f"[Página {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)


# Bloco de teste manual: monta o prompt final e imprime (sem chamar a LLM ainda)
if __name__ == "__main__":
    from src.retrieval.vectorstore import load_vectorstore
    from src.retrieval.retriever import get_retriever

    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4, use_mmr=False)

    question = "Qual foi o lucro líquido atribuível aos acionistas em 2023?"
    docs = retriever.invoke(question)

    context = format_docs_for_context(docs)
    prompt_template = build_rag_prompt()

    final_prompt = prompt_template.format(context=context, question=question)

    print("=== PROMPT FINAL (o que será enviado à LLM) ===\n")
    print(final_prompt)