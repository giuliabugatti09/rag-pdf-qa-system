"""
Avaliação formal do sistema RAG usando o framework RAGAS, medindo
qualidade de retrieval (context precision/recall) e de geração
(faithfulness, answer relevancy) com LLM-as-judge.
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from src.retrieval.vectorstore import load_vectorstore
from src.retrieval.retriever import get_retriever
from src.retrieval.llm import build_rag_chain, get_llm
from src.retrieval.embeddings import get_embedding_model


# Dataset de avaliação: pergunta + resposta de referência (ground truth)
# extraída manualmente do PDF, com confiança alta.
EVAL_DATASET = [
    {
        "question": "Qual foi o lucro líquido atribuível aos acionistas em 2023?",
        "ground_truth": "O lucro líquido atribuível aos acionistas em 2023 foi de R$ 39,9 bilhões.",
    },
    {
        "question": "Qual foi a receita líquida das operações de cobre no 4T23?",
        "ground_truth": "A receita líquida das operações de cobre no 4T23 foi de US$ 605 milhões.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "ground_truth": (
            "Retrieval-Augmented Generation (RAG) is a method that combines "
            "a pre-trained parametric memory (a seq2seq model) with "
            "non-parametric memory (a dense vector index of documents, "
            "accessed via a neural retriever like DPR)."
        ),
    },
]


def run_ragas_evaluation():
    """Executa o pipeline RAG para cada pergunta do dataset e avalia com RAGAS."""
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4, use_mmr=False)
    rag_chain = build_rag_chain(retriever, lang="pt")

    perguntas, respostas, contextos, ground_truths = [], [], [], []

    for item in EVAL_DATASET:
        pergunta = item["question"]
        print(f"Processando: {pergunta}")

        resposta = rag_chain.invoke(pergunta)
        docs_recuperados = retriever.invoke(pergunta)

        perguntas.append(pergunta)
        respostas.append(resposta)
        contextos.append([doc.page_content for doc in docs_recuperados])
        ground_truths.append(item["ground_truth"])

    # Monta o dataset no formato esperado pelo RAGAS
    dataset = Dataset.from_dict({
        "question": perguntas,
        "answer": respostas,
        "contexts": contextos,
        "ground_truth": ground_truths,
    })

    # RAGAS usa uma LLM como "juiz" para avaliar as métricas — usamos
    # a mesma infraestrutura Groq já configurada no projeto, em vez do
    # padrão (OpenAI), para manter consistência de custo/stack.
    llm_juiz = get_llm(temperature=0)
    embeddings_juiz = get_embedding_model()

    resultado = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_juiz,
        embeddings=embeddings_juiz,
    )

    print("\n=== Resultado da Avaliação RAGAS ===")
    print(resultado)

    # Salva como DataFrame para inspeção detalhada por pergunta
    df = resultado.to_pandas()
    df.to_csv("evaluation_results/ragas_eval.csv", index=False)
    print("\nResultados detalhados salvos em: evaluation_results/ragas_eval.csv")

    return resultado


if __name__ == "__main__":
    run_ragas_evaluation()