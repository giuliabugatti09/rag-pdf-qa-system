"""
Script de avaliação manual: roda uma bateria de perguntas de teste
contra o sistema RAG completo e imprime os resultados para análise,
salvando também em um arquivo para referência futura.
"""

import json
from datetime import datetime
from pathlib import Path

from src.retrieval.vectorstore import load_vectorstore
from src.retrieval.retriever import get_retriever
from src.retrieval.llm import build_rag_chain


# Perguntas de teste cobrindo diferentes cenários
TEST_QUESTIONS = [
    # Perguntas diretas
    "Qual foi o lucro líquido atribuível aos acionistas em 2023?",
    "Qual foi o EBITDA ajustado da Vale em 2023?",
    "Quanto a Vale distribuiu em dividendos em 2023?",

    # Pergunta comparativa
    "Como o lucro líquido de 2023 se compara com o de 2022?",

    # Pergunta sobre tabela específica (operações de cobre)
    "Qual foi a receita líquida das operações de cobre no 4T23?",

    # Fora do escopo do documento (deve dizer que não sabe)
    "Qual é a capital da França?",

    # Relacionado ao tema, mas provavelmente não coberto com detalhe
    "Qual é o salário do CEO da Vale?",
]

def run_evaluation():
    """Executa todas as perguntas de teste e salva os resultados."""
    vectorstore = load_vectorstore()
    retriever = get_retriever(vectorstore, k=4, use_mmr=False)
    rag_chain = build_rag_chain(retriever)

    results = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(TEST_QUESTIONS)}] Pergunta: {question}")
        print('='*60)

        try:
            resposta = rag_chain.invoke(question)
            print(f"Resposta: {resposta}")
        except Exception as e:
            resposta = f"ERRO: {str(e)}"
            print(resposta)

        results.append({
            "pergunta": question,
            "resposta": resposta,
        })

    # Salva os resultados em um arquivo para consulta futura
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"manual_eval_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n\nResultados salvos em: {output_file}")
    return results


if __name__ == "__main__":
    run_evaluation()