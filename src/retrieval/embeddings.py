"""
Módulo responsável por gerar embeddings (representações vetoriais)
a partir de texto, usando um modelo local via sentence-transformers.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
) -> HuggingFaceEmbeddings:
    """
    Carrega e retorna o modelo de embeddings.

    Args:
        model_name: nome do modelo no Hugging Face Hub.
            'all-MiniLM-L6-v2' é leve (~90MB), rápido e produz
            vetores de 384 dimensões — bom equilíbrio entre
            qualidade e custo computacional para portfólio.

    Returns:
        Instância de HuggingFaceEmbeddings, pronta para uso com
        ChromaDB ou para gerar embeddings manualmente.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # troque para "cuda" se tiver GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


# Bloco de teste manual
if __name__ == "__main__":
    model = get_embedding_model()

    # Testa a intuição central: frases parecidas devem gerar vetores parecidos
    frase_pt = "lucro líquido da empresa"
    frase_en = "net income attributable to shareholders"
    frase_irrelevante = "receita mensal de queijo suíço na Europa"

    vetor_pt = model.embed_query(frase_pt)
    vetor_en = model.embed_query(frase_en)
    vetor_irrelevante = model.embed_query(frase_irrelevante)

    print(f"Dimensão do vetor: {len(vetor_pt)}")

    # Similaridade de cosseno manual (sem depender de outra lib ainda)
    import math

    def cosine_similarity(v1, v2):
        dot = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(b * b for b in v2))
        return dot / (norm1 * norm2)

    sim_relacionadas = cosine_similarity(vetor_pt, vetor_en)
    sim_nao_relacionadas = cosine_similarity(vetor_pt, vetor_irrelevante)

    print(f"\nSimilaridade entre '{frase_pt}' e '{frase_en}': {sim_relacionadas:.4f}")
    print(f"Similaridade entre '{frase_pt}' e '{frase_irrelevante}': {sim_nao_relacionadas:.4f}")