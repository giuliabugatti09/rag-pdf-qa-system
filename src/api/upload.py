"""
Lógica de upload e ingestão incremental de novos documentos PDF
na base de conhecimento do RAG, sem reprocessar documentos existentes.
"""

from pathlib import Path
from fastapi import UploadFile

from src.ingestion.loader import load_pdf
from src.ingestion.chunker import split_documents
from src.retrieval.vectorstore import load_vectorstore, add_documents_to_vectorstore

RAW_DATA_DIR = "data/raw"


async def process_uploaded_pdf(file: UploadFile) -> dict:
    """
    Salva um PDF enviado via upload, processa (chunking) e adiciona
    ao vector store existente.

    Args:
        file: arquivo enviado através do endpoint FastAPI.

    Returns:
        Dicionário com informações sobre o processamento (nome do
        arquivo, número de chunks adicionados).
    """
    # Valida que é realmente um PDF antes de processar
    if not file.filename.lower().endswith(".pdf"):
        raise ValueError("Apenas arquivos .pdf são aceitos.")

    # Salva o arquivo em data/raw/, preservando o nome original
    destino = Path(RAW_DATA_DIR) / file.filename
    conteudo = await file.read()
    destino.write_bytes(conteudo)

    # Processa: carrega o PDF recém salvo e fragmenta em chunks
    pages = load_pdf(str(destino))
    chunks = split_documents(pages)

    # Adiciona ao vector store existente, sem apagar o que já havia
    vectorstore = load_vectorstore()
    add_documents_to_vectorstore(vectorstore, chunks)

    return {
        "filename": file.filename,
        "pages_processed": len(pages),
        "chunks_added": len(chunks),
    }