"""
API REST para o sistema RAG, construída com FastAPI.

Expõe um endpoint /query que recebe uma pergunta e retorna a resposta
gerada pelo pipeline RAG, junto com as fontes (arquivo + página) usadas.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from src.rag_application import RAGApplication
from src.api.upload import process_uploaded_pdf


# Estado global da aplicação — inicializado uma vez, no lifespan
rag_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação: carrega o RAG uma única vez
    quando o servidor sobe, e disponibiliza para todas as requisições.
    """
    print("Carregando RAG Application (isso acontece uma única vez)...")
    rag_app_state["rag"] = RAGApplication(k=4, use_mmr=False)
    print("API pronta para receber requisições.")
    yield
    rag_app_state.clear()


app = FastAPI(
    title="RAG PDF Q&A API",
    description="API para busca semântica e Q&A sobre documentos PDF usando RAG.",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Schema de entrada para o endpoint /query."""
    question: str = Field(
        ...,
        min_length=3,
        description="A pergunta a ser respondida com base nos documentos indexados.",
        examples=["Qual foi o lucro líquido atribuível aos acionistas em 2023?"],
    )
    language: str = Field(default="pt", description="Idioma da resposta: 'pt' ou 'en'.")

class SourceReference(BaseModel):
    """Referência de uma fonte usada para gerar a resposta."""
    arquivo: str
    pagina: int


class QueryResponse(BaseModel):
    """Schema de saída do endpoint /query."""
    question: str
    answer: str
    sources: list[SourceReference]


@app.get("/")
def health_check():
    """Endpoint simples para verificar se a API está no ar."""
    return {"status": "ok", "message": "RAG API está funcionando."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    rag = rag_app_state.get("rag")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG Application ainda não foi inicializada.")

    try:
        rag.chain = build_rag_chain(rag.retriever, lang=request.language)
        resultado = rag.perguntar(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

    return QueryResponse(
        question=resultado["pergunta"],
        answer=resultado["resposta"],
        sources=resultado["fontes"],
    )

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Recebe um novo arquivo PDF, processa e adiciona à base de
    conhecimento existente, sem reprocessar documentos já indexados.
    """
    try:
        resultado = await process_uploaded_pdf(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar o upload: {str(e)}")

    return {
        "message": "Documento processado e adicionado com sucesso.",
        **resultado,
    }
from src.retrieval.vectorstore import get_vectorstore_stats

@app.get("/stats")
def stats():
    """Retorna estatísticas da base de conhecimento atual."""
    rag = rag_app_state.get("rag")
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG Application ainda não foi inicializada.")

    vectorstore = rag.retriever.vectorstore
    return get_vectorstore_stats(vectorstore)