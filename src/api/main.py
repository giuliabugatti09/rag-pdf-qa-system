"""
API REST para o sistema RAG, construída com FastAPI.

Expõe um endpoint /query que recebe uma pergunta e retorna a resposta
gerada pelo pipeline RAG, junto com as páginas-fonte utilizadas.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File
from src.api.upload import process_uploaded_pdf
from pydantic import BaseModel, Field

from src.rag_application import RAGApplication


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
    # Código de limpeza (se necessário) iria aqui, após o yield
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


class QueryResponse(BaseModel):
    """Schema de saída do endpoint /query."""
    question: str
    answer: str
    source_pages: list[int]


@app.get("/")
def health_check():
    """Endpoint simples para verificar se a API está no ar."""
    return {"status": "ok", "message": "RAG API está funcionando."}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Recebe uma pergunta e retorna a resposta gerada pelo pipeline RAG.
    """
    rag = rag_app_state.get("rag")
    if rag is None:
        raise HTTPException(
            status_code=503,
            detail="RAG Application ainda não foi inicializada.",
        )

    try:
        resultado = rag.perguntar(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar a pergunta: {str(e)}")

    return QueryResponse(
        question=resultado["pergunta"],
        answer=resultado["resposta"],
        source_pages=resultado["paginas_fonte"],
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