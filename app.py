from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rag_pipeline import get_rag_instance

app = FastAPI(
    title="LexRetriever",
    description="Production-ready legal retrieval-augmented generation API with hybrid retrieval, reranking, citation enforcement, and metrics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

rag = get_rag_instance()

class QueryRequest(BaseModel):
    question: str

class Citation(BaseModel):
    act: str
    section: str
    source: str
    page: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    latency_ms: float
    retrieval_ms: float
    rerank_ms: float
    generation_ms: float
    cost_usd: float
    request_id: str
    step_logs: list[str] = Field(default_factory=list)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "ready": rag.is_ready(),
        "documents_loaded": rag.document_count,
        "last_step": rag.step_logs[-1] if getattr(rag, "step_logs", None) else None,
        "metrics": rag.metrics.summary(),
    }

@app.post("/query", response_model=QueryResponse)
async def query(payload: QueryRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        result = rag.ask(payload.question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    citations = [
        Citation(
            act=str(item.get("act", "Unknown")),
            section=str(item.get("section", "Unknown")),
            source=str(item.get("source", "Unknown")),
            page=str(item.get("page", "Unknown")),
        )
        for item in result.get("sources", [])
    ]

    return QueryResponse(
        question=payload.question,
        answer=result.get("answer", ""),
        citations=citations,
        latency_ms=result.get("latency_ms", 0.0),
        retrieval_ms=result.get("retrieval_ms", 0.0),
        rerank_ms=result.get("rerank_ms", 0.0),
        generation_ms=result.get("generation_ms", 0.0),
        cost_usd=result.get("cost_usd", 0.0),
        request_id=result.get("request_id", ""),
        step_logs=result.get("step_logs", []),
    )

@app.get("/metrics")
async def metrics():
    return rag.metrics.summary()

@app.get("/")
async def root():
    return RedirectResponse(url="/static/index.html")
