from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.metrics import latency_tracker
from app.rag.pipeline import RAGPipeline, create_pipeline
from app.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = create_pipeline(cfg)
        yield
        app.state.pipeline = None

    app = FastAPI(
        title=cfg.app_name,
        version=__version__,
        description=(
            "Production-style RAG API for digital-health Q&A over openly licensed "
            "public sources (WHO, FDA/Health Canada/MHRA, NCI, MedlinePlus). "
            "Authenticate with the X-API-Key header. Not medical advice."
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def chat_ui() -> FileResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(status_code=404, detail="Chat UI not found")
        return FileResponse(index)

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        pipe: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
        return HealthResponse(
            status="ok",
            app=cfg.app_name,
            environment=cfg.environment,
            generation_mode=(
                cfg.generation_mode if pipe is None else pipe.generation_mode
            ),
            indexed_chunks=0 if pipe is None else pipe.indexed_chunks,
            embedding_model=(
                pipe.embedding_model
                if pipe is not None
                else ("hash-offline" if cfg.offline_embeddings else cfg.embedding_model)
            ),
            generator_model=None if pipe is None else pipe.generator_model,
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
    def metrics(_: str = Depends(verify_api_key)) -> MetricsResponse:
        snap = latency_tracker.snapshot()
        return MetricsResponse(**snap)

    @app.post("/v1/query", response_model=QueryResponse, tags=["rag"])
    def query(
        body: QueryRequest,
        request: Request,
        _: str = Depends(verify_api_key),
    ) -> QueryResponse:
        pipe: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
        if pipe is None:
            raise RuntimeError("Pipeline is not initialized")
        result = pipe.query(body.question, top_k=body.top_k)
        latency_tracker.record(result.latency_ms)
        return result

    @app.post("/v1/chat", response_model=ChatResponse, tags=["rag"])
    def chat(
        body: ChatRequest,
        request: Request,
        _: str = Depends(verify_api_key),
    ) -> ChatResponse:
        pipe: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
        if pipe is None:
            raise RuntimeError("Pipeline is not initialized")
        if not any(m.role == "user" for m in body.messages):
            raise HTTPException(status_code=400, detail="At least one user message is required")
        result = pipe.chat(body.messages, top_k=body.top_k)
        latency_tracker.record(result.latency_ms)
        return result

    return app


app = create_app()
