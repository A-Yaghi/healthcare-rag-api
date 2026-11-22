from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request

from app import __version__
from app.auth import verify_api_key
from app.config import Settings, get_settings
from app.metrics import latency_tracker
from app.rag.pipeline import RAGPipeline, create_pipeline
from app.schemas import HealthResponse, MetricsResponse, QueryRequest, QueryResponse


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
            "Production-style RAG API for digital-health Q&A. "
            "Authenticate with the X-API-Key header."
        ),
        lifespan=lifespan,
    )
    app.state.settings = cfg

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(request: Request) -> HealthResponse:
        pipe: RAGPipeline | None = getattr(request.app.state, "pipeline", None)
        return HealthResponse(
            status="ok",
            app=cfg.app_name,
            environment=cfg.environment,
            generation_mode=cfg.generation_mode,
            indexed_chunks=0 if pipe is None else pipe.indexed_chunks,
            embedding_model=(
                "hash-offline" if cfg.offline_embeddings else cfg.embedding_model
            ),
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

    return app


app = create_app()
