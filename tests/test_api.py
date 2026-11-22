from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.documents import load_knowledge_chunks
from app.rag.embeddings import HashEmbeddingBackend
from app.rag.generator import ExtractiveGenerator
from app.rag.pipeline import RAGPipeline, build_or_load_store


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(
        api_key="test-key",
        app_name="Healthcare RAG API Test",
        environment="test",
        generation_mode="retrieve",
        offline_embeddings=True,
        top_k=2,
        index_dir=tmp_path / "index",
        knowledge_dir=ROOT / "data" / "knowledge",
    )


@pytest.fixture()
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["indexed_chunks"] > 0


def test_query_requires_api_key(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "What is RAG?"})
    assert response.status_code == 422


def test_query_rejects_bad_key(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        json={"question": "What is RAG in healthcare?"},
        headers={"X-API-Key": "wrong"},
    )
    assert response.status_code == 401


def test_query_returns_sources(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        json={"question": "How does multimodal fusion help breast imaging?"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1
    assert body["latency_ms"] >= 0
    assert body["generation_mode"] == "retrieve"


def test_metrics(client: TestClient) -> None:
    client.post(
        "/v1/query",
        json={"question": "What is RAG for clinical support?"},
        headers={"X-API-Key": "test-key"},
    )
    response = client.get("/metrics", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["request_count"] >= 1


def test_pipeline_offline(settings: Settings) -> None:
    embedder = HashEmbeddingBackend()
    chunks = load_knowledge_chunks(settings.knowledge_dir)
    assert len(chunks) >= 3
    store = build_or_load_store(
        settings,
        embedder,
        embedder_name="hash",
        rebuild=True,
    )
    pipe = RAGPipeline(
        store=store,
        embedder=embedder,
        generator=ExtractiveGenerator(),
        generation_mode="retrieve",
        default_top_k=2,
    )
    result = pipe.query("Explain RAG safety controls")
    assert result.sources
    assert "RAG" in result.answer or "retrieval" in result.answer.lower()
