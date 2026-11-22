from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.rag.documents import load_knowledge_chunks
from app.rag.embeddings import HashEmbeddingBackend
from app.rag.chat import build_search_question
from app.rag.generator import ExtractiveGenerator, _is_weak_answer
from app.rag.normalize import QueryNormalizer, build_normalizer_from_chunks
from app.rag.pipeline import RAGPipeline, build_or_load_store
from app.rag.retrieve import evidence_is_sufficient, is_general_radiation_question
from app.schemas import ChatMessage


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
        json={"question": "What are microcalcifications on a mammogram?"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert len(body["sources"]) >= 1
    assert body["latency_ms"] >= 0
    assert body["generation_mode"] == "retrieve"
    assert body["sources"][0]["source"]
    assert body["sources"][0]["url"]


def test_metrics(client: TestClient) -> None:
    client.post(
        "/v1/query",
        json={"question": "What is RAG for clinical support?"},
        headers={"X-API-Key": "test-key"},
    )
    response = client.get("/metrics", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["request_count"] >= 1


def test_knowledge_has_source_metadata(settings: Settings) -> None:
    chunks = load_knowledge_chunks(settings.knowledge_dir)
    assert len(chunks) >= 5
    assert any(c.source for c in chunks)
    assert any("WHO" in c.source for c in chunks)
    assert any(c.url.startswith("http") for c in chunks)


def test_chat_endpoint(client: TestClient) -> None:
    response = client.post(
        "/v1/chat",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "What are the benefits and negatives of using AI in the healthcare field?",
                }
            ]
        },
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "benefit" in body["answer"].lower() or "risk" in body["answer"].lower()
    assert body["sources"]
    assert body["search_question"]


def test_chat_ui_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Healthcare RAG Chat" in response.text


def test_search_question_uses_history_for_short_followups() -> None:
    question = build_search_question(
        [
            ChatMessage(role="user", content="Tell me about mammograms"),
            ChatMessage(role="assistant", content="A mammogram is an x-ray..."),
            ChatMessage(role="user", content="What about microcalcifications?"),
        ]
    )
    assert "microcalcifications" in question.lower()
    assert "mammograms" in question.lower()


def test_search_question_ignores_history_for_new_full_questions() -> None:
    question = build_search_question(
        [
            ChatMessage(
                role="user",
                content="What are the current regulations for using AI with medical imaging",
            ),
            ChatMessage(role="assistant", content="..."),
            ChatMessage(
                role="user",
                content="Are the ai regulations in the US and the EU different?",
            ),
        ]
    )
    assert question == "Are the ai regulations in the US and the EU different?"
    assert "medical imaging" not in question.lower()


def test_search_question_ignores_history_on_topic_shift() -> None:
    question = build_search_question(
        [
            ChatMessage(role="user", content="Can xray screening be harmful?"),
            ChatMessage(role="assistant", content="..."),
            ChatMessage(
                role="user",
                content="not just breast screening. are all xray screening harmful?",
            ),
        ]
    )
    assert "Can xray screening" not in question
    assert "all xray" in question.lower()


def test_weak_answer_detection() -> None:
    assert _is_weak_answer("protect autonomy")
    assert not _is_weak_answer(
        "AI can improve diagnosis and treatment, but it also introduces risks "
        "such as bias, privacy concerns, and inaccurate outputs that need governance."
    )


def test_off_corpus_question_refuses(settings: Settings) -> None:
    chunks = load_knowledge_chunks(settings.knowledge_dir)[:3]
    answer = ExtractiveGenerator().generate("Is AI research profitable?", chunks)
    assert "don't know" in answer.lower() or "do not know" in answer.lower()


def test_privacy_medical_data_question(settings: Settings) -> None:
    chunks = [
        c
        for c in load_knowledge_chunks(settings.knowledge_dir)
        if "privacy" in c.text.lower() or "data protection" in c.text.lower()
    ][:4]
    answer = ExtractiveGenerator().generate(
        "Can we freely use AI on medical data?",
        chunks,
    )
    assert "don't know" not in answer.lower()
    assert "privacy" in answer.lower() or "constrained" in answer.lower()


def test_mammography_safety_question(settings: Settings) -> None:
    chunks = [
        c
        for c in load_knowledge_chunks(settings.knowledge_dir)
        if "mammogram" in c.text.lower() or "mammography" in c.text.lower()
    ][:4]
    answer = ExtractiveGenerator().generate("Are mammographies safe?", chunks)
    assert "don't know" not in answer.lower()
    assert any(
        term in answer.lower()
        for term in ("mammogram", "radiation", "risk", "benefit", "harm", "screening")
    )


def test_general_xray_question_uses_radiation_docs(settings: Settings) -> None:
    assert is_general_radiation_question("Can xray screening be harmful?")
    assert is_general_radiation_question(
        "not just breast screening. are all xray screening harmful?"
    )
    assert not is_general_radiation_question("Are mammographies safe?")

    mammo_chunks = [
        c
        for c in load_knowledge_chunks(settings.knowledge_dir)
        if "mammogram" in c.text.lower() and "ionizing radiation has" not in c.text.lower()
    ][:3]
    assert not evidence_is_sufficient(
        "not just breast screening. are all xray screening harmful?",
        mammo_chunks,
    )

    radiation_chunks = [
        c
        for c in load_knowledge_chunks(settings.knowledge_dir)
        if "ionizing radiation has" in c.text.lower()
        or "appropriately justified" in c.text.lower()
    ][:3]
    assert radiation_chunks
    for q in (
        "Can xray screening be harmful?",
        "Are all x-ray screening exams harmful?",
    ):
        answer = ExtractiveGenerator().generate(q, radiation_chunks)
        assert "don't know" not in answer.lower()
        assert "breast cancer" not in answer.lower()
        assert any(
            term in answer.lower()
            for term in ("radiation", "ionizing", "risk", "benefit", "justified", "exposure")
        )


def test_extractive_pros_cons(settings: Settings) -> None:
    chunks = [
        c
        for c in load_knowledge_chunks(settings.knowledge_dir)
        if "benefit" in c.text.lower() or "risk" in c.text.lower()
    ][:4]
    answer = ExtractiveGenerator().generate(
        "What are the benefits and negatives of using AI in the healthcare field?",
        chunks,
    )
    assert "positive side" in answer.lower()
    assert "risk side" in answer.lower()
    assert "…" not in answer
    assert "..." not in answer


def test_pipeline_offline(settings: Settings) -> None:
    embedder = HashEmbeddingBackend()
    chunks = load_knowledge_chunks(settings.knowledge_dir)
    assert len(chunks) >= 5
    store = build_or_load_store(
        settings,
        embedder,
        embedder_name="hash-offline",
        rebuild=True,
    )
    pipe = RAGPipeline(
        store=store,
        embedder=embedder,
        generator=ExtractiveGenerator(),
        generation_mode="retrieve",
        default_top_k=2,
        embedding_model="hash-offline",
        generator_model=None,
    )
    result = pipe.query("What are WHO ethical principles for AI in health?")
    assert result.sources
    assert any(s.source for s in result.sources)
    assert "WHO" in result.answer or "autonomy" in result.answer.lower()


def test_query_normalizer_fixes_misspellings(settings: Settings) -> None:
    chunks = load_knowledge_chunks(settings.knowledge_dir)
    normalizer = build_normalizer_from_chunks(chunks)
    result = normalizer.normalize("Are mamografies safe for screening?")
    assert "mammograph" in result.corrected.lower() or "mammogram" in result.corrected.lower()
    assert result.corrections
    assert any("mamograf" in k.lower() for k in result.corrections)


def test_query_normalizer_expands_synonyms() -> None:
    normalizer = QueryNormalizer({"mammogram", "mammography", "screening", "benefits", "risks"})
    result = normalizer.normalize("What are the benefits of mammography screening?")
    assert "benefit" in result.search_text.lower() or "advantage" in result.search_text.lower()
    assert result.search_text != result.corrected or "advantage" in result.search_text.lower()


def test_misspelled_query_returns_corrections(client: TestClient) -> None:
    response = client.post(
        "/v1/query",
        json={"question": "Are mamografies safe?"},
        headers={"X-API-Key": "test-key"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["sources"]
    assert body.get("corrections") or body.get("normalized_question")
    assert "don't know" not in body["answer"].lower()
