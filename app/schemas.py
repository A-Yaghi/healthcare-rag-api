from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, examples=[
        "What are WHO ethical principles for AI in health?"
    ])
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    doc_id: str
    title: str
    text: str
    score: float
    source: str = ""
    license: str = ""
    url: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    generation_mode: str
    latency_ms: float
    embedding_model: str
    generator_model: str | None = None
    normalized_question: str | None = None
    corrections: dict[str, str] | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, max_length=40)
    top_k: int | None = Field(default=None, ge=1, le=10)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    generation_mode: str
    latency_ms: float
    embedding_model: str
    generator_model: str | None = None
    search_question: str
    normalized_question: str | None = None
    corrections: dict[str, str] | None = None


class HealthResponse(BaseModel):
    status: str
    app: str
    environment: str
    generation_mode: str
    indexed_chunks: int
    embedding_model: str
    generator_model: str | None = None


class MetricsResponse(BaseModel):
    request_count: int
    avg_latency_ms: float
    p95_latency_ms: float
    last_latency_ms: float | None
