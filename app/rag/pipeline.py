from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from app.config import Settings
from app.rag.documents import knowledge_fingerprint, load_knowledge_chunks
from app.rag.embeddings import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    SentenceTransformerBackend,
)
from app.rag.chat import build_search_question
from app.rag.generator import (
    AnswerGenerator,
    ExtractiveGenerator,
    FlanT5Generator,
    HybridGenerator,
)
from app.rag.store import FaissStore
from app.schemas import ChatMessage, ChatResponse, QueryResponse, SourceChunk


class RAGPipeline:
    def __init__(
        self,
        store: FaissStore,
        embedder: EmbeddingBackend,
        generator: AnswerGenerator,
        generation_mode: str,
        default_top_k: int,
        embedding_model: str,
        generator_model: str | None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.generation_mode = generation_mode
        self.default_top_k = default_top_k
        self.embedding_model = embedding_model
        self.generator_model = generator_model

    @property
    def indexed_chunks(self) -> int:
        return self.store.size

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        k = top_k or self.default_top_k
        query_vec = self.embedder.embed([question])
        hits = self.store.search(query_vec, k)
        contexts = [chunk for chunk, _ in hits]
        answer = self.generator.generate(question, contexts)
        latency_ms = (time.perf_counter() - started) * 1000

        sources = [
            SourceChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                score=round(score, 4),
                source=chunk.source,
                license=chunk.license,
                url=chunk.url,
            )
            for chunk, score in hits
        ]
        return QueryResponse(
            answer=answer,
            sources=sources,
            generation_mode=self.generation_mode,
            latency_ms=round(latency_ms, 2),
            embedding_model=self.embedding_model,
            generator_model=self.generator_model,
        )

    def chat(
        self,
        messages: list[ChatMessage],
        top_k: int | None = None,
    ) -> ChatResponse:
        search_question = build_search_question(messages)
        if not search_question:
            raise ValueError("Chat requires at least one user message")

        latest_user = next(
            m.content.strip()
            for m in reversed(messages)
            if m.role == "user" and m.content.strip()
        )

        started = time.perf_counter()
        k = top_k or self.default_top_k
        query_vec = self.embedder.embed([search_question])
        hits = self.store.search(query_vec, k)
        contexts = [chunk for chunk, _ in hits]
        answer = self.generator.generate(latest_user, contexts)
        latency_ms = (time.perf_counter() - started) * 1000

        sources = [
            SourceChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                text=chunk.text,
                score=round(score, 4),
                source=chunk.source,
                license=chunk.license,
                url=chunk.url,
            )
            for chunk, score in hits
        ]
        return ChatResponse(
            answer=answer,
            sources=sources,
            generation_mode=self.generation_mode,
            latency_ms=round(latency_ms, 2),
            embedding_model=self.embedding_model,
            generator_model=self.generator_model,
            search_question=search_question,
        )


def _build_embedder(settings: Settings, offline: bool) -> EmbeddingBackend:
    if offline:
        return HashEmbeddingBackend()
    return SentenceTransformerBackend(settings.embedding_model)


def _build_generator(settings: Settings, offline: bool) -> AnswerGenerator:
    mode = settings.generation_mode.lower().strip()
    if offline or mode == "retrieve":
        return ExtractiveGenerator()
    if mode == "generate":
        return HybridGenerator(FlanT5Generator(settings.generator_model))
    raise ValueError("GENERATION_MODE must be 'retrieve' or 'generate'")


def build_or_load_store(
    settings: Settings,
    embedder: EmbeddingBackend,
    embedder_name: str,
    rebuild: bool = False,
) -> FaissStore:
    index_dir = Path(settings.index_dir)
    index_file = index_dir / "index.faiss"
    chunks_file = index_dir / "chunks.json"
    meta_file = index_dir / "meta.json"

    knowledge_dir = Path(settings.knowledge_dir)
    fingerprint = knowledge_fingerprint(knowledge_dir)

    if not rebuild and index_file.exists() and chunks_file.exists() and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if (
            meta.get("embedder") == embedder_name
            and meta.get("knowledge_fingerprint") == fingerprint
        ):
            return FaissStore.load(index_dir)

    chunks = load_knowledge_chunks(knowledge_dir)
    vectors = embedder.embed([c.text for c in chunks])
    store = FaissStore.build(np.asarray(vectors, dtype=np.float32), chunks)
    store.save(index_dir)
    meta_file.write_text(
        json.dumps(
            {
                "embedder": embedder_name,
                "knowledge_fingerprint": fingerprint,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return store


def create_pipeline(settings: Settings, offline: bool | None = None) -> RAGPipeline:
    use_offline = settings.offline_embeddings if offline is None else offline
    embedder = _build_embedder(settings, offline=use_offline)
    generator = _build_generator(settings, offline=use_offline)

    requested_mode = settings.generation_mode.lower().strip()
    if use_offline or requested_mode == "retrieve":
        mode = "retrieve"
        generator_model = None
    else:
        mode = "generate"
        generator_model = settings.generator_model

    embedder_name = "hash-offline" if use_offline else settings.embedding_model
    store = build_or_load_store(settings, embedder, embedder_name=embedder_name)
    return RAGPipeline(
        store=store,
        embedder=embedder,
        generator=generator,
        generation_mode=mode,
        default_top_k=settings.top_k,
        embedding_model=embedder_name,
        generator_model=generator_model,
    )
