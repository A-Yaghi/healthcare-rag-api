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
from app.rag.documents import Chunk
from app.rag.normalize import QueryNormalizer, build_normalizer_from_chunks
from app.rag.retrieve import extra_search_queries, rerank_hits
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
        normalizer: QueryNormalizer | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.generator = generator
        self.generation_mode = generation_mode
        self.default_top_k = default_top_k
        self.embedding_model = embedding_model
        self.generator_model = generator_model
        self.normalizer = normalizer or build_normalizer_from_chunks(store.chunks)

    @property
    def indexed_chunks(self) -> int:
        return self.store.size

    def _retrieve(self, question: str, top_k: int) -> list[tuple[Chunk, float]]:
        # Over-fetch then rerank so different questions don't collapse to the same docs.
        candidate_k = min(max(top_k * 5, top_k), max(self.store.size, top_k))
        queries = [question, *extra_search_queries(question)]

        merged: dict[str, tuple[Chunk, float]] = {}
        for query in queries:
            query_vec = self.embedder.embed([query])
            for chunk, score in self.store.search(query_vec, candidate_k):
                prev = merged.get(chunk.doc_id)
                if prev is None or score > prev[1]:
                    merged[chunk.doc_id] = (chunk, float(score))

        raw_hits = list(merged.values())
        return rerank_hits(question, raw_hits, top_k=top_k)

    def _normalize_payload(self, original: str, corrected: str, corrections: dict[str, str]):
        changed = corrected.lower() != original.lower() or bool(corrections)
        return (
            corrected if changed else None,
            corrections or None,
        )

    def query(self, question: str, top_k: int | None = None) -> QueryResponse:
        started = time.perf_counter()
        k = top_k or self.default_top_k
        normalized = self.normalizer.normalize(question)
        hits = self._retrieve(normalized.search_text, k)
        contexts = [chunk for chunk, _ in hits]
        answer = self.generator.generate(normalized.corrected, contexts)
        latency_ms = (time.perf_counter() - started) * 1000
        normalized_question, corrections = self._normalize_payload(
            question, normalized.corrected, normalized.corrections
        )

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
            normalized_question=normalized_question,
            corrections=corrections,
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
        norm_search = self.normalizer.normalize(search_question)
        norm_latest = self.normalizer.normalize(latest_user)
        hits = self._retrieve(norm_search.search_text, k)
        contexts = [chunk for chunk, _ in hits]
        answer = self.generator.generate(norm_latest.corrected, contexts)
        latency_ms = (time.perf_counter() - started) * 1000
        normalized_question, corrections = self._normalize_payload(
            latest_user, norm_latest.corrected, norm_latest.corrections
        )

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
            search_question=norm_search.search_text,
            normalized_question=normalized_question,
            corrections=corrections,
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
        normalizer=build_normalizer_from_chunks(store.chunks),
    )
