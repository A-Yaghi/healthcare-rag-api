from __future__ import annotations

from typing import Protocol

from app.rag.documents import Chunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> str: ...


class ExtractiveGenerator:
    """Compose a grounded answer from retrieved chunks (no LLM download)."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the knowledge base "
                f"for: {question}"
            )

        bullets = []
        for chunk in contexts:
            citation = chunk.source or chunk.title
            bullets.append(f"- ({citation}) {chunk.text}")
        joined = "\n".join(bullets)
        return (
            f"Based on the indexed open digital-health sources, here is the most "
            f"relevant evidence for '{question}':\n{joined}"
        )


class FlanT5Generator:
    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=180,
        )
        self.model_name = model_name

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the knowledge base "
                f"for: {question}"
            )

        context_block = "\n".join(
            f"[{c.source or c.title}] {c.text}" for c in contexts
        )
        prompt = (
            "Answer the question using only the context. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = self._pipe(prompt)[0]["generated_text"].strip()
        if not result:
            return ExtractiveGenerator().generate(question, contexts)

        citations = "; ".join(
            sorted({c.source or c.title for c in contexts if c.source or c.title})
        )
        if citations:
            return f"{result}\n\nSources: {citations}"
        return result
