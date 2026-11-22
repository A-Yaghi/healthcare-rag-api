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

        bullets = "\n".join(f"- ({c.title}) {c.text}" for c in contexts)
        return (
            f"Based on the indexed digital-health documents, here is the most "
            f"relevant evidence for '{question}':\n{bullets}"
        )


class FlanT5Generator:
    def __init__(self, model_name: str) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=160,
        )

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the knowledge base "
                f"for: {question}"
            )

        context_block = "\n".join(c.text for c in contexts)
        prompt = (
            "Answer the question using only the context. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\nAnswer:"
        )
        result = self._pipe(prompt)[0]["generated_text"].strip()
        return result or ExtractiveGenerator().generate(question, contexts)
