from __future__ import annotations

import re
from typing import Protocol

from app.rag.documents import Chunk


class AnswerGenerator(Protocol):
    def generate(self, question: str, contexts: list[Chunk]) -> str: ...


def _cite(chunk: Chunk) -> str:
    return chunk.source or chunk.title or chunk.doc_id


def _truncate(text: str, max_chars: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def _is_weak_answer(text: str) -> bool:
    cleaned = text.strip()
    if len(cleaned) < 80:
        return True
    words = re.findall(r"[A-Za-z0-9']+", cleaned)
    if len(words) < 14:
        return True
    # Single short clause with no structure is usually a FLAN-T5 under-answer
    if "\n" not in cleaned and len(words) < 25 and "," not in cleaned:
        return True
    return False


def _wants_pros_cons(question: str) -> bool:
    q = question.lower()
    cues = (
        "benefit",
        "risk",
        "negative",
        "positive",
        "pros",
        "cons",
        "advantage",
        "disadvantage",
        "challenge",
        "drawback",
    )
    return any(cue in q for cue in cues)


class ExtractiveGenerator:
    """Compose a grounded answer from retrieved chunks (no LLM download)."""

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        if _wants_pros_cons(question):
            return self._pros_cons_answer(question, contexts)

        bullets = "\n".join(f"- ({_cite(c)}) {_truncate(c.text, 320)}" for c in contexts)
        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        return (
            f"Based on the indexed open digital-health sources, here is the most "
            f"relevant evidence for '{question}':\n{bullets}\n\nSources: {citations}"
        )

    def _pros_cons_answer(self, question: str, contexts: list[Chunk]) -> str:
        benefit_cues = (
            "benefit",
            "improve",
            "assist",
            "support",
            "promise",
            "help",
            "reduce",
            "enable",
        )
        risk_cues = (
            "risk",
            "bias",
            "hallucin",
            "privacy",
            "inaccurate",
            "harm",
            "challenge",
            "limitation",
            "false",
            "threat",
        )

        benefits: list[str] = []
        risks: list[str] = []
        other: list[str] = []

        for chunk in contexts:
            snippet = _truncate(chunk.text, 260)
            line = f"- ({_cite(chunk)}) {snippet}"
            lower = chunk.text.lower()
            has_benefit = any(cue in lower for cue in benefit_cues)
            has_risk = any(cue in lower for cue in risk_cues)
            if has_benefit and not has_risk:
                benefits.append(line)
            elif has_risk and not has_benefit:
                risks.append(line)
            elif has_benefit and has_risk:
                benefits.append(line)
                risks.append(line)
            else:
                other.append(line)

        parts = [
            f"Here is a corpus-grounded summary for '{question}':",
            "",
            "Potential benefits / positive uses:",
        ]
        parts.extend(benefits or other[:1] or ["- Not enough benefit-focused evidence in the top matches."])
        parts.extend(["", "Potential risks / negatives:"])
        parts.extend(risks or other[1:2] or other[:1] or ["- Not enough risk-focused evidence in the top matches."])
        if other and (benefits or risks):
            parts.extend(["", "Additional context:"])
            parts.extend(other[:2])

        citations = "; ".join(sorted({_cite(c) for c in contexts}))
        parts.extend(["", f"Sources: {citations}"])
        return "\n".join(parts)


class FlanT5Generator:
    def __init__(self, model_name: str, max_new_tokens: int = 256) -> None:
        from transformers import pipeline

        self._pipe = pipeline(
            "text2text-generation",
            model=model_name,
            tokenizer=model_name,
            max_new_tokens=max_new_tokens,
        )
        self.model_name = model_name

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return (
                "I could not find relevant information in the indexed corpus "
                f"for: {question}"
            )

        # Keep prompts inside FLAN-T5's practical input budget.
        context_block = "\n".join(
            f"[{_cite(c)}] {_truncate(c.text, 220)}" for c in contexts[:4]
        )

        if _wants_pros_cons(question):
            prompt = (
                "Using only the context, write a clear answer with two sections: "
                "Benefits and Risks/Negatives. Use short bullet points. "
                "If something is missing from the context, say so.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\nAnswer:"
            )
        else:
            prompt = (
                "Using only the context, write a helpful multi-sentence answer. "
                "Include the key points. If the answer is not in the context, "
                "say you do not know.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\nAnswer:"
            )

        result = self._pipe(prompt)[0]["generated_text"].strip()
        return result


class HybridGenerator:
    """FLAN-T5 first; fall back to structured extractive answers when output is weak."""

    def __init__(self, llm: FlanT5Generator) -> None:
        self.llm = llm
        self.extractive = ExtractiveGenerator()

    def generate(self, question: str, contexts: list[Chunk]) -> str:
        if not contexts:
            return self.extractive.generate(question, contexts)

        llm_answer = self.llm.generate(question, contexts)
        citations = "; ".join(sorted({_cite(c) for c in contexts}))

        if _is_weak_answer(llm_answer):
            return self.extractive.generate(question, contexts)

        if "Sources:" not in llm_answer and citations:
            return f"{llm_answer}\n\nSources: {citations}"
        return llm_answer
