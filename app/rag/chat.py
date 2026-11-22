from __future__ import annotations

from app.schemas import ChatMessage


def build_search_question(messages: list[ChatMessage]) -> str:
    """Turn chat history into a retrieval query focused on the latest user turn."""
    if not messages:
        return ""

    latest_user = next(
        (m.content.strip() for m in reversed(messages) if m.role == "user" and m.content.strip()),
        "",
    )
    if not latest_user:
        return ""

    prior_user = [
        m.content.strip()
        for m in messages[:-1]
        if m.role == "user" and m.content.strip()
    ][-2:]

    if not prior_user:
        return latest_user

    prior = " | ".join(prior_user)
    return f"Earlier topics: {prior}. Current question: {latest_user}"
