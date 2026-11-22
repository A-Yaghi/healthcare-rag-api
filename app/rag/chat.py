from __future__ import annotations

import re

from app.schemas import ChatMessage

_FOLLOWUP_RE = re.compile(
    r"^(and |also |what about|how about|tell me more|more about|"
    r"compared to that|versus that|vs\.? that)\b",
    re.I,
)

_TOPIC_SHIFT_RE = re.compile(
    r"\b(not just|not only|more generally|in general|generally|"
    r"besides|other than|instead|another topic|forget (that|about)|"
    r"all (x-?rays?|screening)|beyond)\b",
    re.I,
)


def _is_short_followup(text: str) -> bool:
    """Only blend history for clearly dependent short follow-ups."""
    if _TOPIC_SHIFT_RE.search(text):
        return False
    words = text.split()
    if len(words) <= 8:
        return True
    if len(words) <= 12 and _FOLLOWUP_RE.search(text.strip()):
        return True
    return False


def build_search_question(messages: list[ChatMessage]) -> str:
    """Use the latest user question for retrieval.

    Only blend prior turns for short/ambiguous follow-ups so a new full question
    does not keep retrieving the previous topic's documents.
    """
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
    ][-1:]

    if prior_user and _is_short_followup(latest_user):
        return f"{prior_user[0]} :: {latest_user}"
    return latest_user
