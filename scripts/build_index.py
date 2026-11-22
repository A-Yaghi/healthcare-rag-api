"""Rebuild the FAISS index from markdown knowledge files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402
from app.rag.pipeline import _build_embedder, build_or_load_store  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG FAISS index")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use hash embeddings (no model download)",
    )
    args = parser.parse_args()

    settings = get_settings()
    embedder = _build_embedder(settings, offline=args.offline)
    embedder_name = "hash" if args.offline else settings.embedding_model
    store = build_or_load_store(
        settings,
        embedder,
        embedder_name=embedder_name,
        rebuild=True,
    )
    print(f"Indexed {store.size} chunks into {settings.index_dir}")


if __name__ == "__main__":
    main()
