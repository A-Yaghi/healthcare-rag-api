from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_key: str = "dev-secret-change-me"
    app_name: str = "Healthcare RAG API"
    environment: str = "development"

    # retrieve | generate
    generation_mode: str = "retrieve"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # flan-t5-base is a stronger local default; set flan-t5-large for higher quality (slower).
    generator_model: str = "google/flan-t5-base"
    top_k: int = 4

    index_dir: Path = ROOT_DIR / "data" / "index"
    knowledge_dir: Path = ROOT_DIR / "data" / "knowledge"

    # Use lightweight hash embeddings (no model download) for tests/CI
    offline_embeddings: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
