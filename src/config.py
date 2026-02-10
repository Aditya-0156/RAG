"""
Configuration management for RAG Knowledge Base.
Uses pydantic-settings for type-safe configuration with environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    google_api_key: str = ""

    # Model Configuration
    embedding_model: str = "models/gemini-embedding-001"
    llm_model: str = "models/gemini-flash-latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    chroma_persist_directory: Path = Path("data/chroma_db")

    # Vector Store Settings
    collection_name: str = "rag_documents"
    top_k_results: int = 5

    # Temperature for the model
    temperature: float = 0.7

    # Model configuration to load from .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Global settings instance
settings = Settings()
