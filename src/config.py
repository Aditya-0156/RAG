"""
Configuration management for RAG Knowledge Base
Uses ppydantic-settings for type-safe configuration with environment vairables
"""

from pydantic_settings import BaseSettings, SetttingsConfigDict
from pathlib import Path 

class Settings(BaseSettings):

    #API KEYS
    google_api_key: str=""

    #Model Configuration 
    embedding_model: str="models/text-embedding-004"
    llm_model: str ="gemini-2.0-flash-exp"
    chunk_size: int =1000
    chunk_overlap: int =200

    #Paths
    
    data_dir: Path =Path("data")
    raw_data_dir: Path= Path("data/raw")
    processed_data_dir: Path =Path("data/processed")
    chroma_persist_directory: Path= Path("data/chroma_db")

    # Vector Store Settings

    collection_name: str ="rag_documents"
    top_k_results: int =5
    
    #Temperature for the model 
    temperature: float =0.7
    
    #Model configuration to load from .env file 

    model_config= SettingsConfigDict(
        env_file=".env",
         env_file_encoding="utf-8",
        case_sensitive=False)
    
#Global settings instance
settings = Settings()