"""
Configuration management for RAG Knowledge Base
Uses ppydantic-settings for type-safe configuration with environment vairables
"""

from pydantic_settings import BaseSettings, SetttingsConfigDict
from pathib import Path 

class Settings(BaseSettings):

    #API KEYS

    openai_api-key: str =""

    #Model Configuration 
    embedding_model: str=""
    llm_model: str =""
    chunk_size: int =1000
    chunk_overlap: int =200

    #Paths
    
    data_dir: Path =Path("data")
    raw_data_dir: Path= Path("data/raw")
    processed_data_dir: Path =Path("data/processed")
    chroma_persist_directory: Path= Path("data/chroma_db")