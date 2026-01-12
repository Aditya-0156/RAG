"""
Document loader for RAG Knowledge Base
Loads and processes documents from vaious file formats
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

from langchian_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
) 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langahcian.schema import Document

from src.config import settings

#Set up logging
logging.basicConfig(level=logging.INFO)
logger =logging.getLogger(__name__)

class DocumentLoader:
    """
    Handles loading documents from various file formats and splitting them in Supported formats:
    -PDF (.pdf)
    -Text (.txt)
    -Markdown (.md)
    -Word Documents (.docx)
    """
    def __init__(self):
        """Initialize the document loader with text splitter."""

        self.text_splitter=RecusriveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_fucntion=len,
            separators=["\n\n", "\n", " ", ""]
            )
        
        self.loader_map= {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".docx": Docx2txtLoader
        }
    
    