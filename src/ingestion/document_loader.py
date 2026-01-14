"""
Document loader for RAG Knowledge Base
Loads and processes documents from vaious file formats
"""

from pathlib import Path
from typing import List, Dict, Any
import logging

from langchain_community.document_loaders import(
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader
) 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

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
    
    def load_document(self, file_path:str) -> List[document]:
        """
        Docstring for load_document
        Load a single doument from a file path
        Args:
            file_path: Path to the document file
        Returns:
            List of Document objects(after splitting into chunks)
        """
        path= Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found at {file_path}")
        
        file_extension= path.suffix.lower()
        if file_extension not in self.loader_map:
            raise ValueError(
                f"Unsupported file format: {file_extension}."
                f"Supported formats:{list(self.loader_map.keys())}"
            )
        
        # Get appropriate loader
        loader_class= self.loader_map[file_extension]
        loader=loader_class(str(path))

        logger.info(f"Loading document: {path.name}")

        #Load document
        documents= loader.load()

        #split into chunks 
        chunks= self.text_splitter.split_documents(documents)

        logger.info(f"document split into {len(chunks)} chunks")

        return chunks
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of all Document chunks from all files
        """
        dir_path = Path(directory_path)
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        all_chunks = []
        
        # Get all files with supported extensions
        for extension in self.loader_map.keys():
            files = list(dir_path.glob(f"*{extension}"))
            
            for file_path in files:
                try:
                    chunks = self.load_document(str(file_path))
                    all_chunks.extend(chunks)
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {e}")
                    continue
        
        logger.info(f"Loaded {len(all_chunks)} total chunks from {directory_path}")
        
        return all_chunks
    
    def get_document_metadata(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        Get metadata summary about loaded documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            Dictionary with metadata statistics
        """
        if not chunks:
            return {"num_chunks": 0, "total_characters": 0}
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        sources = set(chunk.metadata.get("source", "unknown") for chunk in chunks)
        
        return {
            "num_chunks": len(chunks),
            "total_characters": total_chars,
            "average_chunk_size": total_chars // len(chunks) if chunks else 0,
            "num_source_files": len(sources),
            "sources": list(sources)
        }


# Convenience function for quick usage
def load_documents(path: str) -> List[Document]:
    """
    Convenience function to load documents from a file or directory.
    
    Args:
        path: Path to file or directory
        
    Returns:
        List of Document chunks
    """
    loader = DocumentLoader()
    
    path_obj = Path(path)
    
    if path_obj.is_file():
        return loader.load_document(path)
    elif path_obj.is_dir():
        return loader.load_directory(path)
    else:
        raise ValueError(f"Invalid path: {path}")
    

docloader=DocumentLoader()