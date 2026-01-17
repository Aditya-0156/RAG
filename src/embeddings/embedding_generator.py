"""
Embedding generator for RAG Knowledge Base.
Converts text into vector embeddings using Google Gemini.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List
import logging

import google.generativeai as genai
from langchain_core.documents import Document

#Setup logging
from src.config import settings
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)



class EmbeddingGenerator:
    """
    Generates embeddings for text using Google Gemini API.
    
    Embeddings are numerical representations of text that capture semantic meaning.
    Similar texts will have similar embeddings.
    """
    
    def __init__(self):
        """Initialize the embedding generator with Google API."""
        # Configure Google API
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=settings.google_api_key)
        self.model_name = settings.embedding_model
        
        logger.info(f"Embedding generator initialized with model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")
        
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        
        return result['embedding']
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text)
                embeddings.append(embedding)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Generated embeddings for {i + 1}/{len(texts)} texts")
                    
            except Exception as e:
                logger.error(f"Error generating embedding for text {i}: {e}")
                embeddings.append([])  # Empty embedding for failed text
        
        logger.info(f"Generated {len(embeddings)} embeddings total")
        return embeddings
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for a search query.
        
        Uses 'retrieval_query' task type optimized for search queries.
        
        Args:
            query: The search query
            
        Returns:
            Embedding vector for the query
        """
        if not query or not query.strip():
            raise ValueError("Cannot generate embedding for empty query")
        
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"  # Optimized for queries
        )
        
        return result['embedding']
    
    def embed_documents(self, documents: List[Document]) -> List[dict]:
        """
        Generate embeddings for a list of Document objects.
        
        Args:
            documents: List of Document objects (from document loader)
            
        Returns:
            List of dicts with 'content', 'embedding', and 'metadata'
        """
        results = []
        
        for i, doc in enumerate(documents):
            try:
                embedding = self.generate_embedding(doc.page_content)
                
                results.append({
                    "content": doc.page_content,
                    "embedding": embedding,
                    "metadata": doc.metadata
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Embedded {i + 1}/{len(documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error embedding document {i}: {e}")
                continue
        
        logger.info(f"Successfully embedded {len(results)}/{len(documents)} documents")
        return results


# Convenience function
def get_embedding(text: str) -> List[float]:
    """Quick function to get embedding for a single text."""
    generator = EmbeddingGenerator()
    return generator.generate_embedding(text)


if __name__ == "__main__":
    # Test the embedding generator
    print("Testing Embedding Generator...")
    print("=" * 50)
    
    try:
        generator = EmbeddingGenerator()
        
        # Test single embedding
        test_text = "What is Retrieval Augmented Generation?"
        embedding = generator.generate_embedding(test_text)
        
        print(f"✅ Generated embedding for: '{test_text}'")
        print(f"   Embedding dimensions: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Test query embedding
        query = "How does RAG work?"
        query_embedding = generator.generate_query_embedding(query)
        
        print(f"\n✅ Generated query embedding for: '{query}'")
        print(f"   Embedding dimensions: {len(query_embedding)}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Make sure you have GOOGLE_API_KEY in your .env file")
