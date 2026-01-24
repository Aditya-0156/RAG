"""
Retriever for RAG Knowledge Base.
Finds relevant documents for user queries.
"""

import sys
from pathlib import Path

#Add project root to Python Path
project_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))

from typing import List, Dict, Any, Optional
import logging

from langchain_core.documents import Document

from src.config import settings
from src.vectorstore.chroma_store import VectorStore

#Set up logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class Retriever:
    """
    Retrieves relevant documents for user queries.
    
    Acts as the bridge between user questions and the vector store,
    formatting results for use with the LLM.
    """
    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the retriever.
        
        Args:
            collection_name: Name of the vector store collection
        """
        self.vector_store = VectorStore(collection_name=collection_name)
        self.top_k = settings.top_k_results
        
        logger.info(f"Retriever initialized with top_k={self.top_k}")
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        Args:
            query: User's question or search query
            top_k: Number of documents to retrieve (optional)
            
        Returns:
            List of Document objects with relevant content
        """
        top_k = top_k or self.top_k
        
        # Search vector store
        results = self.vector_store.search(query, top_k=top_k)
        
        # Convert to Document objects
        documents = []
        for result in results:
            doc = Document(
                page_content=result['content'],
                metadata={
                    **result['metadata'],
                    'distance': result['distance']
                }
            )
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents for query: '{query[:50]}...'")
        
        return documents
    
    def retrieve_with_scores(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents with their similarity scores.
        
        Args:
            query: User's question
            top_k: Number of results
            
        Returns:
            List of dicts with 'document' and 'score'
        """
        top_k = top_k or self.top_k
        
        results = self.vector_store.search(query, top_k=top_k)
        
        return [
            {
                'document': Document(
                    page_content=r['content'],
                    metadata=r['metadata']
                ),
                'score': 1 - r['distance']  # Convert distance to similarity score
            }
            for r in results
        ]
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get('source', 'Unknown')
            content = doc.page_content.strip()
            
            context_parts.append(f"[Document {i}] (Source: {source})\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_relevant_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        One-step method: retrieve documents and format as context.
        
        Args:
            query: User's question
            top_k: Number of documents to use
            
        Returns:
            Formatted context string ready for LLM
        """
        documents = self.retrieve(query, top_k=top_k)
        return self.format_context(documents)

if __name__ == "__main__":
    print("Testing Retriever...")
    print("=" * 50)
    
    try:
        # First, add some test documents to vector store
        from src.vectorstore.chroma_store import VectorStore
        
        store = VectorStore(collection_name="test_retriever")
        store.clear_collection()
        
        # Add test data
        test_texts = [
            "Python is a high-level programming language known for readability.",
            "Machine learning uses algorithms to learn patterns from data.",
            "RAG (Retrieval Augmented Generation) combines search with AI generation.",
            "Vector databases enable fast similarity search using embeddings.",
            "LLMs like GPT and Gemini can generate human-like text."
        ]
        
        test_metadatas = [
            {"source": "python_guide.txt"},
            {"source": "ml_basics.txt"},
            {"source": "rag_intro.txt"},
            {"source": "vector_db.txt"},
            {"source": "llm_overview.txt"}
        ]
        
        print("1. Adding test documents to vector store...")
        store.add_texts(test_texts, test_metadatas)
        print("   ✅ Added 5 test documents")
        
        # Test retriever
        print("\n2. Testing retriever...")
        retriever = Retriever(collection_name="test_retriever")
        
        query = "What is RAG and how does it work?"
        print(f"   Query: '{query}'")
        
        # Test retrieve
        docs = retriever.retrieve(query, top_k=3)
        print(f"\n   Retrieved {len(docs)} documents:")
        for i, doc in enumerate(docs, 1):
            print(f"   {i}. {doc.page_content[:60]}...")
            print(f"      Source: {doc.metadata.get('source')}")
        
        # Test formatted context
        print("\n3. Formatted context for LLM:")
        print("-" * 40)
        context = retriever.get_relevant_context(query, top_k=2)
        print(context)
        print("-" * 40)
        
        # Cleanup
        store.delete_collection()
        print("\n✅ All tests passed! (Test collection cleaned up)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
