"""
Vector store for RAG Knowledge Base.
Uses ChromaDB to store and search document embeddings.
"""
import sys
from pathlib import Path 

#Add projetc root to Python Path
project_root=Path(__file__).parent.parent.parent
sys.path.insert(0,str(project_root))

from typing import List, Dict, Any, Optional
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.embeddings.embedding_generator import EmbeddingGenerator

#Set up logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)

class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.
    
    ChromaDB stores embeddings and allows fast similarity search
    to find documents related to a query.
    """

    def __init__(self, collection_name: Optional[str]=None):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name for the collection (default from settings)
        """
        self.collection_name=collection_name or settings.collection_name

        #Create persist directory if it does not exist

        persist_dir=Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        #Initialize ChromaDB client wiht persistence
        self.client= chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
            )
        
        #Get or create collection
        self.collection=self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base documents"}
        )
        
        #Initialize Embedding Generator
        self.embedding_generator=EmbeddingGenerator()

        logging.info(f"Vector store initialized with collection:{self.collection_name}")
        logging.info(f"Persist directory: {persist_dir}")
        logging.info(f"Current documnet count: {self.collection.count()}")

    def add_documents(self, documents: List[Dict[str,Any]]) ->int:
        """
        Add documents with embeddings to the vector store.
        
        Args:
            documents: List of dicts with 'content', 'embedding', 'metadata'
            
        Returns:
            Number of documents added
        """
        if not documents:
            logger.warning("No Documents to add")
            return 0
        
        #Prepare Data for ChromaDB
        ids=[]
        embeddings=[]
        contents=[]
        metadatas=[]

        existing_count=self.collection.count()

        for i,doc in enumerate(documents):
            docs_id=f"doc_{existing_count+i}"
            ids.append(docs_id)
            embeddings.append(doc['embedding'])
            contents.append(doc['content'])
            metadatas.append(doc.get('metadata',{}))
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas
        )

        logger.info(f"Added {len(documents)} documents to vector store")
        logger.info(f"Total documents now: {self.collection.count()}")
        return len(documents)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> int:
        """
        Add raw texts to the vector store (generates embeddings automatically).
        
        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts
            
        Returns:
            Number of texts added
        """
        if not texts:
            return 0
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_generator.generate_embeddings_batch(texts)
        
        # Prepare documents
        documents = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            if embedding:  # Skip if embedding failed
                doc = {
                    'content': text,
                    'embedding': embedding,
                    'metadata': metadatas[i] if metadatas else {}
                }
                documents.append(doc)
        
        return self.add_documents(documents)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.
        
        Args:
            query: Search query text
            top_k: Number of results to return (default from settings)
            
        Returns:
            List of dicts with 'content', 'metadata', 'distance'
        """
        top_k = top_k or settings.top_k_results
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                search_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else 0
                })
        
        logger.info(f"Found {len(search_results)} results for query: '{query[:50]}...'")
        
        return search_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dict with collection statistics
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(settings.chroma_persist_directory)
        }
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base documents"}
        )
        logger.info(f"Cleared collection: {self.collection_name}")


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store...")
    print("=" * 50)
    
    try:
        # Initialize vector store
        store = VectorStore(collection_name="test_collection")
        
        # Clear any existing test data
        store.clear_collection()
        
        # Test adding texts
        test_texts = [
            "Python is a programming language known for its simplicity.",
            "Machine learning is a subset of artificial intelligence.",
            "RAG combines retrieval with text generation.",
            "Vector databases store embeddings for similarity search.",
            "ChromaDB is an open-source vector database."
        ]
        
        test_metadatas = [
            {"source": "python_guide.txt", "topic": "programming"},
            {"source": "ml_intro.txt", "topic": "AI"},
            {"source": "rag_overview.txt", "topic": "RAG"},
            {"source": "vector_db.txt", "topic": "databases"},
            {"source": "chromadb_docs.txt", "topic": "databases"}
        ]
        
        print("\n1. Adding documents...")
        added = store.add_texts(test_texts, test_metadatas)
        print(f"   ✅ Added {added} documents")
        
        # Test search
        print("\n2. Testing search...")
        query = "How does RAG work?"
        results = store.search(query, top_k=3)
        
        print(f"   Query: '{query}'")
        print(f"   Results:")
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result['content'][:60]}...")
            print(f"      Distance: {result['distance']:.4f}")
            print(f"      Source: {result['metadata'].get('source', 'unknown')}")
        
        # Show stats
        print("\n3. Collection stats:")
        stats = store.get_collection_stats()
        for key, value in stats.items():
            print(f"   {key}: {value}")
        
        # Cleanup test collection
        store.delete_collection()
        print("\n✅ All tests passed! (Test collection cleaned up)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()