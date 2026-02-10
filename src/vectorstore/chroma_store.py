"""
Vector store for RAG Knowledge Base.
Uses ChromaDB to store and search document embeddings.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from typing import List, Dict, Any, Optional
import logging

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import settings
from src.embeddings.embedding_generator import EmbeddingGenerator

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages vector storage and retrieval using ChromaDB.

    ChromaDB stores embeddings and allows fast similarity search
    to find documents related to a query.
    """

    def __init__(self, collection_name: Optional[str] = None):
        """
        Initialize the vector store.

        Args:
            collection_name: Name for the collection (default from settings).
        """
        self.collection_name = collection_name or settings.collection_name

        # Create persist directory if it does not exist
        persist_dir = Path(settings.chroma_persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base documents"},
        )

        # Initialize embedding generator
        self.embedding_generator = EmbeddingGenerator()

        logger.info(f"Vector store initialized with collection: {self.collection_name}")
        logger.info(f"Persist directory: {persist_dir}")
        logger.info(f"Current document count: {self.collection.count()}")

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents with embeddings to the vector store.

        Args:
            documents: List of dicts with 'content', 'embedding', 'metadata'.

        Returns:
            Number of documents added.
        """
        if not documents:
            logger.warning("No documents to add")
            return 0

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        contents = []
        metadatas = []

        existing_count = self.collection.count()

        for i, doc in enumerate(documents):
            doc_id = f"doc_{existing_count + i}"
            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            contents.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(documents)} documents to vector store")
        logger.info(f"Total documents now: {self.collection.count()}")
        return len(documents)

    def add_texts(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> int:
        """
        Add raw texts to the vector store (generates embeddings automatically).

        Args:
            texts: List of text strings.
            metadatas: Optional list of metadata dicts.

        Returns:
            Number of texts added.
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
                    "content": text,
                    "embedding": embedding,
                    "metadata": metadatas[i] if metadatas else {},
                }
                documents.append(doc)

        return self.add_documents(documents)

    def search(
        self, query: str, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for documents similar to the query.

        Args:
            query: Search query text.
            top_k: Number of results to return (default from settings).

        Returns:
            List of dicts with 'content', 'metadata', 'distance'.
        """
        top_k = top_k or settings.top_k_results

        # Generate query embedding
        query_embedding = self.embedding_generator.generate_query_embedding(query)

        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        search_results = []

        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "content": results["documents"][0][i],
                    "metadata": (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    ),
                    "distance": (
                        results["distances"][0][i] if results["distances"] else 0
                    ),
                })

        logger.info(
            f"Found {len(search_results)} results for query: '{query[:50]}...'"
        )

        return search_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.

        Returns:
            Dict with collection statistics.
        """
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_directory": str(settings.chroma_persist_directory),
        }

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "RAG Knowledge Base documents"},
        )
        logger.info(f"Cleared collection: {self.collection_name}")
